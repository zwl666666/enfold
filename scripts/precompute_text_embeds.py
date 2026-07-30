import hashlib
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from enfold.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from enfold.utils.config_resolvers import register_default_resolvers
from enfold.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _iter_dataset_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for key, value in node.items():
            yield from _iter_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_dataset_nodes(value, f"{path}[{idx}]")


def _collect_dataset_settings(data_cfg: DictConfig):
    dataset_dirs: list[str] = []
    cache_dirs: list[Path] = []
    episode_indices_by_dir: dict[str, list[int]] = {}
    context_lens = set()
    use_instruction_segments = False
    instruction_segment_tracks = set()
    text_embedding_backends = set()
    text_embedding_cache_suffixes = set()

    for node_path, node in _iter_dataset_nodes(data_cfg, path="data"):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue

        cache_dir = node.get("text_embedding_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(
                f"Missing `text_embedding_cache_dir` for dataset node `{node_path}` "
                "(this node defines `dataset_dirs`)."
            )

        for ds in raw_dirs:
            ds_str = str(ds)
            if ds_str not in dataset_dirs:
                dataset_dirs.append(ds_str)
            selected_episodes = _load_episode_indices_from_node(node)
            if selected_episodes is not None:
                episode_indices_by_dir[ds_str] = selected_episodes

        cache_dir_path = Path(str(cache_dir)).expanduser()
        if cache_dir_path not in cache_dirs:
            cache_dirs.append(cache_dir_path)

        context_len = node.get("context_len")
        if context_len is not None:
            context_lens.add(int(context_len))

        if bool(node.get("use_instruction_segments", False)):
            use_instruction_segments = True
            instruction_segment_tracks.add(str(node.get("instruction_segment_track", "default")))

        backend = str(node.get("text_embedding_backend", "cosmos")).strip().lower()
        text_embedding_backends.add(backend)
        suffix = node.get("text_embedding_cache_suffix")
        if suffix is not None and str(suffix).strip():
            text_embedding_cache_suffixes.add(str(suffix).strip())

        logger.info("Discovered dataset node `%s` with %d dataset_dirs.", node_path, len(raw_dirs))

    if len(instruction_segment_tracks) > 1:
        raise ValueError(
            f"Found multiple instruction_segment_track values: {sorted(instruction_segment_tracks)}. "
            "Please keep them consistent."
        )
    instruction_segment_track = next(iter(instruction_segment_tracks), "default")
    return (
        dataset_dirs,
        cache_dirs,
        episode_indices_by_dir,
        context_lens,
        use_instruction_segments,
        instruction_segment_track,
        text_embedding_backends,
        text_embedding_cache_suffixes,
    )


def _resolve_context_len(context_lens: set[int]) -> int:
    if len(context_lens) != 1:
        raise ValueError(
            f"Found multiple context_len values in data config: {sorted(context_lens)}. "
            "Please keep them consistent."
        )
    return next(iter(context_lens))


def _load_episode_indices_from_node(node: DictConfig) -> list[int] | None:
    episode_indices = node.get("episode_indices")
    episode_index_file = node.get("episode_index_file")
    if episode_indices is not None and episode_index_file is not None:
        raise ValueError("Use only one of `episode_indices` or `episode_index_file`, not both.")
    if episode_indices is not None:
        return [int(idx) for idx in episode_indices]
    if episode_index_file is None or not str(episode_index_file).strip():
        return None
    path = Path(str(episode_index_file)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"episode_index_file not found: {path}")
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("episode_indices", payload.get("episodes"))
        if payload is None:
            raise ValueError(f"JSON episode index file must contain a list or an `episode_indices` field: {path}")
        return [int(idx) for idx in payload]
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_unique_prompts(
    dataset_dirs: list[str],
    *,
    use_instruction_segments: bool = False,
    instruction_segment_track: str = "default",
    episode_indices_by_dir: dict[str, list[int]] | None = None,
) -> list[str]:
    prompts: list[str] = []
    seen = set()
    total_task_rows = 0

    for ds_dir in dataset_dirs:
        selected_episodes = None if episode_indices_by_dir is None else episode_indices_by_dir.get(ds_dir)
        if use_instruction_segments:
            info_path = Path(ds_dir) / "meta" / "info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"Missing info file: {info_path}")
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            for segments in info.get("instruction_segments", {}).values():
                for seg in segments:
                    task = str(seg.get("instruction", "")).strip()
                    if not task:
                        continue
                    prompt = DEFAULT_PROMPT.format(task=task)
                    total_task_rows += 1
                    if prompt not in seen:
                        seen.add(prompt)
                        prompts.append(prompt)
            continue

        if selected_episodes is not None:
            episodes_path = Path(ds_dir) / "meta" / "episodes.jsonl"
            if not episodes_path.exists():
                raise FileNotFoundError(f"Missing episodes file: {episodes_path}")
            selected_set = set(int(idx) for idx in selected_episodes)
            with episodes_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    episode_index = int(record["episode_index"])
                    if episode_index not in selected_set:
                        continue
                    for task in record.get("tasks", []):
                        task = str(task)
                        prompt = DEFAULT_PROMPT.format(task=task)
                        total_task_rows += 1
                        if prompt not in seen:
                            seen.add(prompt)
                            prompts.append(prompt)
            continue

        tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_path}")

        with tasks_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "task" not in record:
                    raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                task = str(record["task"])
                prompt = DEFAULT_PROMPT.format(task=task)
                total_task_rows += 1
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)

    logger.info(
        "Loaded %d %s rows from %d datasets, deduplicated to %d prompts.",
        total_task_rows,
        "instruction segment" if use_instruction_segments else "task",
        len(dataset_dirs),
        len(prompts),
    )
    return prompts


def _get_override_prompt(override_instruction: Any) -> str | None:
    if override_instruction is None:
        return None
    task = str(override_instruction).strip()
    if task == "":
        return None
    return DEFAULT_PROMPT.format(task=task)


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _default_cache_suffix(backend: str, model_cfg: DictConfig) -> str:
    backend = str(backend).strip().lower()
    if backend != "cosmos":
        raise ValueError(f"Unsupported text_embedding_backend={backend!r}; expected 'cosmos'.")
    cosmos_cfg = model_cfg.get("video_dit_config", {}).get("cosmos", {})
    model_name = cosmos_cfg.get("model_name")
    if model_name is not None and str(model_name).strip():
        return re.sub(r"[^a-z0-9]+", "", str(model_name).lower())
    checkpoint_path = cosmos_cfg.get("checkpoint_path", model_cfg.get("model_id", "cosmos"))
    return _model_id_to_enc_id(str(checkpoint_path))


def _build_cosmos_text_encoder(model_cfg: DictConfig, device: str, context_len: int):
    from enfold.models.cosmos.enfold import CosmosPromptEncoder, _resolve_cosmos_config
    from omegaconf import OmegaConf

    video_dit_config = model_cfg.get("video_dit_config")
    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if video_dit_config is None:
        video_dit_config = {}
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`model.video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    model_id = str(model_cfg.get("model_id", "cosmos"))
    cosmos_cfg = _resolve_cosmos_config(model_id=model_id, video_dit_config=video_dit_config)
    cosmos_cfg["context_len"] = int(context_len)
    encoder = CosmosPromptEncoder(
        repo_path=str(cosmos_cfg["repo_path"]),
        cosmos_cfg=cosmos_cfg,
        context_len=int(context_len),
        device=device,
    )
    return encoder.eval()


def _encode_prompts(
    prompts: list[str],
    *,
    device: str,
    cosmos_encoder,
) -> tuple[torch.Tensor, torch.Tensor]:
    context, mask = cosmos_encoder.encode_prompts(prompts)
    return context.to(device=device), mask.to(device=device, dtype=torch.bool)


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone --nproc_per_node=%d scripts/precompute_text_embeds.py",
            torch.cuda.device_count(),
        )

    overwrite = _to_bool(cfg.get("overwrite", True))
    model_cfg = cfg.model
    if model_cfg is None:
        raise ValueError("`cfg.model` is required.")
    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")

    (
        dataset_dirs,
        cache_dirs,
        episode_indices_by_dir,
        context_lens,
        use_instruction_segments,
        instruction_segment_track,
        text_embedding_backends,
        text_embedding_cache_suffixes,
    ) = _collect_dataset_settings(cfg.data)
    if not cache_dirs:
        raise ValueError("No `text_embedding_cache_dir` found under `cfg.data`.")

    context_len = _resolve_context_len(context_lens)
    if len(text_embedding_backends) != 1:
        raise ValueError(
            f"Found multiple text_embedding_backend values in data config: {sorted(text_embedding_backends)}. "
            "Please keep them consistent."
        )
    text_embedding_backend = next(iter(text_embedding_backends))
    if text_embedding_backend != "cosmos":
        raise ValueError(
            f"Unsupported text_embedding_backend={text_embedding_backend!r}; expected 'cosmos'."
        )
    if len(text_embedding_cache_suffixes) > 1:
        raise ValueError(
            f"Found multiple text_embedding_cache_suffix values in data config: {sorted(text_embedding_cache_suffixes)}. "
            "Please keep them consistent."
        )
    override_prompt = _get_override_prompt(cfg.get("override_instruction"))
    if override_prompt is not None:
        prompts = [override_prompt]
        logger.info("Using override_instruction; skipping dataset scan and encoding exactly 1 prompt.")
    else:
        if not dataset_dirs:
            raise ValueError("No `dataset_dirs` found under `cfg.data`.")
        prompts = _read_unique_prompts(
            dataset_dirs,
            use_instruction_segments=use_instruction_segments,
            instruction_segment_track=instruction_segment_track,
            episode_indices_by_dir=episode_indices_by_dir,
        )
    if not prompts:
        logger.warning("No prompts found from tasks.jsonl; nothing to do.")
        return

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16
    enc_id = (
        next(iter(text_embedding_cache_suffixes))
        if text_embedding_cache_suffixes
        else _default_cache_suffix(text_embedding_backend, model_cfg)
    )

    logger.info(
        "Preparing Cosmos text encoder device=%s dtype=%s context_len=%d overwrite=%s cache_suffix=%s",
        device,
        torch_dtype,
        context_len,
        overwrite,
        enc_id,
    )

    cosmos_encoder = _build_cosmos_text_encoder(
        model_cfg=model_cfg,
        device=device,
        context_len=context_len,
    )

    stats = {
        str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0}
        for cache_dir in cache_dirs
    }

    prompts = prompts[rank::world_size] if is_distributed else prompts

    if not overwrite:
        fully_cached_local = 0
        prompts_to_encode: list[str] = []
        for prompt in prompts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            fully_cached = True
            for cache_dir in cache_dirs:
                cache_path = cache_dir / filename
                if not cache_path.exists():
                    fully_cached = False
                    break
            if fully_cached:
                fully_cached_local += 1
                for cache_dir in cache_dirs:
                    stats[str(cache_dir)]["skip"] += 1
            else:
                prompts_to_encode.append(prompt)

        prompts = prompts_to_encode

        fully_cached_global = fully_cached_local
        to_encode_global = len(prompts)
        if is_distributed:
            reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
            count_tensor = torch.tensor([fully_cached_local, len(prompts)], device=reduce_device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            fully_cached_global = int(count_tensor[0].item())
            to_encode_global = int(count_tensor[1].item())

        if (not is_distributed) or rank == 0:
            logger.info(
                "overwrite=false: fully cached prompts=%d, prompts to encode=%d",
                fully_cached_global,
                to_encode_global,
            )

    logger.info("Writing caches to %d directories.", len(cache_dirs))
    prompts_encoded_local = len(prompts)
    prompts_encoded_global = prompts_encoded_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([prompts_encoded_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor.item())

    over_length_prompts = 0
    with tqdm(
        total=len(prompts),
        desc=f"Encoding prompts (rank {rank}/{world_size})" if is_distributed else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                context, mask = _encode_prompts(
                    batch_prompts,
                    device=device,
                    cosmos_encoder=cosmos_encoder,
                )
                over_length_prompts += int(mask.all(dim=1).sum().item())

                for i, prompt in enumerate(batch_prompts):
                    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    payload = {
                        "context": context_i,
                        "mask": mask_i,
                    }

                    for cache_dir in cache_dirs:
                        cache_path = cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                        key = str(cache_dir)
                        if cache_path.exists() and not overwrite:
                            stats[key]["skip"] += 1
                            continue

                        if cache_path.exists():
                            stats[key]["overwrite"] += 1
                        else:
                            stats[key]["new"] += 1

                        _atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    over_length_global = over_length_prompts
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_prompts], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length_global = int(over_tensor.item())

        counts_tensor = torch.tensor(
            [
                [stats[str(cache_dir)]["new"], stats[str(cache_dir)]["overwrite"], stats[str(cache_dir)]["skip"]]
                for cache_dir in cache_dirs
            ],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            for idx, cache_dir in enumerate(cache_dirs):
                key = str(cache_dir)
                stats[key]["new"] = int(counts_tensor[idx, 0].item())
                stats[key]["overwrite"] = int(counts_tensor[idx, 1].item())
                stats[key]["skip"] = int(counts_tensor[idx, 2].item())

    if (not is_distributed) or rank == 0:
        logger.info("Finished precomputing text embeddings.")
        logger.info(
            "Over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
            context_len,
            over_length_global,
            prompts_encoded_global,
        )
        for cache_dir in cache_dirs:
            key = str(cache_dir)
            logger.info(
                "Cache dir: %s | new=%d overwrite=%d skip=%d",
                key,
                stats[key]["new"],
                stats[key]["overwrite"],
                stats[key]["skip"],
            )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
