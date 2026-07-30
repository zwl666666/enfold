import argparse
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_HYDRA_CONFIG_NAME = "train"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _resolve_path(container: Any, path: str) -> Any:
    current = container
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _resolve_interpolation(value: Any, *roots: dict[str, Any]) -> Any:
    if not _is_unresolved_interpolation(value):
        return value
    text = str(value).strip()
    if not (text.startswith("${") and text.endswith("}")):
        return value
    expr = text[2:-1]
    candidate_paths = [expr]
    if expr.startswith("model."):
        candidate_paths.append(expr.split(".", 1)[1])
    for root in roots:
        for path in candidate_paths:
            try:
                resolved = _resolve_path(root, path)
            except KeyError:
                continue
            if not _is_unresolved_interpolation(resolved):
                return resolved
    return value


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _interp_depth_stack(tensors: list[torch.Tensor], target_layers: int) -> list[torch.Tensor]:
    if len(tensors) == 0:
        raise ValueError("Cannot depth-interpolate an empty tensor list.")
    if target_layers <= 0:
        raise ValueError(f"`target_layers` must be positive, got {target_layers}")

    src_layers = len(tensors)
    if src_layers == target_layers:
        return [tensor.clone() for tensor in tensors]
    if src_layers == 1:
        return [tensors[0].clone() for _ in range(target_layers)]

    stacked = torch.stack([tensor.to(torch.float32) for tensor in tensors], dim=0)
    out: list[torch.Tensor] = []
    if target_layers == 1:
        positions = [0.5 * float(src_layers - 1)]
    else:
        positions = [i * float(src_layers - 1) / float(target_layers - 1) for i in range(target_layers)]

    for pos in positions:
        left = int(pos)
        right = min(left + 1, src_layers - 1)
        alpha = float(pos - left)
        if left == right:
            value = stacked[left]
        else:
            value = stacked[left] * (1.0 - alpha) + stacked[right] * alpha
        out.append(value.to(dtype=tensors[0].dtype))
    return out


def _require_int_config(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return int(value)


def _require_float_config(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return float(value)


def _extract_model_sections(root: dict[str, Any], *, source_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_root = root.get("model", root)
    if not isinstance(model_root, dict):
        raise ValueError(f"`{source_name}` model root must be dict-like.")

    if "video_dit_config" not in model_root:
        raise ValueError(f"`{source_name}` must contain `video_dit_config` either at top level or under `model`.")

    action_cfg_key = "action_head_config" if "action_head_config" in model_root else "action_dit_config"
    if action_cfg_key not in model_root:
        raise ValueError(
            f"`{source_name}` must contain `action_head_config` or `action_dit_config` either at top level or under `model`."
        )

    video_cfg = dict(model_root["video_dit_config"])
    action_cfg = dict(model_root[action_cfg_key])

    for key in ["hidden_dim", "action_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim", "text_dim", "freq_dim", "eps"]:
        if key in action_cfg:
            action_cfg[key] = _resolve_interpolation(action_cfg[key], root, model_root, action_cfg, video_cfg)

    if _is_unresolved_interpolation(action_cfg.get("action_dim")):
        print(
            "[WARN] `action_dim` is still unresolved after config composition; "
            "defaulting to 7 for preprocessing."
        )
        action_cfg["action_dim"] = 7

    return video_cfg, action_cfg, model_root


def _load_model_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    cfg = OmegaConf.load(str(path))
    root = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(root, dict):
        raise ValueError(f"`{path}` must resolve to a dict-like config.")
    video_cfg, action_cfg, model_root = _extract_model_sections(root, source_name=str(path))
    return video_cfg, action_cfg, model_root, cfg


def _load_hydra_composed_config(
    *,
    config_dir: Path,
    task: str,
    overrides: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    from hydra import compose, initialize_config_dir

    compose_overrides = [f"task={task}"] + list(overrides)
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.resolve())):
        cfg = compose(config_name=DEFAULT_HYDRA_CONFIG_NAME, overrides=compose_overrides)
    root = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(root, dict):
        raise ValueError("Hydra composed config must resolve to a dict-like object.")
    video_cfg, action_cfg, model_root = _extract_model_sections(
        root,
        source_name=f"Hydra config `{DEFAULT_HYDRA_CONFIG_NAME}` with task `{task}`",
    )
    return video_cfg, action_cfg, model_root, cfg


def _find_matching_source_keys(state: dict[str, torch.Tensor], prefix: str) -> list[str]:
    return sorted(key for key in state.keys() if key.startswith(prefix))


def _get_depth_resampled_src(
    *,
    state: dict[str, torch.Tensor],
    source_prefix: str,
    target_layers: int,
) -> list[torch.Tensor]:
    src_keys = _find_matching_source_keys(state, source_prefix)
    if not src_keys:
        raise KeyError(f"No source keys found for prefix `{source_prefix}`")
    tensors = [state[key] for key in src_keys]
    return _interp_depth_stack(tensors, target_layers)


def _broadcast_head_norm(src: torch.Tensor, *, target_num_heads: int, target_head_dim: int) -> torch.Tensor:
    if src.ndim != 1:
        raise ValueError(f"Expected 1D head norm tensor, got shape {tuple(src.shape)}")
    per_head = _resize_tensor_to_shape(src, (target_head_dim,))
    return per_head.repeat(target_num_heads)


def _select_adaln_weight_key(state: dict[str, torch.Tensor], prefix: str) -> str:
    candidates = []
    for key, value in state.items():
        if key.startswith(prefix) and key.endswith(".weight") and value.ndim == 2 and value.shape[0] % 3 == 0:
            candidates.append(key)
    if not candidates:
        raise KeyError(f"No AdaLN weight found for prefix `{prefix}`")
    candidates.sort()
    return candidates[-1]


def _build_time_projection_init(
    *,
    teacher_state: dict[str, torch.Tensor],
    teacher_layers: int,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    self_key = _select_adaln_weight_key(teacher_state, "blocks.0.adaln_modulation_self_attn")
    mlp_key = _select_adaln_weight_key(teacher_state, "blocks.0.adaln_modulation_mlp")
    self_suffix = self_key.split("blocks.0.", 1)[1]
    mlp_suffix = mlp_key.split("blocks.0.", 1)[1]

    self_depth = _interp_depth_stack(
        [teacher_state[f"blocks.{i}.{self_suffix}"] for i in range(teacher_layers)],
        1,
    )[0]
    mlp_depth = _interp_depth_stack(
        [teacher_state[f"blocks.{i}.{mlp_suffix}"] for i in range(teacher_layers)],
        1,
    )[0]
    combined = torch.cat([self_depth, mlp_depth], dim=0)
    return _resize_tensor_to_shape(combined, target_shape)


def _zero_like(target: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(target, memory_format=torch.contiguous_format)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ActionDiT backbone weights from Cosmos-Predict2.5 teacher and save as .pt payload."
    )
    parser.add_argument("--model-config", help="Legacy path to model yaml or composed config yaml.")
    parser.add_argument("--config-dir", default="configs", help="Hydra config directory.")
    parser.add_argument("--task", help="Hydra task override used to compose the final config.")
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override, repeatable.")
    parser.add_argument("--output", required=True, help="Output .pt path for preprocessed ActionDiT backbone.")
    parser.add_argument("--device", default="cuda", help="Device for loading model and preprocessing.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(dv/da) when the last dimension is resized (true/false). Default: true.",
    )
    args = parser.parse_args()

    from enfold.models.action_dit import ActionDiT
    from enfold.models.cosmos.enfold import _load_cosmos_teacher_model

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)
    torch_dtype = _parse_dtype(args.dtype)

    use_hydra = args.task is not None
    if use_hydra:
        config_source = f"{Path(args.config_dir).resolve()}::{DEFAULT_HYDRA_CONFIG_NAME} task={args.task}"
        video_cfg, action_cfg, model_root, _cfg = _load_hydra_composed_config(
            config_dir=Path(args.config_dir),
            task=str(args.task),
            overrides=list(args.override),
        )
    else:
        if not args.model_config:
            raise ValueError(
                "Provide either `--model-config` or `--task` "
                "(optionally with `--config-dir` and repeated `--override`)."
            )
        model_config_path = Path(args.model_config)
        config_source = str(model_config_path)
        video_cfg, action_cfg, model_root, _cfg = _load_model_config(model_config_path)

    int_fields = ["hidden_dim", "action_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim", "text_dim", "freq_dim"]
    for key in int_fields:
        action_cfg[key] = _require_int_config(action_cfg, key)
    action_cfg["eps"] = _require_float_config(action_cfg, "eps")

    print(
        "[INFO] Resolved concat-conditioned ActionDiT config: "
        f"action_dim={action_cfg['action_dim']}, "
        f"hidden_dim={action_cfg['hidden_dim']}, "
        f"ffn_dim={action_cfg['ffn_dim']}, "
        f"num_layers={action_cfg['num_layers']}, "
        f"num_heads={action_cfg['num_heads']}, "
        f"attn_head_dim={action_cfg['attn_head_dim']}, "
        f"text_dim={action_cfg['text_dim']}, "
        f"freq_dim={action_cfg['freq_dim']}, "
        f"eps={action_cfg['eps']}"
    )

    model_id = model_root.get("model_id")
    if model_id is None:
        raise ValueError("`model_id` is required to load the Cosmos teacher model.")

    print(
        f"[INFO] Loaded model config from {config_source}. "
        f"Preprocessing Cosmos ActionDiT backbone with dtype={torch_dtype} on device={args.device}, "
        f"apply_alpha_scaling={apply_alpha_scaling}."
    )

    teacher_model, _, _ = _load_cosmos_teacher_model(
        device=args.device,
        model_id=str(model_id),
        video_dit_config=video_cfg,
    )
    teacher_net = getattr(teacher_model, "net", teacher_model)
    if not hasattr(teacher_net, "blocks"):
        raise ValueError("Cosmos teacher is missing `.net.blocks`; cannot preprocess ActionDiT backbone.")
    teacher_layers = int(len(teacher_net.blocks))
    teacher_state = teacher_net.state_dict()

    action_expert = ActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)
    action_state = action_expert.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())
    backbone_state_dict = {
        key: value.detach().to(device="cpu").contiguous() for key, value in action_state.items() if key in backbone_keys
    }

    target_layers = int(action_cfg["num_layers"])
    teacher_num_heads = int(getattr(teacher_net, "num_heads"))
    teacher_head_dim = int(getattr(teacher_net, "model_channels")) // teacher_num_heads
    target_num_heads = int(action_cfg["num_heads"])
    target_head_dim = int(action_cfg["attn_head_dim"])

    direct_block_mappings = {
        "self_attn.q.weight": "self_attn.q_proj.weight",
        "self_attn.k.weight": "self_attn.k_proj.weight",
        "self_attn.v.weight": "self_attn.v_proj.weight",
        "self_attn.o.weight": "self_attn.output_proj.weight",
        "ffn.0.weight": "mlp.layer1.weight",
        "ffn.2.weight": "mlp.layer2.weight",
    }

    depth_cache: dict[str, list[torch.Tensor]] = {}

    def get_depth_values(source_suffix: str) -> list[torch.Tensor]:
        if source_suffix not in depth_cache:
            depth_cache[source_suffix] = _interp_depth_stack(
                [teacher_state[f"blocks.{i}.{source_suffix}"] for i in range(teacher_layers)],
                target_layers,
            )
        return depth_cache[source_suffix]

    copied = 0
    resized = 0
    depth_interpolated = 0
    zero_filled = 0
    random_kept = 0

    for key in sorted(backbone_keys):
        target = backbone_state_dict[key]

        if key == "time_projection.1.weight":
            value = _build_time_projection_init(
                teacher_state=teacher_state,
                teacher_layers=teacher_layers,
                target_shape=tuple(target.shape),
            )
            if tuple(value.shape) != tuple(target.shape):
                value = _resize_tensor_to_shape(value, tuple(target.shape))
            backbone_state_dict[key] = value.to(dtype=target.dtype, device="cpu").contiguous()
            resized += 1
            continue

        if key == "time_projection.1.bias":
            backbone_state_dict[key] = _zero_like(target)
            zero_filled += 1
            continue

        if key.startswith("blocks."):
            parts = key.split(".")
            layer_idx = int(parts[1])
            suffix = ".".join(parts[2:])

            if suffix in direct_block_mappings:
                src = get_depth_values(direct_block_mappings[suffix])[layer_idx]
                value = _resize_tensor_to_shape(src, tuple(target.shape))
                if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                    alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                    value = value.to(torch.float32) * alpha
                if teacher_layers != target_layers:
                    depth_interpolated += 1
                if tuple(src.shape) != tuple(target.shape):
                    resized += 1
                else:
                    copied += 1
                backbone_state_dict[key] = value.to(dtype=target.dtype, device="cpu").contiguous()
                continue

            if suffix == "self_attn.norm_q.weight":
                src = get_depth_values("self_attn.q_norm.weight")[layer_idx]
                value = _broadcast_head_norm(src, target_num_heads=target_num_heads, target_head_dim=target_head_dim)
                backbone_state_dict[key] = _resize_tensor_to_shape(value, tuple(target.shape)).to(
                    dtype=target.dtype, device="cpu"
                ).contiguous()
                depth_interpolated += int(teacher_layers != target_layers)
                resized += 1
                continue

            if suffix == "self_attn.norm_k.weight":
                src = get_depth_values("self_attn.k_norm.weight")[layer_idx]
                value = _broadcast_head_norm(src, target_num_heads=target_num_heads, target_head_dim=target_head_dim)
                backbone_state_dict[key] = _resize_tensor_to_shape(value, tuple(target.shape)).to(
                    dtype=target.dtype, device="cpu"
                ).contiguous()
                depth_interpolated += int(teacher_layers != target_layers)
                resized += 1
                continue


            if suffix.endswith(".bias") or suffix == "modulation":
                backbone_state_dict[key] = _zero_like(target)
                zero_filled += 1
                continue

        random_kept += 1

    payload = {
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "depth_interpolation": "linear_align_endpoints",
            "tensor_resize": "sequential_1d_linear_align_corners_true",
            "teacher_backend": "cosmos_predict2.5",
            "teacher_num_layers": teacher_layers,
            "teacher_num_heads": teacher_num_heads,
            "teacher_head_dim": teacher_head_dim,
            "non_isomorphic_init": {
                "random_kept_prefixes": ["text_embedding.", "time_embedding."],
                "zero_filled_suffixes": ["*.bias", "blocks.*.modulation", "time_projection.1.bias"],
                "shared_time_projection_from_teacher": True,
            },
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        },
    }
    torch.save(payload, str(output_path))

    skipped = len(action_state) - len(backbone_keys)
    print(
        "[INFO] Saved ActionDiT backbone payload to "
        f"{output_path} (copied={copied}, resized={resized}, depth_interpolated={depth_interpolated}, "
        f"zero_filled={zero_filled}, random_kept={random_kept}, skipped={skipped})."
    )


if __name__ == "__main__":
    main()
