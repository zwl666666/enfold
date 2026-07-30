from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image

from enfold.utils.logging_config import get_logger

from ..action_dit import ActionDiT
from ..components import (
    DinoTextImageStudent,
    EnfoldModelBase,
    TokenDiTHead,
)
from .scheduler_rectified_flow import CosmosRectifiedFlowScheduler

logger = get_logger(__name__)
_ENFOLD_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_enfold_path(path_value: str | Path, *, force_project_relative: bool = False) -> str:
    """Resolve local Enfold paths independently of the caller's working directory.

    Evaluation invokes the policy from the RoboTwin checkout, while the model
    configuration lives in the Enfold checkout. Relative local paths therefore
    need to be anchored to the latter. Non-local identifiers (for example S3
    URIs or Hugging Face model names that do not exist locally) are preserved.
    """
    raw_path = str(path_value)
    if "://" in raw_path:
        return raw_path

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    project_path = (_ENFOLD_PROJECT_ROOT / path).resolve()
    if force_project_relative or project_path.exists():
        return str(project_path)
    return raw_path


def _import_cosmos_module(repo_path: str, module_name: str):
    repo_path = _resolve_enfold_path(repo_path, force_project_relative=True)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    try:
        return importlib.import_module(module_name)
    except RuntimeError as exc:
        if "CUDA extra not installed" not in str(exc):
            raise
        about_path = Path(repo_path) / "cosmos_predict2" / "__about__.py"
        version = "0.0.0"
        if about_path.exists():
            scope: dict[str, Any] = {}
            exec(about_path.read_text(encoding="utf-8"), scope)
            version = str(scope.get("__version__", version))
        fake_cuda = types.ModuleType("cosmos_cuda")
        fake_cuda.__version__ = version
        sys.modules["cosmos_cuda"] = fake_cuda
        sys.modules.pop("cosmos_predict2", None)
        return importlib.import_module(module_name)


class CosmosPromptEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        repo_path: str,
        cosmos_cfg: dict[str, Any],
        context_len: int,
        device: str,
    ):
        super().__init__()
        self.repo_path = _resolve_enfold_path(repo_path, force_project_relative=True)
        self.cosmos_cfg = dict(cosmos_cfg)
        self.model_name = str(self.cosmos_cfg.get("model_name", "custom"))
        self.experiment_name = str(self.cosmos_cfg.get("experiment_name", ""))
        self.config_file = str(self.cosmos_cfg.get("config_file", ""))
        self.context_len = int(context_len)
        self.device_name = str(device)
        self.backend_kind = "t5"
        self.tokenizer = None
        self.encoder = None
        text_embedding_kind = str(self.cosmos_cfg.get("text_embedding_kind", "auto")).strip().lower()

        if text_embedding_kind == "auto":
            config_mod = _import_cosmos_module(self.repo_path, "cosmos_predict2.config")
            utils_mod = _import_cosmos_module(
                self.repo_path, "cosmos_predict2._src.imaginaire.utils.config_helper"
            )
            experiment_name = self.experiment_name
            if not experiment_name and self.model_name and self.model_name in config_mod.MODEL_KEYS:
                checkpoint = config_mod.MODEL_CHECKPOINTS[config_mod.MODEL_KEYS[self.model_name]]
                experiment_name = checkpoint.experiment
            if not experiment_name:
                raise ValueError(
                    "Cosmos prompt encoder could not infer an experiment name. "
                    "Please set `video_dit_config.cosmos.experiment_name`."
                )
            config_module = utils_mod.get_config_module(self.config_file)
            config = importlib.import_module(config_module).make_config()
            config = utils_mod.override(config, ["--", f"experiment={experiment_name}", "~data_train"])
            model_cfg = config.model.config
            text_encoder_class = str(model_cfg.text_encoder_class).lower()
            if text_encoder_class == "t5":
                text_embedding_kind = "t5"
                self.cosmos_cfg.setdefault("text_embedding_model_name", "google-t5/t5-11b")
                self.cosmos_cfg.setdefault("text_embedding_local_files_only", True)
            elif text_encoder_class.startswith("reason1"):
                text_embedding_kind = "reason1"
                self.cosmos_cfg.setdefault("text_embedding_ckpt_path", model_cfg.text_encoder_config.ckpt_path)
                self.cosmos_cfg.setdefault(
                    "text_embedding_concat_strategy",
                    str(model_cfg.text_encoder_config.embedding_concat_strategy),
                )
                self.cosmos_cfg.setdefault(
                    "text_embedding_n_layers_per_group",
                    int(model_cfg.text_encoder_config.n_layers_per_group),
                )
            else:
                raise ValueError(
                    f"Unsupported Cosmos text_encoder_class={model_cfg.text_encoder_class!r} inferred from experiment."
                )

        if text_embedding_kind == "t5":
            t5_mod = _import_cosmos_module(
                self.repo_path, "cosmos_predict2._src.predict2.inference.get_t5_emb"
            )
            model_name = str(self.cosmos_cfg.get("text_embedding_model_name", "google-t5/t5-11b"))
            local_files_only = bool(self.cosmos_cfg.get("text_embedding_local_files_only", True))
            self.encoder = t5_mod.CosmosT5TextEncoder(
                model_name=model_name,
                device=self.device_name,
                local_files_only=local_files_only,
            )
            self.tokenizer = self.encoder.tokenizer
            self.text_encoder_class = "T5"
            self.backend_kind = "t5"
            return

        if text_embedding_kind == "reason1":
            text_encoder_mod = _import_cosmos_module(
                self.repo_path, "cosmos_predict2._src.predict2.text_encoders.text_encoder"
            )
            model_config_qwen_mod = _import_cosmos_module(
                self.repo_path, "cosmos_predict2._src.reason1.configs.default.model_config_qwen"
            )
            ckpt_path = self.cosmos_cfg.get("text_embedding_ckpt_path")
            if ckpt_path is None:
                raise ValueError(
                    "Cosmos Reason1 prompt encoding requires `video_dit_config.cosmos.text_embedding_ckpt_path`."
                )
            embedding_concat_strategy = str(
                self.cosmos_cfg.get("text_embedding_concat_strategy", "mean_pooling")
            )
            n_layers_per_group = int(self.cosmos_cfg.get("text_embedding_n_layers_per_group", 5))
            reason_model_type = str(
                self.cosmos_cfg.get("text_embedding_model_type", "Qwen/Qwen2.5-VL-7B-Instruct")
            )
            reason_tokenizer_type = str(self.cosmos_cfg.get("text_embedding_tokenizer_type", reason_model_type))
            reason_name_or_path = str(self.cosmos_cfg.get("text_embedding_name_or_path", reason_model_type))
            text_encoder_config = text_encoder_mod.TextEncoderConfig(
                compute_online=True,
                embedding_concat_strategy=embedding_concat_strategy,
                n_layers_per_group=n_layers_per_group,
                ckpt_path=str(ckpt_path),
                model_config=text_encoder_mod.L(text_encoder_mod.QwenVLBaseModel)(
                    model_config=text_encoder_mod.L(model_config_qwen_mod.QwenModelConfig)(
                        tokenizer_type=reason_tokenizer_type,
                        name_or_path=reason_name_or_path,
                        hidden_size=int(self.cosmos_cfg.get("text_embedding_hidden_size", 3584)),
                        intermediate_size=int(self.cosmos_cfg.get("text_embedding_intermediate_size", 18944)),
                        max_window_layers=int(self.cosmos_cfg.get("text_embedding_max_window_layers", 28)),
                        num_attention_heads=int(self.cosmos_cfg.get("text_embedding_num_attention_heads", 28)),
                        num_hidden_layers=int(self.cosmos_cfg.get("text_embedding_num_hidden_layers", 28)),
                        num_key_value_heads=int(self.cosmos_cfg.get("text_embedding_num_key_value_heads", 4)),
                        tie_word_embeddings=False,
                        vocab_size=int(self.cosmos_cfg.get("text_embedding_vocab_size", 152064)),
                        vision_config=text_encoder_mod.L(model_config_qwen_mod.QwenVisionConfig)(
                            out_hidden_size=int(self.cosmos_cfg.get("text_embedding_hidden_size", 3584))
                        ),
                        output_hidden_states=True,
                    ),
                    tokenizer=text_encoder_mod.L(text_encoder_mod.build_tokenizer)(
                        tokenizer_type=reason_tokenizer_type,
                    ),
                ),
            )
            self.encoder = text_encoder_mod.TextEncoder(text_encoder_config, device=self.device_name)
            self.tokenizer = self.encoder.model.tokenizer
            self.backend_kind = "reason1"
            self.reason1_mod = text_encoder_mod
            self.embedding_concat_strategy = str(self.encoder.config.embedding_concat_strategy)
            self.n_layers_per_group = int(self.encoder.config.n_layers_per_group)
            self.text_encoder_class = "reason1_7B"
            return

        raise ValueError(
            f"Unsupported cosmos text_embedding_kind={text_embedding_kind!r}. "
            "Expected one of: auto, t5, reason1."
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and len(args) > 0:
            device = args[0]
        if device is not None:
            self.device_name = str(device)
        if self.backend_kind == "t5":
            self.encoder.text_encoder.to(*args, **kwargs)
        elif self.backend_kind == "reason1":
            self.encoder.model.to(*args, **kwargs)
        return self

    def _encode_reason1(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids_batch = []
        batch_masks = []
        for prompt in prompts:
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                },
            ]
            tokenizer_output = self.encoder.model.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
            )
            input_ids = tokenizer_output["input_ids"]
            pad_id = self.encoder.model.tokenizer.pad_id
            valid_len = min(len(input_ids), self.context_len)
            padded_ids = input_ids.tolist()[: self.context_len] + [pad_id] * max(self.context_len - valid_len, 0)
            mask = torch.zeros(self.context_len, dtype=torch.bool)
            mask[:valid_len] = True
            input_ids_batch.append(torch.LongTensor(padded_ids))
            batch_masks.append(mask)

        input_ids_batch = torch.stack(input_ids_batch, dim=0).to(device=self.device_name)
        batch_masks_tensor = torch.stack(batch_masks, dim=0).to(device=self.device_name)
        self.encoder.model = self.encoder.model.to(self.device_name)
        with torch.no_grad():
            _, outputs_batch = self.encoder.model(input_ids_batch, {})
        hidden_states = outputs_batch["hidden_states"]
        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_hidden_states.append(self.reason1_mod.TextEncoder.mean_normalize(hidden_states[layer_idx]))

        strategy_enum = self.reason1_mod.EmbeddingConcatStrategy
        if self.embedding_concat_strategy == str(strategy_enum.FULL_CONCAT):
            text_embeddings = torch.cat(normalized_hidden_states, dim=-1)
        elif self.embedding_concat_strategy == str(strategy_enum.MEAN_POOLING):
            text_embeddings = torch.stack(normalized_hidden_states).mean(dim=0)
        elif self.embedding_concat_strategy == str(strategy_enum.POOL_EVERY_N_LAYERS_AND_CONCAT):
            pooled_groups = []
            for i in range(0, len(normalized_hidden_states), self.n_layers_per_group):
                pooled_groups.append(torch.stack(normalized_hidden_states[i : i + self.n_layers_per_group]).mean(dim=0))
            text_embeddings = torch.cat(pooled_groups, dim=-1)
        else:
            raise ValueError(f"Unsupported Reason1 embedding_concat_strategy: {self.embedding_concat_strategy}")

        text_embeddings = text_embeddings.masked_fill(~batch_masks_tensor.unsqueeze(-1), 0.0)
        return text_embeddings, batch_masks_tensor

    @torch.no_grad()
    def encode_prompts(self, prompts: Union[str, Sequence[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = [str(prompt) for prompt in prompts]
        if self.backend_kind == "t5":
            context, mask = self.encoder.encode_prompts(
                prompt_list,
                max_length=self.context_len,
                return_mask=True,
            )
            return context.to(device=self.device_name), mask.to(device=self.device_name, dtype=torch.bool)
        if self.backend_kind == "reason1":
            return self._encode_reason1(prompt_list)
        raise ValueError(f"Unsupported Cosmos prompt encoder backend: {self.backend_kind}")


def _resolve_cosmos_config(model_id: str, video_dit_config: dict[str, Any]) -> dict[str, Any]:
    backend = str(video_dit_config.get("backend", "cosmos")).lower()
    if backend != "cosmos":
        raise ValueError(
            "Cosmos student-action adapters require `video_dit_config.backend=cosmos`. "
            f"Got backend={backend!r}."
        )
    cosmos_cfg = dict(video_dit_config.get("cosmos", {}))
    if "checkpoint_path" not in cosmos_cfg and model_id:
        cosmos_cfg["checkpoint_path"] = model_id
    model_name = cosmos_cfg.get("model_name")
    repo_path = cosmos_cfg.get("repo_path")
    if not repo_path:
        raise ValueError("`video_dit_config.cosmos.repo_path` is required for Cosmos student-action adapters.")
    cosmos_cfg["repo_path"] = _resolve_enfold_path(repo_path, force_project_relative=True)
    repo_path = cosmos_cfg["repo_path"]

    # If the caller provides an official Cosmos model name such as `2B/post-trained`,
    # resolve the matching experiment metadata from the Cosmos registry and still allow
    # a local checkpoint path to override the actual weights file.
    if model_name is not None:
        repo_path_obj = Path(repo_path).expanduser().resolve()
        repo_path_str = str(repo_path_obj)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)
        try:
            from cosmos_predict2.config import MODEL_CHECKPOINTS, MODEL_KEYS
        except Exception as exc:
            raise RuntimeError(
                "Failed to import `cosmos_predict2.config` while resolving official Cosmos model metadata."
            ) from exc
        if model_name not in MODEL_KEYS:
            raise ValueError(
                f"Unknown Cosmos model_name={model_name!r}. "
                f"Expected one of: {sorted(MODEL_KEYS.keys())}"
            )
        checkpoint = MODEL_CHECKPOINTS[MODEL_KEYS[model_name]]
        cosmos_cfg.setdefault("checkpoint_path", checkpoint.s3.uri)
        cosmos_cfg.setdefault("experiment_name", checkpoint.experiment)
        cosmos_cfg.setdefault(
            "config_file",
            "cosmos_predict2/_src/predict2/distill/configs/registry_predict2p5.py"
            if MODEL_KEYS[model_name].distilled
            else "cosmos_predict2/_src/predict2/configs/video2world/config.py",
        )
    else:
        cosmos_cfg.setdefault("config_file", "cosmos_predict2/_src/predict2/configs/video2world/config.py")

    experiment_name = cosmos_cfg.get("experiment_name")
    checkpoint_path = cosmos_cfg.get("checkpoint_path")
    if not experiment_name:
        raise ValueError(
            "`video_dit_config.cosmos.experiment_name` is required unless "
            "`video_dit_config.cosmos.model_name` is provided."
        )
    if not checkpoint_path:
        raise ValueError(
            "`video_dit_config.cosmos.checkpoint_path` or `model_id` is required for Cosmos adapters."
        )

    for path_key in ("checkpoint_path", "tokenizer_vae_pth", "text_embedding_ckpt_path"):
        path_value = cosmos_cfg.get(path_key)
        if path_value:
            cosmos_cfg[path_key] = _resolve_enfold_path(path_value, force_project_relative=True)
    for path_key in ("text_embedding_name_or_path", "text_embedding_tokenizer_type"):
        path_value = cosmos_cfg.get(path_key)
        if path_value:
            cosmos_cfg[path_key] = _resolve_enfold_path(path_value)

    cosmos_cfg.setdefault("fps", 16)
    cosmos_cfg.setdefault("num_conditional_frames", 1)
    cosmos_cfg.setdefault("context_len", 512)
    cosmos_cfg.setdefault("load_ema_to_reg", True)
    cosmos_cfg.setdefault("enable_fsdp", False)
    cosmos_cfg.setdefault("seed", 0)
    cosmos_cfg.setdefault("disable_text_encoder", True)
    return cosmos_cfg


def _load_cosmos_teacher_model(
    *,
    device: str,
    model_id: str,
    video_dit_config: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    cosmos_cfg = _resolve_cosmos_config(model_id=model_id, video_dit_config=video_dit_config)
    repo_path = Path(cosmos_cfg["repo_path"]).expanduser().resolve()
    if not repo_path.is_dir():
        raise ValueError(f"Cosmos repo_path does not exist: {repo_path}")
    repo_path_str = str(repo_path)
    if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)

    try:
        from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Cosmos-Predict2.5. "
            "Please make sure the repo and its dependencies are installed and "
            "`video_dit_config.cosmos.repo_path` points to the Cosmos checkout."
        ) from exc

    experiment_opts = [
        "~data_train",
        *(
            ["model.config.text_encoder_config.compute_online=false"]
            if bool(cosmos_cfg.get("disable_text_encoder", False))
            else []
        ),
    ]
    tokenizer_vae_pth = cosmos_cfg.get("tokenizer_vae_pth")
    if tokenizer_vae_pth:
        tokenizer_vae_path = str(Path(str(tokenizer_vae_pth)).expanduser().resolve())
        experiment_opts.append(f"+model.config.tokenizer.vae_pth={tokenizer_vae_path}")

    if str(cosmos_cfg.get("text_embedding_concat_strategy", "none")).strip().lower() == "mean_pooling":
        experiment_opts.append(
            f"model.config.net.crossattn_proj_in_channels={int(video_dit_config['text_dim'])}"
        )

    model, _config = load_model_from_checkpoint(
        experiment_name=str(cosmos_cfg["experiment_name"]),
        s3_checkpoint_dir=str(cosmos_cfg["checkpoint_path"]),
        config_file=str(cosmos_cfg["config_file"]),
        enable_fsdp=bool(cosmos_cfg["enable_fsdp"]),
        load_ema_to_reg=bool(cosmos_cfg["load_ema_to_reg"]),
        instantiate_ema=False,
        seed=int(cosmos_cfg["seed"]),
        experiment_opts=experiment_opts,
        to_device=device,
    )
    model.eval()
    model.requires_grad_(False)

    if str(cosmos_cfg.get("text_embedding_concat_strategy", "none")).strip().lower() == "mean_pooling":
        crossattn_proj = getattr(getattr(model, "net", None), "crossattn_proj", None)
        if crossattn_proj is None:
            raise RuntimeError("Cosmos teacher is missing `net.crossattn_proj`; cannot adapt teacher text projection.")
        for module in crossattn_proj.modules():
            reset_fn = getattr(module, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()
        logger.info(
            "Reinitialized Cosmos teacher crossattn_proj for mean_pooling input dim=%d.",
            int(video_dit_config["text_dim"]),
        )

    model_paths = {
        "teacher_video_dit": str(cosmos_cfg["checkpoint_path"]),
        "vae": str(cosmos_cfg["checkpoint_path"]),
        "text_encoder": None,
        "tokenizer": None,
        "cosmos_repo": repo_path_str,
        "cosmos_experiment": str(cosmos_cfg["experiment_name"]),
    }
    return model, cosmos_cfg, model_paths


def _load_cosmos_text_components(
    *,
    device: str,
    cosmos_cfg: dict[str, Any],
    context_len: int,
) -> tuple[CosmosPromptEncoder, Any, dict[str, Any]]:
    prompt_encoder = CosmosPromptEncoder(
        repo_path=str(cosmos_cfg["repo_path"]),
        cosmos_cfg=cosmos_cfg,
        context_len=int(context_len),
        device=device,
    )
    model_paths = {
        "text_encoder": (
            f"{cosmos_cfg.get('text_embedding_kind', 't5')}::"
            f"{cosmos_cfg.get('text_embedding_model_name', cosmos_cfg.get('text_embedding_model_type', 'custom'))}"
        ),
        "tokenizer": getattr(prompt_encoder.tokenizer, "name_or_path", None),
    }
    return prompt_encoder, prompt_encoder.tokenizer, model_paths


class EnfoldModel(EnfoldModelBase):
    """Cosmos-backed Enfold model.

    Assumptions:
    1. `sample['context']` / `context_mask` are already cached in the Cosmos text-embedding space.
    2. The caller points configs/targets at `enfold.models.cosmos.*` explicitly.
    """

    def __init__(self, *args, cosmos_config: Optional[dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_backend = "cosmos"
        self.cosmos_config = {} if cosmos_config is None else dict(cosmos_config)
        self.cosmos_fps = int(self.cosmos_config.get("fps", 16))
        self.cosmos_num_conditional_frames = int(self.cosmos_config.get("num_conditional_frames", 1))
        self.cosmos_validate_infer_shape = bool(self.cosmos_config.get("validate_infer_shape", False))
        self.student_text_pooling = str(self.cosmos_config.get("student_text_pooling", "none")).strip().lower()
        self.student_text_dim = int(self.cosmos_config.get("student_text_dim", 3584))
        if self.student_text_pooling not in {"none", "mean_pooling"}:
            raise ValueError(
                f"Unsupported `video_dit_config.cosmos.student_text_pooling={self.student_text_pooling!r}`. "
                "Expected one of: none, mean_pooling."
            )

    @classmethod
    def from_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "",
        tokenizer_model_id: str = "",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = False,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        skip_dit_load_from_pretrain: bool = False,
        action_dit_pretrained_path: str | None = None,
        student_config: dict[str, Any] | None = None,
        prediction_head_config: dict[str, Any] | None = None,
        action_head_config: dict[str, Any] | None = None,
        proprio_dim: int = 0,
        teacher_feature_layers: Sequence[int] = (27,),
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_num_train_timesteps: int = 1000,
        action_uniform_sigma_sampling: bool = False,
        video_train_time_distribution: str = "uniform",
        video_train_time_weight: str = "gaussian",
        loss_lambda_video: float = 1.0,
        loss_lambda_repr: float = 1.0,
        loss_lambda_action: float = 1.0,
        action_only_eval: bool = False,
        repr_distillation: Optional[dict[str, Any]] = None,
        disable_action_head: bool = False,
    ):
        del tokenizer_model_id, tokenizer_max_len, redirect_common_files
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required.")
        if student_config is None:
            raise ValueError("`student_config` is required.")
        if not action_only_eval and prediction_head_config is None:
            raise ValueError("`prediction_head_config` is required.")
        if action_head_config is None:
            raise ValueError("`action_head_config` is required.")
        cosmos_cfg = _resolve_cosmos_config(model_id=model_id, video_dit_config=video_dit_config)
        teacher_text_dim = int(video_dit_config["text_dim"])
        student_text_pooling = str(cosmos_cfg.get("student_text_pooling", "none")).strip().lower()
        if student_text_pooling not in {"none", "mean_pooling"}:
            raise ValueError(
                f"Unsupported `video_dit_config.cosmos.student_text_pooling={student_text_pooling!r}`. "
                "Expected one of: none, mean_pooling."
            )
        student_text_dim = (
            int(cosmos_cfg.get("student_text_dim", cosmos_cfg.get("text_embedding_hidden_size", 3584)))
            if student_text_pooling == "mean_pooling"
            else teacher_text_dim
        )
        cosmos_cfg["student_text_pooling"] = student_text_pooling
        cosmos_cfg["student_text_dim"] = int(student_text_dim)

        student = DinoTextImageStudent(text_dim=student_text_dim, **student_config)
        prediction_head_config = cls._resolve_prediction_head_config(prediction_head_config, student)
        action_head = ActionDiT.from_pretrained(
            action_dit_config=action_head_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        if action_only_eval:
            teacher_video_expert = None
            vae = None
            prediction_head = None
            text_encoder = None
            tokenizer = None
            model_paths = {
                "teacher_video_dit": None,
                "vae": None,
                "text_encoder": None,
                "tokenizer": None,
                "student": student_config.get("model_id"),
                "cosmos_repo": cosmos_cfg["repo_path"],
                "cosmos_experiment": cosmos_cfg["experiment_name"],
            }
        else:
            teacher_video_expert, cosmos_cfg, model_paths = _load_cosmos_teacher_model(
                device=device,
                model_id=model_id,
                video_dit_config=video_dit_config,
            )
            vae = teacher_video_expert
            prediction_head = TokenDiTHead(**prediction_head_config)
            text_encoder = None
            tokenizer = None
            model_paths["student"] = student_config.get("model_id")

        if load_text_encoder:
            prompt_encoder, prompt_tokenizer, text_model_paths = _load_cosmos_text_components(
                device=device,
                cosmos_cfg=cosmos_cfg,
                context_len=int(cosmos_cfg.get("context_len", 512)),
            )
            text_encoder = prompt_encoder
            tokenizer = prompt_tokenizer
            model_paths.update(text_model_paths)

        model = cls(
            teacher_video_expert=teacher_video_expert,
            vae=vae,
            student=student,
            prediction_head=prediction_head,
            action_head=action_head,
            proprio_dim=proprio_dim,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            teacher_feature_layers=teacher_feature_layers,
            text_dim=teacher_text_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_repr=loss_lambda_repr,
            loss_lambda_action=loss_lambda_action,
            repr_distillation=repr_distillation,
            disable_action_head=disable_action_head,
            action_uniform_sigma_sampling=action_uniform_sigma_sampling,
            cosmos_config=cosmos_cfg,
        )
        model.train_video_scheduler = CosmosRectifiedFlowScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
            train_time_distribution=video_train_time_distribution,
            train_time_weight=video_train_time_weight,
        )
        model.infer_video_scheduler = CosmosRectifiedFlowScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
            train_time_distribution=video_train_time_distribution,
            train_time_weight=video_train_time_weight,
        )
        model.model_paths = model_paths
        return model

    def _context_for_student(self, context: torch.Tensor) -> torch.Tensor:
        if self.student_text_pooling != "mean_pooling":
            return context
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B,L,D], got {tuple(context.shape)}")
        if context.shape[-1] == self.student_text_dim:
            return context
        if context.shape[-1] % self.student_text_dim != 0:
            raise ValueError(
                "Cannot mean-pool full-concat context for student: "
                f"context_dim={context.shape[-1]} is not divisible by student_text_dim={self.student_text_dim}."
            )
        n_groups = context.shape[-1] // self.student_text_dim
        return context.view(context.shape[0], context.shape[1], n_groups, self.student_text_dim).mean(dim=2)

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.text_encoder, CosmosPromptEncoder):
            context, mask = self.text_encoder.encode_prompts(prompt)
            return context.to(device=self.device, dtype=self.torch_dtype), mask.to(device=self.device, dtype=torch.bool)
        return super().encode_prompt(prompt)

    def _context_to_cosmos_crossattn(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError(f"Cosmos teacher expects `context` to be [B,L,D], got {tuple(context.shape)}")
        return context.to(device=self.device, dtype=self.torch_dtype)

    def _video_to_cosmos_float(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Match Cosmos training path: float video in [-1, 1] before tokenizer.encode."""
        video_tensor = video_tensor.detach().to(device=self.device, dtype=self.torch_dtype)
        if video_tensor.dtype == torch.uint8:
            video_tensor = video_tensor.float().div(255.0).mul(2.0).sub(1.0).to(dtype=self.torch_dtype)
        elif video_tensor.max().item() > 1.5:
            # Handle [0,255] float inputs conservatively.
            video_tensor = video_tensor.float().div(255.0).mul(2.0).sub(1.0).to(dtype=self.torch_dtype)
        elif video_tensor.min().item() >= 0.0:
            # Map [0,1] float to [-1,1].
            video_tensor = video_tensor.mul(2.0).sub(1.0)
        return video_tensor.clamp(-1.0, 1.0)

    def _build_cosmos_latent_mask(self, latents: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(
            (latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]),
            device=latents.device,
            dtype=latents.dtype,
        )
        cond_frames = min(self.cosmos_num_conditional_frames, latents.shape[2])
        mask[:, :, :cond_frames] = 1
        return mask

    def _build_cosmos_fps(self, batch_size: int) -> torch.Tensor:
        return torch.full((batch_size,), self.cosmos_fps, device=self.device, dtype=torch.int64)

    def _build_cosmos_padding_mask(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (latents.shape[0], 1, latents.shape[3], latents.shape[4]),
            device=latents.device,
            dtype=latents.dtype,
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        del tiled, tile_size, tile_stride
        if self.teacher_video_expert is None:
            raise RuntimeError("Video latent encoding is unavailable in action-only eval mode.")
        video_float = self._video_to_cosmos_float(video_tensor)
        return self.teacher_video_expert.encode(video_float).to(device=self.device, dtype=self.torch_dtype)

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        del tiled, tile_size, tile_stride
        if self.teacher_video_expert is None:
            raise RuntimeError("Video latent decoding is unavailable in action-only eval mode.")
        video_tensor = self.teacher_video_expert.decode(latents).squeeze(0).detach().float().clamp(-1.0, 1.0)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        return [Image.fromarray(video_tensor[:, t].permute(1, 2, 0).numpy()) for t in range(video_tensor.shape[1])]

    def _teacher_video_prediction(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        conditional_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del context_mask, fuse_vae_embedding_in_latents
        if self.teacher_video_expert is None:
            raise RuntimeError("Cosmos teacher video expert is not loaded.")
        from cosmos_predict2._src.predict2.conditioner import DataType

        xt = latents
        condition_mask = self._build_cosmos_latent_mask(latents)
        if conditional_latents is not None:
            xt = conditional_latents.to(device=latents.device, dtype=latents.dtype) * condition_mask + xt * (1 - condition_mask)
        timesteps = timestep.view(-1, 1).to(device=latents.device, dtype=torch.float32)
        conditional_frame_timestep = float(getattr(self.teacher_video_expert.config, "conditional_frame_timestep", -1.0))
        if conditional_frame_timestep >= 0:
            timesteps = timesteps.expand(-1, latents.shape[2]).clone()
            condition_video_mask_B_1_T_1_1 = condition_mask.mean(dim=[1, 3, 4], keepdim=True)
            timestep_cond = torch.full_like(condition_video_mask_B_1_T_1_1, conditional_frame_timestep, dtype=torch.float32)
            timesteps = (
                timestep_cond * condition_video_mask_B_1_T_1_1
                + timesteps.view(timesteps.shape[0], 1, timesteps.shape[1], 1, 1) * (1 - condition_video_mask_B_1_T_1_1)
            ).squeeze()
            timesteps = timesteps.unsqueeze(0) if timesteps.ndim == 1 else timesteps
        out = self.teacher_video_expert.net(
            x_B_C_T_H_W=xt,
            timesteps_B_T=timesteps,
            crossattn_emb=self._context_to_cosmos_crossattn(context),
            fps=self._build_cosmos_fps(latents.shape[0]),
            padding_mask=self._build_cosmos_padding_mask(latents),
            data_type=DataType.VIDEO,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
        )
        return out[0] if isinstance(out, tuple) else out

    def _teacher_video_prediction_and_features(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        conditional_latents: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        del context_mask, fuse_vae_embedding_in_latents
        if self.teacher_video_expert is None:
            raise RuntimeError("Cosmos teacher video expert is not loaded.")
        from cosmos_predict2._src.predict2.conditioner import DataType

        xt = latents
        condition_mask = self._build_cosmos_latent_mask(latents)
        if conditional_latents is not None:
            xt = conditional_latents.to(device=latents.device, dtype=latents.dtype) * condition_mask + xt * (1 - condition_mask)
        timesteps = timestep.view(-1, 1).to(device=latents.device, dtype=torch.float32)
        conditional_frame_timestep = float(getattr(self.teacher_video_expert.config, "conditional_frame_timestep", -1.0))
        if conditional_frame_timestep >= 0:
            timesteps = timesteps.expand(-1, latents.shape[2]).clone()
            condition_video_mask_B_1_T_1_1 = condition_mask.mean(dim=[1, 3, 4], keepdim=True)
            timestep_cond = torch.full_like(condition_video_mask_B_1_T_1_1, conditional_frame_timestep, dtype=torch.float32)
            timesteps = (
                timestep_cond * condition_video_mask_B_1_T_1_1
                + timesteps.view(timesteps.shape[0], 1, timesteps.shape[1], 1, 1) * (1 - condition_video_mask_B_1_T_1_1)
            ).squeeze()
            timesteps = timesteps.unsqueeze(0) if timesteps.ndim == 1 else timesteps
        pred_video, feature_list = self.teacher_video_expert.net(
            x_B_C_T_H_W=xt,
            timesteps_B_T=timesteps,
            crossattn_emb=self._context_to_cosmos_crossattn(context),
            fps=self._build_cosmos_fps(latents.shape[0]),
            padding_mask=self._build_cosmos_padding_mask(latents),
            data_type=DataType.VIDEO,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            intermediate_feature_ids=list(self.teacher_feature_layers),
        )
        if not feature_list:
            raise ValueError(f"No Cosmos teacher layers selected from {self.teacher_feature_layers}.")
        grid_size = (int(latents.shape[2]), int(latents.shape[3] / 2), int(latents.shape[4] / 2)) # 经过teacher_video_expert.net后，特征图的时间维不变，空间维减半
        return pred_video, torch.cat([feat.detach() for feat in feature_list], dim=-1), grid_size

    def _validate_cosmos_infer_shape(self, height: int, width: int, num_frames: int) -> None:
        if not self.cosmos_validate_infer_shape:
            return
        expected_h, expected_w = self.teacher_video_expert.get_video_height_width()
        expected_frames = self.teacher_video_expert.tokenizer.get_pixel_num_frames(self.teacher_video_expert.config.state_t)
        if (height, width) != (expected_h, expected_w):
            raise ValueError(
                f"Cosmos teacher expects input size {(expected_h, expected_w)}, got {(height, width)}. "
                "Please update your dataset/video_size to the Cosmos checkpoint resolution."
            )
        if int(num_frames) != int(expected_frames):
            raise ValueError(
                f"Cosmos teacher expects num_frames={expected_frames}, got {num_frames}. "
                "Please align the rollout horizon/video frame count with the Cosmos checkpoint."
            )

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action=None,
        action_horizon=None,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del action, action_horizon, proprio, action_cfg_scale, tiled
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        _, _, height, width = input_image.shape
        self._validate_cosmos_infer_shape(height, width, num_frames)
        context_pos, mask_pos = self._prepare_infer_context(prompt, context, context_mask)
        context_neg = None
        mask_neg = None
        if text_cfg_scale != 1.0:
            context_neg, mask_neg = self._prepare_infer_context(
                "" if negative_prompt is None else negative_prompt,
                None,
                None,
            )
        _, _, student_pos, student_mask_pos = self._student_condition(
            input_image=input_image,
            prompt=None,
            context=context_pos,
            context_mask=mask_pos,
            drop_text=False,
        )
        context_pos, mask_pos = self._build_teacher_context(context_pos, mask_pos, student_pos, student_mask_pos)
        if context_neg is not None:
            _, _, student_neg, student_mask_neg = self._student_condition(
                input_image=input_image,
                prompt=None,
                context=context_neg,
                context_mask=mask_neg,
                drop_text=False,
            )
            context_neg, mask_neg = self._build_teacher_context(context_neg, mask_neg, student_neg, student_mask_neg)
        first_frame_video = input_image.unsqueeze(2).to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_video_latents(first_frame_video)
        latent_t = int(self.teacher_video_expert.tokenizer.get_latent_num_frames(num_frames))
        latent_h = int(first_frame_latents.shape[-2])
        latent_w = int(first_frame_latents.shape[-1])

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents = torch.randn(
            (1, int(first_frame_latents.shape[1]), latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents[:, :, : first_frame_latents.shape[2]] = first_frame_latents
        timesteps, deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.unsqueeze(0).to(device=self.device, dtype=latents.dtype)
            pred_pos = self._teacher_video_prediction(latents, timestep, context_pos, mask_pos, True)
            pred = pred_pos
            if context_neg is not None:
                pred_neg = self._teacher_video_prediction(latents, timestep, context_neg, mask_neg, True)
                pred = pred_neg + text_cfg_scale * (pred_pos - pred_neg)
            pred = pred.to(device=latents.device, dtype=latents.dtype)
            latents = self.infer_video_scheduler.step(pred, step_delta, latents)
            latents[:, :, : first_frame_latents.shape[2]] = first_frame_latents
        return {"video": self._decode_latents(latents)}

    def training_loss(self, sample, tiled: bool = False):
        if self.teacher_video_expert is None:
            return self._training_loss_action_only(sample, tiled=tiled)
        del tiled
        video = sample["video"]
        action = sample.get("action")
        if action is None:
            raise ValueError("`sample['action']` is required for Enfold training.")
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be [B,3,T,H,W], got {tuple(video.shape)}")
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B,T,D], got {tuple(action.shape)}")
        context, context_mask = self._prepare_context(sample)
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("sample proprio is required.")
        if proprio.ndim != 3:
            raise ValueError(f"sample proprio must be [B,T,D], got {tuple(proprio.shape)}")
        current_proprio = proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video)
        noise = torch.randn_like(input_latents)
        video_timestep = self.train_video_scheduler.sample_training_t(input_latents.shape[0], self.device, torch.float32)
        noised_latents = self.train_video_scheduler.add_noise(input_latents, noise, video_timestep)
        noised_latents[:, :, : self.cosmos_num_conditional_frames] = input_latents[:, :, : self.cosmos_num_conditional_frames]
        video_target = self.train_video_scheduler.training_target(input_latents, noise, video_timestep)

        first_frame = input_video[:, :, 0]
        student_context = self._context_for_student(context)
        (
            _,
            _,
            action_condition,
            action_condition_mask,
            prediction_head_condition,
            prediction_head_condition_mask,
        ) = self.student(
            first_frame,
            student_context,
            context_mask,
        )
        video_context, video_context_mask = self._build_teacher_context(
            context,
            context_mask,
            action_condition,
            action_condition_mask,
        )
        pred_video, teacher_features, teacher_grid_size = self._teacher_video_prediction_and_features(
            noised_latents,
            video_timestep,
            video_context,
            video_context_mask,
            True,
            conditional_latents=input_latents,
        )
        pred_video = pred_video[:, :, self.cosmos_num_conditional_frames :]
        video_target = video_target[:, :, self.cosmos_num_conditional_frames :]
        loss_video_per_sample = F.mse_loss(pred_video.float(), video_target.float(), reduction="none").mean(dim=(1, 2, 3, 4))
        video_weight = self.train_video_scheduler.training_weight(video_timestep).to(
            device=loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        
        num_conditional_frames = (
            self.cosmos_num_conditional_frames if self.repr_exclude_conditional_frames else 0
        )
        loss_repr = self._compute_repr_branch(
            teacher_features=teacher_features,
            teacher_grid_size=teacher_grid_size,
            video_timestep=video_timestep,
            prediction_condition=prediction_head_condition,
            prediction_condition_mask=prediction_head_condition_mask,
            num_conditional_frames=num_conditional_frames,
            normalize_teacher=True,
        )

        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_noise = torch.randn_like(action)
        action_timestep = self.train_action_scheduler.sample_training_t(action.shape[0], self.device, action.dtype)
        noised_action = self.train_action_scheduler.add_noise(action, action_noise, action_timestep)
        action_target = self.train_action_scheduler.training_target(action, action_noise, action_timestep)
        action_head_condition = action_condition.detach()
        action_head_condition_mask = action_condition_mask
        action_head_condition, action_head_condition_mask = self._append_proprio_to_action_condition(
            action_head_condition,
            action_head_condition_mask,
            current_proprio,
        )
        pred_action = self.action_head(
            action_tokens=noised_action,
            timestep=action_timestep,
            context=action_head_condition,
            context_mask=action_head_condition_mask,
        )
        action_mask = sample.get("action_is_pad")
        if action_mask is not None:
            action_mask = action_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            valid = (~action_mask).to(dtype=pred_action.dtype).unsqueeze(-1)
            denom = valid.sum().clamp(min=1.0) * pred_action.shape[-1]
            loss_action = ((pred_action.float() - action_target.float()).pow(2) * valid).sum() / denom
        else:
            loss_action = F.mse_loss(pred_action.float(), action_target.float())
        loss = self.loss_lambda_video * loss_video + self.loss_lambda_repr * loss_repr + self.loss_lambda_action * loss_action
        return loss, {
            "loss_video": float(loss_video.detach().item()),
            "loss_repr": float(loss_repr.detach().item()),
            "loss_action": float(loss_action.detach().item())
        }
