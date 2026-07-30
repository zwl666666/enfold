import logging
import os
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .trainer import EnfoldTrainer
from .utils import misc
from .utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    precision = mixed_precision.strip().lower()
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return precision


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _to_dict(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return value


def create_enfold_model(
    model_id: str,
    video_dit_config,
    student_config,
    prediction_head_config,
    action_head_config,
    proprio_dim: int,
    tokenizer_model_id: str = "",
    tokenizer_max_len: int = 128,
    load_text_encoder: bool = False,
    skip_dit_load_from_pretrain: bool = False,
    action_dit_pretrained_path: str | None = None,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    repr_distillation=None,
    teacher_feature_layers=None,
    disable_action_head: bool = False,
    action_only_eval: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    **unused_options,
):
    from .models.cosmos import EnfoldModel

    video_dit_config = _to_dict(video_dit_config, "video_dit_config")
    student_config = _to_dict(student_config, "student_config")
    prediction_head_config = _to_dict(prediction_head_config, "prediction_head_config")
    action_head_config = _to_dict(action_head_config, "action_head_config")
    video_scheduler = _to_dict(video_scheduler, "video_scheduler")
    action_scheduler = _to_dict(action_scheduler, "action_scheduler")
    loss = _to_dict(loss, "loss")
    repr_distillation = _to_dict(repr_distillation, "repr_distillation")

    if teacher_feature_layers is None:
        teacher_feature_layers = [27]
    if isinstance(teacher_feature_layers, DictConfig):
        teacher_feature_layers = OmegaConf.to_container(teacher_feature_layers, resolve=True)
    teacher_feature_layers = [int(layer) for layer in teacher_feature_layers]

    ignored = sorted(
        key for key, value in unused_options.items() if value not in (None, {}, False, "")
    )
    if ignored:
        logger.warning("Ignoring unsupported Enfold model options: %s", ignored)

    return EnfoldModel.from_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        video_dit_config=video_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        student_config=student_config,
        prediction_head_config=prediction_head_config,
        action_head_config=action_head_config,
        proprio_dim=int(proprio_dim),
        teacher_feature_layers=teacher_feature_layers,
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_num_train_timesteps=int(action_scheduler.get("num_train_timesteps", 1000)),
        action_uniform_sigma_sampling=bool(action_scheduler.get("uniform_sigma_sampling", False)),
        video_train_time_distribution=str(
            video_scheduler.get("train_time_distribution", "logitnormal")
        ),
        video_train_time_weight=str(video_scheduler.get("train_time_weight", "uniform")),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_repr=float(loss.get("lambda_repr", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        repr_distillation=repr_distillation,
        disable_action_head=bool(disable_action_head),
        action_only_eval=bool(action_only_eval),
    )


def build_datasets(data_cfg: DictConfig):
    train_dataset = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        return train_dataset, train_dataset

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    train_stats_path = data_cfg.train.get("pretrained_norm_stats")
    val_stats_path = data_cfg.val.get("pretrained_norm_stats")
    stats_path = val_stats_path or train_stats_path or os.path.join(
        misc.get_work_dir(), "dataset_stats.json"
    )
    logger.info("Building validation dataset with stats: %s", stats_path)
    val_dataset = instantiate(data_cfg.val, pretrained_norm_stats=stats_path)
    return train_dataset, val_dataset


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return f"cuda:{local_rank}" if 0 <= local_rank < device_count else "cuda:0"


def run_training(cfg: DictConfig):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(output_dir)
    setup_logging(
        log_level=logging.INFO,
        preserve_hydra_handlers=False,
        log_file_path=str(output_dir / "train.log"),
    )
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if 0 <= local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)

    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), output_dir / "config.yaml")
    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(cfg.mixed_precision),
        device=_resolve_train_device(),
    )
    train_dataset, val_dataset = build_datasets(cfg.data)
    EnfoldTrainer(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ).train()
