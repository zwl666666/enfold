#!/usr/bin/env python3
"""Export the RobTwin DINO student and action denoiser as TensorRT engines."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.wan_video_dit import precompute_freqs_real
from fastwam.utils.tensorrt_inference import finalize_action_head_inputs

DEFAULT_TASK = (
    "robotwin_dino_student_action_3cam_384_proprio_concatpred_actionconcat_"
    "wancond_detachactstudent_detachwan_textlinearconcat_vith16plus_1e-4_"
    "cosmos_reprpool2x2_deepsup"
)


def mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[mixed_precision]


def student_action_condition(
    model: Any,
    input_image: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model.student(input_image, context, context_mask, drop_text=False)
    if len(outputs) < 4:
        raise RuntimeError(f"Expected at least four student outputs, got {len(outputs)}.")
    return outputs[2], outputs[3]


class StudentConditionWrapper(torch.nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return student_action_condition(self.model, input_image, context, context_mask)


class ActionDenoiseWrapper(torch.nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        action_condition: torch.Tensor,
        action_condition_mask: torch.Tensor,
        action_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.action_head(action_noise, timestep, action_condition, action_condition_mask)


def export_onnx(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        args,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    return output_path


def onnx_input_names(path: Path) -> set[str]:
    import onnx

    graph = onnx.load(str(path)).graph
    initializers = {initializer.name for initializer in graph.initializer}
    return {value.name for value in graph.input if value.name not in initializers}


def _shape_flag(name: str, shapes: dict[str, tuple[int, ...]]) -> str:
    return f"--{name}=" + ",".join(
        f"{tensor_name}:{'x'.join(str(dim) for dim in shape)}"
        for tensor_name, shape in shapes.items()
    )


def build_engine(
    *,
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    min_shapes: dict[str, tuple[int, ...]],
    opt_shapes: dict[str, tuple[int, ...]],
    max_shapes: dict[str, tuple[int, ...]],
    precision: str,
    workspace_mb: int | None,
) -> None:
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        _shape_flag("minShapes", min_shapes),
        _shape_flag("optShapes", opt_shapes),
        _shape_flag("maxShapes", max_shapes),
    ]
    # if precision != "fp32":
    #     command.append(f"--{precision}")
    if workspace_mb is not None:
        command.append(f"--memPoolSize=workspace:{workspace_mb}")
    subprocess.run(command, check=True)


def resolve_trtexec(value: str | None) -> str:
    executable = value or shutil.which("trtexec")
    if executable is None:
        raise FileNotFoundError(
            "`trtexec` is not in PATH. Pass its full path with `--trtexec`, for example "
            "--trtexec /wangx1211/TensorRT-11.0.0.114/bin/trtexec."
        )
    path = Path(executable).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"trtexec not found: {path}")
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(REPO_ROOT / "configs"))
    parser.add_argument("--config-name", default="sim_robotwin_dino_student_action")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--action-onnx", required=True)
    parser.add_argument("--student-onnx", required=True)
    parser.add_argument("--action-engine", help="Build the action-head TensorRT engine at this path.")
    parser.add_argument("--student-engine", help="Build the DINO-student TensorRT engine at this path.")
    parser.add_argument("--trtexec", help="Full path to trtexec; defaults to PATH.")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--trt-precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-task", default="move an object to its target")
    parser.add_argument("--prompt", help="Full prompt; overrides --prompt-task.")
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--max-context-tokens", type=int, default=128)
    parser.add_argument("--max-action-horizon", type=int, default=32)
    parser.add_argument("--workspace-mb", type=int)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    overrides = [
        f"task={args.task}",
        f"ckpt={args.ckpt}",
        f"mixed_precision={args.mixed_precision}",
        *args.override,
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(args.config_dir).resolve())):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    model = instantiate(cfg.model, model_dtype=mixed_precision_to_dtype(args.mixed_precision), device=args.device)
    model.load_checkpoint(str(cfg.ckpt))
    return cfg, model.to(args.device).eval()


def action_horizon_from_cfg(cfg: Any, args: argparse.Namespace) -> int:
    if args.action_horizon is not None:
        return int(args.action_horizon)
    value = cfg.EVALUATION.get("action_horizon")
    return int(value if value is not None else cfg.data.train.num_frames - 1)


def filter_shapes(shapes: dict[str, tuple[int, ...]], names: set[str]) -> dict[str, tuple[int, ...]]:
    return {name: shape for name, shape in shapes.items() if name in names}


def main() -> int:
    args = parse_args()
    cfg, model = load_model(args)
    action_horizon = action_horizon_from_cfg(cfg, args)
    video_size = cfg.data.train.get("video_size", [384, 320])
    image_height = int(args.image_height or video_size[0])
    image_width = int(args.image_width or video_size[1])
    prompt = args.prompt or DEFAULT_PROMPT.format(task=args.prompt_task)

    if getattr(model.action_head, "attn_head_dim", 0) and hasattr(model.action_head, "freqs"):
        model.action_head.freqs = precompute_freqs_real(
            int(model.action_head.attn_head_dim),
            end=int(model.action_head.freqs.shape[0]),
        )

    with torch.no_grad():
        context, context_mask = model._prepare_infer_context(prompt, None, None)
        input_image = torch.zeros(
            (1, 3, image_height, image_width), device=model.device, dtype=model.torch_dtype
        )
        student_condition, student_condition_mask = student_action_condition(
            model, input_image, context, context_mask
        )
        proprio = None
        if getattr(model, "proprio_condition_enabled", False):
            proprio = torch.zeros(
                (1, int(model.proprio_condition_dim)), device=model.device, dtype=model.torch_dtype
            )
        action_condition, action_condition_mask = finalize_action_head_inputs(
            model,
            action_condition=student_condition,
            action_condition_mask=student_condition_mask,
            proprio=proprio,
        )
    if action_condition_mask is None:
        raise RuntimeError("Action condition mask is required for TensorRT export.")

    action_onnx = export_onnx(
        ActionDenoiseWrapper(model).eval(),
        (
            action_condition,
            action_condition_mask,
            torch.zeros(
                (1, action_horizon, int(model.action_head.action_dim)),
                device=model.device,
                dtype=model.torch_dtype,
            ),
            torch.zeros((1,), device=model.device, dtype=model.torch_dtype),
        ),
        Path(args.action_onnx),
        input_names=["action_condition", "action_condition_mask", "action_noise", "timestep"],
        output_names=["pred_action"],
        dynamic_axes={
            "action_condition": {1: "condition_tokens"},
            "action_condition_mask": {1: "condition_tokens"},
            "action_noise": {1: "action_horizon"},
            "pred_action": {1: "action_horizon"},
        },
        opset=args.opset,
    )
    student_onnx = export_onnx(
        StudentConditionWrapper(model).eval(),
        (input_image, context, context_mask),
        Path(args.student_onnx),
        input_names=["input_image", "context", "context_mask"],
        output_names=["student_condition", "student_condition_mask"],
        dynamic_axes={
            "context": {1: "context_tokens"},
            "context_mask": {1: "context_tokens"},
            "student_condition": {1: "student_condition_tokens"},
            "student_condition_mask": {1: "student_condition_tokens"},
        },
        opset=args.opset,
    )
    print(f"Exported action ONNX: {action_onnx}")
    print(f"Exported student ONNX: {student_onnx}")

    if args.action_engine or args.student_engine:
        trtexec = resolve_trtexec(args.trtexec)
        action_inputs = onnx_input_names(action_onnx)
        student_inputs = onnx_input_names(student_onnx)
        action_tokens = int(action_condition.shape[1])
        context_tokens = int(context.shape[1])
        min_action_horizon = max(1, min(action_horizon, 8))
        min_context_tokens = max(1, min(context_tokens, 8))

        action_shapes = {
            "action_condition": (1, action_tokens, int(action_condition.shape[2])),
            "action_condition_mask": (1, action_tokens),
            "action_noise": (1, action_horizon, int(model.action_head.action_dim)),
            "timestep": (1,),
        }
        if args.action_engine:
            build_engine(
                trtexec=trtexec,
                onnx_path=action_onnx,
                engine_path=Path(args.action_engine),
                min_shapes=filter_shapes(
                    {
                        **action_shapes,
                        "action_condition": (1, min_context_tokens, int(action_condition.shape[2])),
                        "action_condition_mask": (1, min_context_tokens),
                        "action_noise": (1, min_action_horizon, int(model.action_head.action_dim)),
                    },
                    action_inputs,
                ),
                opt_shapes=filter_shapes(action_shapes, action_inputs),
                max_shapes=filter_shapes(
                    {
                        **action_shapes,
                        "action_condition": (
                            1,
                            max(action_tokens, int(args.max_context_tokens)),
                            int(action_condition.shape[2]),
                        ),
                        "action_condition_mask": (1, max(action_tokens, int(args.max_context_tokens))),
                        "action_noise": (
                            1,
                            max(action_horizon, int(args.max_action_horizon)),
                            int(model.action_head.action_dim),
                        ),
                    },
                    action_inputs,
                ),
                precision=args.trt_precision,
                workspace_mb=args.workspace_mb,
            )
            print(f"Built action TensorRT engine: {args.action_engine}")

        if args.student_engine:
            student_shapes = {
                "input_image": (1, 3, image_height, image_width),
                "context": (1, context_tokens, int(context.shape[2])),
                "context_mask": (1, context_tokens),
            }
            build_engine(
                trtexec=trtexec,
                onnx_path=student_onnx,
                engine_path=Path(args.student_engine),
                min_shapes=filter_shapes(
                    {
                        **student_shapes,
                        "context": (1, min_context_tokens, int(context.shape[2])),
                        "context_mask": (1, min_context_tokens),
                    },
                    student_inputs,
                ),
                opt_shapes=filter_shapes(student_shapes, student_inputs),
                max_shapes=filter_shapes(
                    {
                        **student_shapes,
                        "context": (1, max(context_tokens, int(args.max_context_tokens)), int(context.shape[2])),
                        "context_mask": (1, max(context_tokens, int(args.max_context_tokens))),
                    },
                    student_inputs,
                ),
                precision=args.trt_precision,
                workspace_mb=args.workspace_mb,
            )
            print(f"Built student TensorRT engine: {args.student_engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
