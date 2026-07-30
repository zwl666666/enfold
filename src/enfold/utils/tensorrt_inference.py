from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def _trt_dtype_to_torch(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}") from exc


def finalize_action_head_inputs(
    model: Any,
    *,
    action_condition: torch.Tensor,
    action_condition_mask: Optional[torch.Tensor],
    proprio: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply the non-exported condition transforms used by ``infer_action``."""
    action_condition, action_condition_mask = model._compress_temporal_condition(
        action_condition,
        action_condition_mask,
    )
    action_condition = action_condition.detach()
    return model._append_proprio_to_action_condition(
        action_condition,
        action_condition_mask,
        proprio,
    )


def student_action_condition(
    model: Any,
    *,
    input_image: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    drop_text: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exactly the student outputs consumed by ``EnfoldModel``.

    Current students return additional prediction/deep-supervision outputs.  The
    action path is always outputs 2 and 3, matching ``_student_condition``.
    """
    outputs = model.student(input_image, context, context_mask, drop_text=drop_text)
    if len(outputs) < 4:
        raise RuntimeError(
            f"Expected student to return at least four tensors, got {len(outputs)}."
        )
    return outputs[2], outputs[3]


class _TensorRTRunner:
    def __init__(self, engine_path: str | Path, device: str | torch.device) -> None:
        import tensorrt as trt

        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TensorRT inference requires an available CUDA device.")

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def _infer(self, available_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [name for name in self.input_names if name not in available_inputs]
        if missing:
            raise ValueError(f"Missing TensorRT inputs for {self.engine_path}: {missing}")

        stream = torch.cuda.current_stream(device=self.device)
        set_profile = getattr(self._context, "set_optimization_profile_async", None)
        if callable(set_profile) and not set_profile(0, stream.cuda_stream):
            raise RuntimeError(f"Could not activate TensorRT profile 0: {self.engine_path}")

        bound_tensors: dict[str, torch.Tensor] = {}
        for name in self.input_names:
            tensor = available_inputs[name].to(
                device=self.device,
                dtype=_trt_dtype_to_torch(self._engine.get_tensor_dtype(name)),
                non_blocking=True,
            ).contiguous()
            if not self._context.set_input_shape(name, tuple(int(dim) for dim in tensor.shape)):
                raise ValueError(
                    f"Input shape {tuple(tensor.shape)} is outside the TensorRT profile for `{name}` "
                    f"in {self.engine_path}."
                )
            bound_tensors[name] = tensor

        for name in self.output_names:
            shape = tuple(int(dim) for dim in self._context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"TensorRT did not resolve output shape for `{name}`: {shape} ({self.engine_path})"
                )
            bound_tensors[name] = torch.empty(
                shape,
                device=self.device,
                dtype=_trt_dtype_to_torch(self._engine.get_tensor_dtype(name)),
            )

        for name, tensor in bound_tensors.items():
            if not self._context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Could not bind TensorRT tensor `{name}`: {self.engine_path}")
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed: {self.engine_path}")
        # PyTorch scheduler operations use this same stream, so a device-wide
        # synchronization here would only serialize otherwise ordered work.
        return {name: bound_tensors[name] for name in self.output_names}


class TensorRTActionDenoiseRunner(_TensorRTRunner):
    def __init__(self, engine_path: str | Path, device: str | torch.device = "cuda") -> None:
        super().__init__(engine_path, device)
        if "pred_action" not in self.output_names:
            raise ValueError(
                f"Expected output `pred_action`, got {self.output_names} in {self.engine_path}."
            )

    def infer(
        self,
        *,
        action_condition: torch.Tensor,
        action_condition_mask: torch.Tensor,
        action_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self._infer(
            {
                "action_condition": action_condition,
                "action_condition_mask": action_condition_mask,
                "action_noise": action_noise,
                "timestep": timestep,
            }
        )["pred_action"]


class TensorRTStudentConditionRunner(_TensorRTRunner):
    def __init__(self, engine_path: str | Path, device: str | torch.device = "cuda") -> None:
        super().__init__(engine_path, device)
        expected = {"student_condition", "student_condition_mask"}
        actual = set(self.output_names)
        if actual != expected:
            raise ValueError(
                f"Expected student outputs {sorted(expected)}, got {self.output_names} in {self.engine_path}."
            )

    def infer(
        self,
        *,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self._infer(
            {
                "input_image": input_image,
                "context": context,
                "context_mask": context_mask,
            }
        )
        return outputs["student_condition"], outputs["student_condition_mask"]


class TensorRTActionInferenceBackend:
    """TensorRT action denoiser with an optional TensorRT DINO student."""

    def __init__(
        self,
        model: Any,
        *,
        action_engine_path: str | Path,
        student_engine_path: str | Path | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.action_runner = TensorRTActionDenoiseRunner(action_engine_path, device=self.device)
        self.student_runner = (
            TensorRTStudentConditionRunner(student_engine_path, device=self.device)
            if student_engine_path is not None
            else None
        )
        self._cached_prompt: Optional[str] = None
        self._cached_context: Optional[torch.Tensor] = None
        self._cached_context_mask: Optional[torch.Tensor] = None

    def _get_cached_prompt_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_prompt == prompt and self._cached_context is not None and self._cached_context_mask is not None:
            return self._cached_context, self._cached_context_mask
        context, context_mask = self.model.encode_prompt(prompt)
        context = context.unsqueeze(0) if context.ndim == 2 else context
        context_mask = context_mask.unsqueeze(0) if context_mask.ndim == 1 else context_mask
        self._cached_prompt = prompt
        self._cached_context = context.to(device=self.device, dtype=self.model.torch_dtype)
        self._cached_context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        return self._cached_context, self._cached_context_mask

    def _prepare_context(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context is None or context_mask is None:
            if prompt is None:
                raise ValueError("Either `prompt` or `context/context_mask` must be provided.")
            return self._get_cached_prompt_context(prompt)
        return self.model._prepare_infer_context(prompt, context, context_mask)

    def _build_action_condition(
        self,
        *,
        input_image: torch.Tensor,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
        drop_text: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, context_mask = self._prepare_context(prompt, context, context_mask)
        input_image = input_image.to(device=self.device, dtype=self.model.torch_dtype)
        if self.student_runner is not None and not drop_text:
            action_condition, action_condition_mask = self.student_runner.infer(
                input_image=input_image,
                context=context,
                context_mask=context_mask,
            )
        else:
            action_condition, action_condition_mask = student_action_condition(
                self.model,
                input_image=input_image,
                context=context,
                context_mask=context_mask,
                drop_text=drop_text,
            )
        action_condition, action_condition_mask = finalize_action_head_inputs(
            self.model,
            action_condition=action_condition,
            action_condition_mask=action_condition_mask,
            proprio=proprio,
        )
        if action_condition_mask is None:
            raise RuntimeError("Action condition mask must be present for TensorRT action inference.")
        return action_condition, action_condition_mask

    @torch.no_grad()
    def infer_action(
        self,
        *,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del num_video_frames, negative_prompt, tiled
        if self.model.disable_action_head or self.model.action_head is None:
            raise RuntimeError("Cannot run action inference because this model has no action head.")
        self.model.eval()

        action_condition, action_condition_mask = self._build_action_condition(
            input_image=input_image,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            drop_text=False,
        )
        uncond_condition = uncond_mask = None
        if text_cfg_scale != 1.0:
            uncond_condition, uncond_mask = self._build_action_condition(
                input_image=input_image,
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                drop_text=True,
            )

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn(
            (1, int(action_horizon), self.model.action_head.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.model.torch_dtype)
        timesteps, deltas = self.model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.unsqueeze(0).to(device=self.device, dtype=action.dtype)
            pred = self.action_runner.infer(
                action_condition=action_condition,
                action_condition_mask=action_condition_mask,
                action_noise=action,
                timestep=timestep,
            )
            if uncond_condition is not None and uncond_mask is not None:
                pred_uncond = self.action_runner.infer(
                    action_condition=uncond_condition,
                    action_condition_mask=uncond_mask,
                    action_noise=action,
                    timestep=timestep,
                )
                pred = pred_uncond + text_cfg_scale * (pred - pred_uncond)
            action = self.model.infer_action_scheduler.step(pred, step_delta, action)
        return {"action": action[0].detach().to(device="cpu", dtype=torch.float32)}
