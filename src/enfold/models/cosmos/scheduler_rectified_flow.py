import torch


class CosmosRectifiedFlowScheduler:
    """Cosmos-style rectified-flow scheduler for latent video training/inference."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        train_time_distribution: str = "logitnormal",
        train_time_weight: str = "uniform",
    ):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        train_time_distribution = str(train_time_distribution).strip().lower()
        if train_time_distribution not in {"uniform", "logitnormal"}:
            raise ValueError(
                f"`train_time_distribution` must be one of ['uniform', 'logitnormal'], got {train_time_distribution!r}"
            )
        train_time_weight = str(train_time_weight).strip().lower()
        if train_time_weight == "reweighting":
            train_time_weight = "uniform"
        if train_time_weight != "uniform":
            raise ValueError(
                "Cosmos rectified flow currently only supports `train_time_weight='uniform'`, "
                f"got {train_time_weight!r}."
            )
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.train_time_distribution = train_time_distribution
        self.train_time_weight = train_time_weight

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        if self.train_time_distribution == "uniform":
            u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        else:
            u = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=torch.float32))
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        weight = torch.ones_like(timestep, dtype=torch.float32)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        model_output = model_output.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta
