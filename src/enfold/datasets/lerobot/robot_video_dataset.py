import hashlib
import json
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from enfold.utils.logging_config import get_logger
from enfold.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        observation_horizon: int = 1,
        video_only: bool = False,
        use_instruction_segments: bool = False,
        instruction_segment_track: str = "default",
        text_embedding_backend: str = "cosmos",
        text_embedding_cache_suffix: Optional[str] = None,
        episode_indices: Optional[list[int]] = None,
        episode_index_file: Optional[str] = None,
    ):
        self.video_only = bool(video_only)
        selected_episode_indices = self._load_episode_indices(episode_indices, episode_index_file)
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            video_only=self.video_only,
            use_instruction_segments=use_instruction_segments,
            instruction_segment_track=instruction_segment_track,
            episode_indices=selected_episode_indices,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"action_chunk_len must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"action_chunk_len // action_video_freq_ratio must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}. Valid action_chunk_len examples for ratio={self.action_video_freq_ratio}: 16, 32, 48."
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.text_embedding_backend = str(text_embedding_backend).strip().lower()
        self.text_embedding_cache_suffix = (
            str(text_embedding_cache_suffix).strip()
            if text_embedding_cache_suffix is not None and str(text_embedding_cache_suffix).strip()
            else self._default_text_embedding_cache_suffix(self.text_embedding_backend)
        )
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.observation_horizon = int(observation_horizon)
        if self.observation_horizon <= 0:
            raise ValueError(f"`observation_horizon` must be positive, got {self.observation_horizon}")

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if getattr(processor, "video_only", False):
                self.lerobot_dataset.set_processor(processor)
            else:
                if not pretrained_norm_stats:
                    if not is_training_set:
                        raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                    if PartialState().is_main_process:
                        logger.info("Calculating dataset stats for normalization...")
                        dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                        work_dir = misc.get_work_dir()
                        save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                    else:
                        dataset_stats = None
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        obj_list = [dataset_stats]
                        torch.distributed.broadcast_object_list(obj_list, src=0)
                        dataset_stats = obj_list[0]
                else:
                    dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                    logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                    if PartialState().is_main_process:
                        work_dir = misc.get_work_dir()
                        stats_dest = os.path.join(work_dir, "dataset_stats.json")
                        stats_src = os.path.abspath(os.path.expanduser(str(pretrained_norm_stats)))
                        if stats_src != os.path.abspath(stats_dest):
                            save_dataset_stats_to_json(dataset_stats, stats_dest)

                processor.set_normalizer_from_stats(dataset_stats)
                self.lerobot_dataset.set_processor(processor)
        
    @staticmethod
    def _load_episode_indices(episode_indices, episode_index_file: Optional[str]) -> Optional[list[int]]:
        if episode_indices is not None and episode_index_file is not None:
            raise ValueError("Use only one of `episode_indices` or `episode_index_file`, not both.")
        if episode_indices is not None:
            return [int(idx) for idx in episode_indices]
        if episode_index_file is None or not str(episode_index_file).strip():
            return None
        path = os.path.expanduser(str(episode_index_file))
        if not os.path.exists(path):
            raise FileNotFoundError(f"episode_index_file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                payload = json.load(f)
                if isinstance(payload, dict):
                    payload = payload.get("episode_indices", payload.get("episodes"))
                if payload is None:
                    raise ValueError(f"JSON episode index file must contain a list or an `episode_indices` field: {path}")
                return [int(idx) for idx in payload]
            return [int(line.strip()) for line in f if line.strip()]

    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            image_is_pad = sample["image_is_pad"]
            has_pad = False
            if bool(image_is_pad.any().item()):
                has_pad = True
            if not self.video_only:
                action_is_pad = sample["action_is_pad"]
                proprio_is_pad = sample["proprio_is_pad"]
                if bool(action_is_pad.any().item()):
                    has_pad = True
                if bool(proprio_is_pad.any().item()):
                    has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        raw_image_is_pad = sample["image_is_pad"]
        raw_video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]

        video, image_is_pad = self._prepare_video(raw_video, self.video_sample_indices, raw_image_is_pad)
        observation_indices = self.video_sample_indices[: self.observation_horizon]
        observation_video, observation_is_pad = self._prepare_video(raw_video, observation_indices, raw_image_is_pad)

        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "observation_video": observation_video,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "observation_is_pad": observation_is_pad,
        }
        if not self.video_only:
            # Proxy (from lerobot):
            #   action: [num_frames-1, action_dim]
            #   proprio: [num_frames, proprio_dim], aligned with video frames
            action = sample["action"]
            proprio = sample["proprio"][:-1, :]
            if action.shape[0] % (video.shape[1] - 1) != 0:
                raise ValueError(
                    f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
                )
            data.update(
                {
                    "action": action,
                    "proprio": proprio,
                    "action_is_pad": sample["action_is_pad"],
                    "proprio_is_pad": sample["proprio_is_pad"],
                }
            )
        return data

    def _prepare_video(self, raw_video: torch.Tensor, frame_indices: list[int], raw_image_is_pad: torch.Tensor):
        if raw_video.ndim == 5:
            video = raw_video[:, frame_indices, :, :, :]  # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert raw_video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {raw_video.shape}"
            video = raw_video[frame_indices, :, :, :]  # [T_video, C, H, W]
            T_video, C, H, W = video.shape
            num_cameras = 1
        image_is_pad = raw_image_is_pad[frame_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        return video.permute(1, 0, 2, 3), image_is_pad

    @staticmethod
    def _default_text_embedding_cache_suffix(text_embedding_backend: str) -> str:
        backend = str(text_embedding_backend).strip().lower()
        if backend == "cosmos":
            return "cosmos"
        raise ValueError(
            f"Unsupported text_embedding_backend={text_embedding_backend!r}; expected 'cosmos'."
        )

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.{self.text_embedding_cache_suffix}.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
