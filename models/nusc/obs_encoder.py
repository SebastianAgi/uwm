"""Pointcloud-only observation encoder for NuScenes variant.

Replaces image-based encoding (ViT/ResNet + video transforms + VAE) with a
lightweight pointcloud projector. Current observations are projected to a
feature vector of size ``embed_dim``; next observations are assumed to already
be pointcloud embeddings (or raw grids) and are simply reshaped to the latent
video format expected downstream: ``(B, V, C, T, H, W)``.

Supported input tensor shapes for pointcloud entries in ``curr_obs_dict`` and
``next_obs_dict``:
    - (B, C, H, W): single view, single timestep
    - (B, T, C, H, W): single view, multi-timestep (``pointcloud_with_time=True``)

If a time dimension is present for current obs, each frame is projected and
the embeddings are averaged across time (configurable if you later want last
frame or concat).

Multiple pointcloud keys are not yet concatenated; the first key is used. This
can be extended easily by iterating over ``self.pc_keys`` and aggregating.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from models.common.language import CLIPTextEncoder
from .pointcloud_encoder import PointCloudProjector


class UWMObservationEncoder(nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        num_frames: int,
        embed_dim: int,
        use_low_dim: bool = True,
        use_language: bool = True,
        # Pointcloud configuration
        use_pointcloud: bool = True,
        pointcloud_in_chans: int = 16,
        pointcloud_key: Optional[str] = None,
        pointcloud_with_time: bool = False,
        projector_hidden_chans: int = 128,
        aggregation: str = "mean",  # how to aggregate time dimension: mean|max|last
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.num_frames = num_frames
        # No RGB keys (image encoder removed)
        self.rgb_keys = []
        self.low_dim_keys = sorted(
            [k for k, v in shape_meta["obs"].items() if v["type"] == "low_dim"]
        )
        self.pc_keys = (
            [pointcloud_key]
            if pointcloud_key is not None
            else sorted(
                [k for k, v in shape_meta["obs"].items() if v.get("type") == "pointcloud"]
            )
        )
        if use_pointcloud and not self.pc_keys:
            raise ValueError("use_pointcloud=True but no pointcloud key found in shape_meta['obs'].")

        self.embed_dim = embed_dim
        self.use_low_dim = use_low_dim
        self.use_language = use_language
        self.use_pointcloud = use_pointcloud
        self.pointcloud_with_time = pointcloud_with_time
        self.aggregation = aggregation

        # Language encoder
        self.text_encoder = CLIPTextEncoder(embed_dim=embed_dim) if use_language else None

        # Pointcloud projector (always present when use_pointcloud True)
        self.pc_projector = (
            PointCloudProjector(
                in_chans=pointcloud_in_chans,
                hidden_chans=projector_hidden_chans,
                out_dim=embed_dim,
            )
            if use_pointcloud
            else None
        )

        # Track latent "image" shape for downstream models expecting (T,H,W) sizes.
        # We treat pointcloud grid as (C,H,W) with a single view; time dimension may vary.
        # If time exists, we keep its length; else use 1.
        self._latent_time = shape_meta.get("pointcloud_time", 1)
        self._latent_hw = shape_meta.get("pointcloud_hw", (50, 50))

    # ----------------------------- Core utilities ----------------------------- #
    def _project_pointcloud(self, pc: torch.Tensor) -> torch.Tensor:
        """Project a pointcloud (B,C,H,W) or (B,T,C,H,W) to (B, embed_dim)."""
        if pc.dim() == 5:  # (B,T,C,H,W)
            B, T, C, H, W = pc.shape
            pc_flat = pc.reshape(B * T, C, H, W)
            embeds = self.pc_projector(pc_flat)  # (B*T, D)
            embeds = embeds.view(B, T, -1)
            if self.aggregation == "mean":
                embeds = embeds.mean(1)
            elif self.aggregation == "max":
                embeds, _ = embeds.max(1)
            elif self.aggregation == "last":
                embeds = embeds[:, -1]
            else:
                raise ValueError(f"Unsupported aggregation: {self.aggregation}")
            return embeds  # (B, D)
        elif pc.dim() == 4:  # (B,C,H,W)
            return self.pc_projector(pc)  # (B, D)
        else:
            raise ValueError(f"Unexpected pointcloud shape {pc.shape}; expected 4D or 5D tensor.")

    def _format_next_pointcloud(self, pc: torch.Tensor) -> torch.Tensor:
        """Format next pointcloud embeddings to (B, V, C, T, H, W).

        Assumptions:
          - Single view (V=1). If pc has time dim (B,T,C,H,W), we keep T; else set T=1.
          - Input channel layout already matches expected latent channels.
        """
        if pc.dim() == 5:  # (B,T,C,H,W)
            pc = rearrange(pc, "b t c h w -> b 1 c t h w")
        elif pc.dim() == 4:  # (B,C,H,W)
            pc = rearrange(pc, "b c h w -> b 1 c 1 h w")
        else:
            raise ValueError(
                f"Unexpected next pointcloud shape {pc.shape}; expected 4D or 5D tensor. "
                "If you have a different format, adapt _format_next_pointcloud."
            )
        return pc

    # ----------------------------- Encoding APIs ----------------------------- #
    def apply_transform(self, obs_dicts: Union[dict, list[dict]]):  # retained for interface compatibility
        """Identity transform placeholder for pointclouds.

        Returns the raw pointcloud tensors wrapped to match image-based caller expectations
        only if needed by downstream code. For current implementation we just return the
        original structure.
        """
        return obs_dicts

    def apply_vae(self, *args, **kwargs):  # retained for interface compatibility
        raise NotImplementedError("VAE path removed in pointcloud-only encoder.")

    def encode_curr_obs(self, curr_obs_dict: dict):
        """Encode current observation dict to feature vector.

        Uses pointcloud projector, optionally concatenating low-dim and language features.
        """
        if not self.use_pointcloud:
            raise RuntimeError("Pointcloud encoding requested but use_pointcloud=False.")
        pc = curr_obs_dict[self.pc_keys[0]]  # (B,C,H,W) or (B,T,C,H,W)
        curr_feats = self._project_pointcloud(pc)  # (B, D)

        if self.use_low_dim and self.low_dim_keys:
            low_dims = [curr_obs_dict[key] for key in self.low_dim_keys]
            low_dims = torch.cat(low_dims, dim=-1).flatten(1)
            curr_feats = torch.cat([curr_feats, low_dims], dim=-1)

        if self.use_language and self.text_encoder is not None:
            lang_feats = self.text_encoder(
                input_ids=curr_obs_dict["input_ids"],
                attention_mask=curr_obs_dict["attention_mask"],
            )
            curr_feats = torch.cat([curr_feats, lang_feats], dim=-1)
        return curr_feats

    def encode_next_obs(self, next_obs_dict: dict):
        """Format next observation pointcloud(s) as latent videos without VAE.

        Expects raw or already-embedded pointcloud grid(s) under the first pc key.
        """
        pc = next_obs_dict[self.pc_keys[0]]
        next_latents = self._format_next_pointcloud(pc)  # (B,1,C,T,H,W)
        return next_latents

    def encode_curr_and_next_obs(self, curr_obs_dict: dict, next_obs_dict: dict):
        """Jointly encode current features and format next latents.

        Combines logic of ``encode_curr_obs`` and ``encode_next_obs`` without applying
        image transforms or VAE.
        """
        curr_feats = self.encode_curr_obs(curr_obs_dict)
        next_latents = self.encode_next_obs(next_obs_dict)
        return curr_feats, next_latents

    def feat_dim(self):
        """Return dimension of encoded current features (pointcloud + optional extras)."""
        low_dim_size = sum(
            self.shape_meta["obs"][key]["shape"][-1] for key in self.low_dim_keys
        ) if self.low_dim_keys else 0
        return (
            int(self.use_pointcloud) * self.embed_dim
            + int(self.use_low_dim) * self.num_frames * low_dim_size
            + int(self.use_language) * self.embed_dim
        )

    def latent_img_shape(self):
        """Return latent video shape (V, C, T, H, W) for pointcloud latents.

        Uses a dummy zero tensor with the detected pointcloud spatial shape.
        """
        if not self.pc_keys:
            raise RuntimeError("latent_img_shape requested but no pointcloud keys present.")
        # Build dummy next obs
        pc_shape = self.shape_meta["obs"][self.pc_keys[0]]["shape"]  # (C,H,W) or (T,C,H,W)
        if len(pc_shape) == 3:  # (C,H,W)
            C, H, W = pc_shape
            dummy = torch.zeros(1, C, H, W)
        elif len(pc_shape) == 4:  # (T,C,H,W)
            T, C, H, W = pc_shape
            dummy = torch.zeros(1, T, C, H, W)
        else:
            raise ValueError(f"Unexpected stored pointcloud shape {pc_shape} in shape_meta.")
        latent = self._format_next_pointcloud(dummy)
        return tuple(latent.shape[1:])  # (V,C,T,H,W)
