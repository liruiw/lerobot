#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Heterogeneous Pre-trained Transformer Policy

As per Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers (TBU).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

from collections import defaultdict
from functools import partial
from itertools import chain
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, einsum, nn
from transformers import T5Model, T5Tokenizer

from lerobot.common.policies.hpt.configuration_hpt import HPTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize

LOSS = partial(F.smooth_l1_loss, beta=0.05)
INIT_CONST = 0.02


class HPTPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "act"],
):
    """
    Heterogeneous Pre-trained Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: TBU, code: https://github.com/liruiw/HPT-Transfer)
    """

    name = "act"

    def __init__(
        self,
        config: HPTConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = HPTConfig()

        self.config: HPTConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.model = HPT(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self.history_buffer = defaultdict(list)

        # current steps in open-loop rollouts
        self.openloop_traj_step = self.action_horizon - 1
        self.language_embedding = None

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], domain=None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self.normalize_inputs(batch)

        def update_history_buffer(key: str, new_obs: torch.Tensor):
            """Update the history buffer with a new observation.

            Args:
                key (str): The key for the observation.
                new_obs (torch.Tensor): The new observation tensor.
            """
            # act like a deque
            self.history_buffer[key].append(new_obs)
            if len(self.history_buffer[key]) > self.observation_horizon:
                self.history_buffer[key].pop(0)

        if domain is None:  # default
            domain = self.domains[0]

        if not hasattr(self, "history_buffer"):
            print("should call policy reset explicitly to avoid problems for evaluation in sequence.")
            self.reset()

        action_dim = len(self.normalizer[domain].params_dict["action"]["input_stats"].min)
        device = next(self.parameters()).device

        if self.openloop_traj_step != self.action_horizon - 1:
            # use previous predictions in open-loop execution
            self.openloop_traj_step += 1
        else:
            data_noimg = {k: v for k, v in batch.items() if "image" not in k}
            data_img = {k: v for k, v in batch.items() if "image" in k}

            # append batch and T dimensions
            data_th = dict_apply(data_noimg, lambda x: torch.FloatTensor(x)[None, None].to(device).float())

            # handle multi-views and history in image
            img_queue = []
            for img_key in data_img:
                if self.stem_spec.precompute_feat and "image" not in self.encoders:
                    # precomputed
                    img_embedding = get_resnet_embeddings(batch[img_key], self.stem_spec.image_encoder)
                    img_queue.append(torch.FloatTensor(img_embedding).to(device).float())
                else:
                    # raw inputs
                    image = normalize_image_numpy(batch[img_key])
                    img_queue.append(torch.FloatTensor(image).to(device).float())
            update_history_buffer("image", torch.cat(img_queue, dim=0))  # concat in channel for views
            data_th["image"] = torch.stack(self.history_buffer["image"], dim=0).float()[None]

            # handle state and language
            for modality in data_noimg:
                update_history_buffer(modality, data_th[modality])

                # language is the same for the whole trajectory
                if "language" in modality:
                    if "language" in self.modalities:
                        if self.language_embedding is None:
                            self.language_embedding = get_t5_embeddings(data_th[modality], per_token=True)
                        data_th[modality] = self.language_embedding
                else:
                    data_th[modality] = torch.cat(self.history_buffer[modality], dim=1).float()

            # handle previous actions
            if "prev_actions" in self.history_buffer:
                data_th["prev_actions"] = torch.cat(self.history_buffer["prev_actions"], dim=1).float()

            action_th = self(domain, data_th)  # forward pass
            self.action_traj = action_th.detach().cpu().numpy()[0]  # batch=1
            self.action_traj = self.action_traj.reshape(-1, action_dim)  # T x Da
            self.openloop_traj_step = 0  # reset steps

        # handle previous actions
        curr_action = self.action_traj[self.openloop_traj_step]
        update_history_buffer(
            "prev_actions", torch.FloatTensor(curr_action)[None, None, None].to(device).float()
        )
        return curr_action

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}

        return loss_dict


class HPT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: HPTConfig):
        super().__init__()
        self.config = config
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.use_env_state = "observation.environment_state" in config.input_shapes

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.parameters()):
            if p.dim() > 1:
                torch.nn.init.trunc_normal_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the HPT

        `batch` should have the following structure:
        {
            "observation.state" (optional): (B, state_dim) batch of robot states.

            "observation.images": (B, n_cameras, C, H, W) batch of images.
                AND/OR
            "observation.environment_state": (B, env_dim) batch of environment states.

            "action" (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        pass

    def preprocess_tokens(self, domain: str, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Shared modality layers and add modality tokens. Add positional and time embeddings.
        """
        processed_features = []
        for modality, feature in zip(self.modalities, features, strict=False):
            modality_embedding = self.modalities_tokens[modality].repeat(
                (feature.shape[0], feature.shape[1], 1)
            )
            feature = feature + modality_embedding
            processed_features.append(feature)

        tokens = torch.cat(processed_features, dim=-2)
        if self.use_domain_embedding:
            domains_tokens = self.domains_tokens[domain].repeat(len(tokens), 1, 1)
            tokens = torch.cat([tokens, domains_tokens], dim=-2)

        if self.token_postprocessing == "action_token":
            action_tokens = self.action_tokens[domain].repeat(len(tokens), 1, 1)
            tokens = torch.cat([tokens, action_tokens], dim=-2)

        position_tokens = self.get_position_embedding(tokens, self.embed_dim)
        tokens = tokens + position_tokens
        return tokens

    def postprocess_tokens(self, trunk_tokens: torch.Tensor) -> torch.Tensor:
        """
        Postprocesses the trunk tokens to obtain the final features.

        Args:
            trunk_tokens (torch.Tensor): The trunk tokens of shape (N, L, D), where N is the batch size,
                                        L is the sequence length, and D is the token dimension.

        Returns:
            torch.Tensor: The postprocessed tokens of shape (N, D), where N is the batch size and D is the
                          final feature dimension.
        """
        if self.token_postprocessing == "mean":
            return trunk_tokens.mean(dim=1)
        elif self.token_postprocessing == "action_token":
            return trunk_tokens[:, -self.action_horizon :]
        elif self.token_postprocessing == "max":
            return trunk_tokens.max(dim=1)[0]
        elif self.token_postprocessing == "last":
            return trunk_tokens[:, -1]
        elif self.token_postprocessing == "attentive":
            return self.attentive_pool(trunk_tokens)[:, 0]
        else:
            raise ValueError(
                "Invalid token_postprocessing value. Must be one of ['mean', 'action_token', 'max', 'last', 'attentive']."
            )


class MLP:
    """Simple MLP based policy head"""

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = (512, 512),
        dropout: bool = False,
        tanh_end: bool = False,
        ln: bool = True,
        **kwargs,
    ) -> None:
        """vanilla MLP head on the pooled feature"""
        super().__init__()
        self.input = input
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if dropout:
                modules.append(nn.Dropout(p=0.1))
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the policy head module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        y = self.net(x)
        return y

    def compute_loss(self, x: torch.Tensor, data: dict) -> torch.Tensor:
        self.target_action = data["action"]
        self.pred_action = self(x).view(self.target_action.shape)
        return LOSS(self.pred_action, self.target_action)


class PolicyStem(nn.Module):
    """policy stem"""

    def __init__(self, **kwargs):
        super().__init__()

    def init_cross_attn(self, stem_spec, modality: str):
        """initialize cross attention module and the learnable tokens"""
        token_num = getattr(stem_spec.crossattn_latent, modality)
        self.tokens = nn.Parameter(torch.randn(1, token_num, stem_spec.modality_embed_dim) * INIT_CONST)

        self.cross_attention = CrossAttention(
            stem_spec.modality_embed_dim,
            heads=stem_spec.crossattn_heads,
            dim_head=stem_spec.crossattn_dim_head,
            dropout=stem_spec.crossattn_modality_dropout,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the latent representations of input data by attention.

        Args:
            Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.

        Returns:
            Output tensor with latent tokens, shape [32, 16, 128], where 16 is the number
            of tokens and 128 is the dimensionality of each token.

        Examples for vision features from ResNet:
        >>> x = np.random.randn(32, 3, 1, 49, 512)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)

        Examples for proprioceptive features:
        >>> x = np.random.randn(32, 3, 1, 7)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)
        """
        # Initial reshape to adapt to token dimensions
        # (32, 3, 1, 49, 128)
        stem_feat = self(x)
        stem_feat = stem_feat.reshape(stem_feat.shape[0], -1, stem_feat.shape[-1])  # (32, 147, 128)
        # Replicating tokens for each item in the batch and computing cross-attention
        stem_tokens = self.tokens.repeat(len(stem_feat), 1, 1)  # (32, 16, 128)
        stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (32, 16, 128)
        return stem_tokens


class MLPStem(PolicyStem):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = (512, 512),
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
        **kwargs,
    ) -> None:
        """vanilla MLP class"""
        super().__init__()
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList([nn.Sequential(*modules) for _ in range(num_of_copy)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size,
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        if self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            y = torch.stack(out, dim=1)
        else:
            y = self.net(x)
        return y


class CrossAttention(nn.Module):
    """
    CrossAttention module used in the Perceiver IO model.

    Args:
        query_dim (int): The dimension of the query input.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): The query input tensor.
            context (torch.Tensor): The context input tensor.
            mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> (b h) n d", h=h)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=h)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=h)

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            # fill in the masks with negative values
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        drop: float = 0.0,
    ):
        """
        Initialize the Transformer model.

        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features. Defaults to None.
            out_features (int, optional): Number of output features. Defaults to None.
            act_layer (torch.nn.Module, optional): Activation layer. Defaults to nn.GELU.
            drop (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BlockWithMasking(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 1e-4,
    ):
        super().__init__()

        assert not isinstance(
            attn_target, nn.Module
        ), "attn_target should be a Callable. Otherwise attn_target is shared across blocks!"
        self.attn = attn_target()
        # if drop_path > 0.0:
        #     requires timm package
        #     self.drop_path = DropPath(drop_path)
        # else:
        self.drop_path = nn.Identity()
        self.norm_1 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
        self.norm_2 = norm_layer(dim)
        self.layer_scale_type = layer_scale_type
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask))
            x = x + self.drop_path(self.mlp(self.norm_2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask)) * self.layer_scale_gamma1
            x = x + self.drop_path(self.mlp(self.norm_2(x))) * self.layer_scale_gamma2
        return x


_LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


class MultiheadAttention(nn.MultiheadAttention):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return super().forward(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        attn_target: Callable,
        embed_dim: int,
        num_blocks: int,
        block: Callable = BlockWithMasking,
        pre_transformer_layer: Optional[Callable] = None,
        post_transformer_layer: Optional[Callable] = None,
        drop_path_rate: float = 0.0,
        drop_path_type: str = "progressive",
        norm_layer: Callable = _LAYER_NORM,
        mlp_ratio: int = 4,
        ffn_dropout_rate: float = 0.0,
        layer_scale_type: Optional[
            str
        ] = None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value: float = 1e-4,  # from cait; float
        weight_init_style: str = "pytorch",  # possible values jax or pytorch
    ):
        """
        Simple Transformer with the following features
        1. Supports masked attention
        2. Supports DropPath
        3. Supports LayerScale
        4. Supports Dropout in Attention and FFN
        5. Makes few assumptions about the input except that it is a Tensor
        """
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_blocks)]
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")

        self.blocks = nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_blocks)
            ]
        )
        self.post_transformer_layer = post_transformer_layer
        self.weight_init_style = weight_init_style
        self.apply(self._init_weights)

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor = None,
        use_checkpoint: bool = False,
        checkpoint_every_n: int = 1,
        checkpoint_blk_ids: Optional[List[int]] = None,
    ):
        """
        Inputs
        - tokens: data of shape N x L x D (or L x N x D depending on the attention implementation)
        - attn: mask of shape L x L

        Output
        - x: data of shape N x L x D (or L x N x D depending on the attention implementation)
        """
        if self.pre_transformer_layer:
            tokens = self.pre_transformer_layer(tokens)
        if use_checkpoint and checkpoint_blk_ids is None:
            checkpoint_blk_ids = [
                blk_id for blk_id in range(len(self.blocks)) if blk_id % checkpoint_every_n == 0
            ]
        if checkpoint_blk_ids:
            checkpoint_blk_ids = set(checkpoint_blk_ids)
        for _, blk in enumerate(self.blocks):
            tokens = blk(tokens, attn_mask=attn_mask)
        if self.post_transformer_layer:
            tokens = self.post_transformer_layer(tokens)
        return tokens

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.weight_init_style == "jax":
                # Based on MAE and official Jax ViT implementation
                torch.nn.init.xavier_uniform_(m.weight)

            elif self.weight_init_style == "pytorch":
                # PyTorch ViT uses trunc_normal_
                torch.nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return rearrange(x, self.rearrange_expr, **self.kwargs)


def get_sinusoid_encoding_table(position_start, position_end, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(position_start, position_end)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def dict_apply(x, func):
    """
    Apply a function to all values in a dictionary recursively.

    Args:
        x (dict or any): The dictionary or value to apply the function to.
        func (function): The function to apply.

    Returns:
        dict or any: The resulting dictionary or value after applying the function.
    """
    dict_type = type(x)
    if type(x) is not dict_type:
        return func(x)

    result = dict_type()
    for key, value in x.items():
        if isinstance(value, dict_type):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def normalize_image_numpy(image: np.ndarray, resize: bool = True) -> np.ndarray:
    """
    Normalize an image in numpy format.

    Args:
        image (numpy.ndarray): The input image in H x W x 3 (uint8) format.

    Returns:
        numpy.ndarray: The normalized image in 3 x H x W format.

    Notes:
        - The input image is resized to (224, 224) using cv2.resize.
        - The image is normalized using the mean and standard deviation values from the ImageNet dataset.
        - The resulting image is transposed to have dimensions of 3 x H x W.
    """
    if resize:
        image = cv2.resize(image, (224, 224))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image / 255.0

    # convert to array
    image = np.asarray(image)

    # normalize
    image = (image - mean) / std
    return image.transpose(2, 0, 1)


@torch.no_grad()
def get_t5_embeddings(language, per_token=True, max_length=16, device="cpu"):
    """Get T5 embedding"""
    global global_language_model, global_language_processor
    if global_language_model is None:
        global_language_model = T5Model.from_pretrained("t5-base").to(device)
        global_language_processor = T5Tokenizer.from_pretrained("t5-base")

    # forward pass through encoder only
    enc = global_language_processor(
        [language],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)

    output = global_language_model.encoder(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True
    )
    torch.cuda.empty_cache()
    if per_token:
        return output.last_hidden_state[0].detach().cpu().numpy()
    else:
        # get the final hidden states. average across tokens.
        emb = output.last_hidden_state[0].mean(dim=0).detach().cpu().numpy()
        return emb


@torch.no_grad()
def get_resnet_embeddings(image, per_token=False, device="cuda", downsample=False):
    """Get Resnet embedding. Input: H x W x 3"""
    global global_vision_model

    if global_vision_model is None:
        global_vision_model = ResNet().to(device)
    device = global_vision_model.device
    image = normalize_image_numpy(image)
    global_vision_model.eval()
    image_th = torch.FloatTensor(image).to(device)
    if len(image_th.shape) == 3:
        image_th = image_th[None]

    # forward pass through encoder only
    output = global_vision_model.net(image_th)
    if downsample:  # pool to 3 x 3
        output = torch.nn.functional.avg_pool2d(output, 2, 2)

    output = output.reshape(output.shape[0], 512, -1).transpose(1, 2)
    return output.detach().cpu().numpy()


class ResNet(PolicyStem):
    def __init__(
        self,
        output_dim: int = 10,
        weights: str = "DEFAULT",
        resnet_model: str = "resnet18",
        num_of_copy: int = 1,
        **kwargs,
    ) -> None:
        """ResNet Encoder for Images"""
        super().__init__()
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)

        # by default we use a separate image encoder for each view in downstream evaluation
        self.num_of_copy = num_of_copy
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])

        if num_of_copy > 1:
            self.net = nn.ModuleList(
                [nn.Sequential(*list(pretrained_model.children())[:-2]) for _ in range(num_of_copy)]
            )
        self.input = input
        self.out_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size,
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        b, *_, h, w = x.shape
        x = x.view(len(x), -1, 3, h, w)
        if self.num_of_copy > 1:
            # separate encoding for each view
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            feat = torch.stack(out, dim=1)
        else:
            x = x.view(-1, 3, h, w)
            feat = self.net(x)
        # concat along time
        feat = feat.reshape(b, -1, feat.shape[-1]).contiguous()
        return feat
