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
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, einsum, nn

from lerobot.common.policies.hpt.configuration_hpt import HPTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize

LOSS = partial(F.smooth_l1_loss, beta=0.05)
INIT_CONST = 0.02
_LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


class HPTPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "hpt"],
):
    """
    Heterogeneous Pre-trained Transformer Policy as per Scaling Proprioceptive-Visual Learning
    with Heterogeneous Pre-trained Transformers  (paper: TBU, code: https://github.com/liruiw/HPT-Transfer)
    """

    name = "hpt"

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
        self.openloop_traj_step = self.config.Network.action_horizon - 1
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
        if domain is None:  # default
            domain = self.domains[0]

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

        if not hasattr(self, "history_buffer"):
            print("should call policy reset explicitly to avoid problems for evaluation in sequence.")
            self.reset()

        if self.openloop_traj_step != self.config.netwaction_horizon - 1:
            # use previous predictions in open-loop execution
            self.openloop_traj_step += 1
        else:
            batch_with_history = {}

            # handle state and language
            for modality, data in batch.items():
                update_history_buffer(modality, data)
                batch_with_history[modality] = torch.cat(self.history_buffer[modality], dim=1).float()

            action_th = self.model(domain, batch_with_history)  # forward pass to generate action
            action_th = self.unnormalize_outputs({"action": action_th})["action"]
            self.action_traj = action_th.detach().cpu().numpy()[0]  # batch=1
            self.action_traj = self.action_traj.reshape(-1, self.config.head.action_dim)  # T x Da
            self.openloop_traj_step = 0  # reset steps

        curr_action = self.action_traj[self.openloop_traj_step]
        return curr_action

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        batch = self.normalize_targets(batch)
        loss = self.model.compute_loss(batch)
        loss_dict = {"loss": loss}
        return loss_dict


class HPT(nn.Module):
    """Heterogeneous Pre-trained Transformer: The underlying neural network for HPTPolicy.

    Note: In this code we use the terms `stem`, 'trunk', `head`. The meanings are as follows.
        -  The stem, consisting of a proprioception tokenizer and a vision tokenizer, maps the
          vision and proprioception observations of different embodiments to a fixed number (e.g. 16) of tokens.
        -  The shared trunk, which is a Transformer, maps the concatenated tokens into shared representations.
        -  The head then maps the processed tokens to actions in different downstream tasks.
        For a specific embodiment, one stem/head pair is activated.

            ┌───────────────────────────────────────┐
            |               Outputs                 |
            |                  ▲                    |
            |             ┌───────┐                 |
            |             | Head. │                 |
            │             └──▲────┘                 │
            │                │                      │
            │            ┌───────┐                  │
            │            | Trunk.│                  │
            │            │ Tranf.│                  │
            │            │       │                  │
            │ ┌───────┬  │       │                  │
            │ │       │  └──▲────┘                  │
            │ │Stem.  │     │                       │
            │ │encoder│ ────┘                       │
            │ └▲──▲─▲─┘                             │
            │  │  │ │                               │
            │ image emb.       ...        ...       │
            │    state emb.                         │
            └───────────────────────────────────────┘
    """

    def __init__(self, config: HPTConfig):
        super().__init__()
        self.config = config
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.use_env_state = "observation.environment_state" in config.input_shapes
        self.embed_dim = config.Network.embed_dim
        self.no_trunk = config.Network.no_trunk

        self.trunk = self._create_policy_trunk(
            config.Network.embed_dim, config.Network.num_blocks, config.Network.num_heads
        )
        self.stems = {}
        self.heads = {}

        # self.normalizer = {}
        self.encoders = {}
        self.domains = []
        self.use_modality_embedding = config.Network.use_modality_embedding
        self.observation_horizon = config.Network.observation_horizon
        self.action_horizon = config.Network.action_horizon
        self.token_postprocessing = config.Network.token_postprocessing
        self.use_domain_embedding = config.Network.use_domain_embedding
        self.modalities_tokens = {}
        self.domains_tokens = {}
        self.action_tokens = {}

        # initialize modules.
        self.init_encoders("image", ResNet())
        self.init_domain_stem(self.config.domain_name)
        self.init_domain_head(self.config.domain_name)
        self.finalize_modules()

    def _init_weights(self, m):
        """
        Weight initialization for transformer
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_encoders(self, modality, encoder):
        """
        Add image/language encoders into the policy parameters in the case of joint finetuning
        """
        self.encoders[modality] = encoder
        self.encoders = nn.ModuleDict(self.encoders)

    def init_domain_stem(self, domain_name):
        """
        Initialize an observation stem for each domain
        """
        self.stem_spec = self.config.network
        self.modalities = self.stem_spec.modalities
        for modality in self.modalities:
            self.stems[domain_name + "_" + modality] = MLPStem(
                input_dim=getattr(self.stem_spec, modality + "_input_dim"),
                output_dim=getattr(self.stem_spec, modality + "_output_dim"),
                widths=getattr(self.stem_spec, modality + "_widths"),
                num_of_copy=getattr(self.stem_spec, modality + "_num_of_copy"),
            )

            self.stems[domain_name + "_" + modality].init_cross_attn(self.stem_spec, modality)
            self.modalities_tokens[modality] = nn.Parameter(
                torch.randn(1, 1, self.stem_spec.modality_embed_dim) * INIT_CONST
            )
        self.domains_tokens[domain_name] = nn.Parameter(torch.randn(1, 1, self.embed_dim) * INIT_CONST)
        if self.token_postprocessing == "action_token":
            self.action_tokens[domain_name] = nn.Parameter(
                torch.randn(1, self.action_horizon, self.embed_dim) * INIT_CONST
            )

    def init_domain_head(self, domain_name):
        """initialize an action head for each domain, along with normalizer"""
        self.head_spec = self.config.network
        self.action_horizon = self.head_spec.action_horizon
        self.domains.append(domain_name)
        self.heads[domain_name] = MLP(
            input_dim=self.head_spec.head_input_dim,
            output_dim=self.head_spec.head_action_dim * self.head_spec.action_horizon,
            widths=self.head_spec.head_widths,
        )

    def finalize_modules(self):
        """
        Finalizes the modules of the policy.

        This method converts the stems, heads, normalizer, modalities_tokens, domains_tokens,
        attentive_pool, and action_tokens into ModuleDict or ParameterDict objects, depending
        on the configuration. It also initializes the weights of the policy.
        """
        self.stems = nn.ModuleDict(self.stems)
        self.heads = nn.ModuleDict(self.heads)

        self.modalities_tokens = nn.ParameterDict(self.modalities_tokens)
        self.domains_tokens = nn.ParameterDict(self.domains_tokens)

        self.apply(self._init_weights)
        if self.token_postprocessing == "action_token":
            self.action_tokens = nn.ParameterDict(self.action_tokens)

    def _create_policy_trunk(
        self,
        embed_dim: int = 1024,
        num_blocks: int = 24,
        num_heads: int = 16,
        drop_path: float = 0.0,
        weight_init_style: str = "pytorch",
        **kwargs,
    ):
        """create the shared representation for pretraining"""

        def instantiate_trunk(embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
                weight_init_style=weight_init_style,
            )

        trunk = {}
        trunk["trunk"] = instantiate_trunk(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=drop_path,
        )
        return nn.ModuleDict(trunk)

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        self.apply(self._init_weights)

    def forward(
        self, batch: dict[str, Tensor], domain=""
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the HPT to generate actions at test time

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
        if len(domain) == 0:
            domain = self.config.domain_nam
        self.train_mode = False  # for random horizon masking
        features = self.forward_features(domain, batch)

        # head pass
        action = self.heads[domain](features)

        # postprocess. unnormalize the outputs
        # action = self.postprocess_actions(domain, action)
        return action

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

    def get_position_embedding(self, feature: torch.Tensor, embed_dim: int) -> torch.Tensor:
        """
        Add positional embedding to the features
        """
        if not hasattr(self, "cached_trunk_pos_embedding"):
            self.cached_trunk_pos_embedding = {}

        tokensize = feature.shape[1]
        if tokensize not in self.cached_trunk_pos_embedding:
            self.cached_trunk_pos_embedding[tokensize] = get_sinusoid_encoding_table(
                0, tokensize, self.embed_dim
            )
            self.cached_trunk_pos_embedding[tokensize] = (
                self.cached_trunk_pos_embedding[tokensize].repeat((1, 1, 1)).to(feature.device)
            )

        return self.cached_trunk_pos_embedding[tokensize]

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

    def mapped_modality_keys(self, modality: str, data: dict[str, Tensor]) -> bool:
        """Returns the corresponding the modality keys."""
        for k in data:
            if modality in k:
                return k
        return None

    def stem_process(self, domain: str, data: dict):
        """
        Pass through the stem to a fixed number of tokens.
        Args:
            data: dictionary of tensors of different modalities
        """
        feats = []

        for policy_modality in self.modalities:
            stem = self.stems[domain + "_" + policy_modality]
            modality = self.mapped_modality_keys(policy_modality, data)

            if not modality:
                print("skip modality", modality)
                continue

            # if len(data[modality].shape) == 4:
            # add time horizon and instance number
            data[modality] = data[modality][:, None, None]

            if "image" in modality and "image" in self.encoders:  # finetuning with encoders
                data[modality] = self.encoders["image"](data[modality])

            # positional embedding for observations
            data_shape = data[modality].shape
            data_horizon = data_shape[1]
            horizon = data_horizon

            if self.train_mode and self.stem_spec.random_horizon_masking and data_horizon > 1:
                horizon = np.random.randint(1, data_horizon + 1)
                data[modality] = data[modality][:, data_horizon - horizon : data_horizon]

            # data is N x T x M x ... x D where M is the # of instances for that sensor
            positional_embedding = get_sinusoid_encoding_table(
                0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]
            ).to(data[modality])
            positional_embedding = repeat(
                positional_embedding, "b h w -> (repeat b) h w", repeat=data_shape[0]
            )

            data[modality] = data[modality] + positional_embedding.view(data[modality].shape)
            stem_token = stem.compute_latent(data[modality])
            feats.append(stem_token)
            # we should make sure the state goes first and then the image

        return feats

    def forward_features(self, domain: str, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the features for the given domain and data.
        Args:
            domain (str): The domain of the data.
            data (Tensor): The input data.
        """
        # data = self.preprocess_states(domain, data)
        if len(domain) == 0:
            domain = self.config.domain_name

        # stem pass
        self.stem_tokens = self.stem_process(domain, data)

        # combine tokens
        self.trunk_tokens = self.preprocess_tokens(domain, self.stem_tokens)

        # trunk pass
        if not self.no_trunk:
            self.trunk_tokens = self.trunk["trunk"](self.trunk_tokens)

        # pooling the features
        return self.postprocess_tokens(self.trunk_tokens)

    def compute_loss(self, batch, domain=""):
        """Compute the loss for the training loop forward pass."""
        if len(domain) == 0:
            domain = self.config.domain_name

        self.train_mode = True
        # domain, data = batch["domain"][0], batch["data"]
        features = self.forward_features(domain, batch)

        # normalize the labels
        # if domain in self.normalizer:
        #     data["action"] = self.normalizer[domain]["action"].normalize(data["action"])

        # batch = self.normalize_targets(batch)
        # head pass
        loss = self.heads[domain].compute_loss(features, batch)
        return loss

    def load_trunk(self, path: str, postfix: str = "_last", extension: str = "pth"):
        """load the trunk part of the model"""
        if "hf://" in path:
            import huggingface_hub

            if "output" in path:
                path = path.replace("output/", "")
            path = huggingface_hub.snapshot_download(path[len("hf://") :])
            self.trunk.load_state_dict(torch.load(path, map_location="cpu"), strict=True)


class MLP(nn.Module):
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
        token_num = getattr(stem_spec, modality + "_crossattn_latent")
        self.tokens = nn.Parameter(torch.randn(1, token_num, stem_spec.modality_embed_dim) * INIT_CONST)

        self.cross_attention = CrossAttention(
            stem_spec.modality_embed_dim,
            heads=stem_spec.crossattn_heads,
            dim_head=stem_spec.crossattn_dim_head,
            dropout=stem_spec.crossattn_modality_dropout,
        )

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
        widths: Tuple[int] = (512, 512),
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
        **kwargs,
    ) -> None:
        """vanilla MLP class"""
        super().__init__()
        print("widths: ", widths, input_dim)
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
        x = x.view(-1, 3, h, w)
        # fixed image size
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.net(x).view(b, 512, -1).transpose(1, 2).contiguous()
        return feat
