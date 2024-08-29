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
from dataclasses import dataclass, field


@dataclass
class HPTConfig:
    """Configuration class for the Heterogeneous Pre-trained Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:

    """

    # Input / output structure.
    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.images.top": [3, 480, 640],
            "observation.state": [14],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [14],
        }
    )

    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.top": "mean_std",
            "observation.state": "mean_std",
        }
    )
    output_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "action": "mean_std",
        }
    )

    # Architecture.
    # Vision backbone.
    domain_name: str = "robotics"
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    # Network configuration
    embed_dim: int = 256  # Transformer model size
    num_blocks: int = 16  # Number of blocks in the trunk transformer
    num_heads: int = 8  # Number of heads in the trunk transformer
    use_modality_embedding: bool = True  # Whether to add modality-specific trainable parameters
    use_domain_embedding: bool = False  # Whether to add domain-specific trainable parameters
    token_postprocessing: str = "mean"  # maxpool or meanpool the tokens
    weight_init_style: str = "pytorch"  # Weight initialization style
    drop_path: float = 0.1  # Drop path in the trunk transformer
    use_gpt_trunk: bool = False  # Load pre-trained trunk from GPT2
    use_llama_trunk: bool = False  # Load pre-trained trunk from LLaMA2
    hf_trunk: str = ""  # Load pre-trained transformer from huggingface

    # Stem network (projectors) for different modalities
    modalities: list = ["image", "state"]  # Modalities (e.g., 'image', 'language')
    modality_embed_dim: int = embed_dim  # Embedding dimension for each modality
    normalize_state: bool = True  # Normalize state vectors
    state_embedding_dim: int = 1  # Dimension of positional encoding for state
    image_encoder: str = "resnet"  # Default image encoder
    crossattn_dim_head: int = 64  # Dimension of each head in cross attention modules
    crossattn_heads: int = 8  # Number of heads in cross attention
    crossattn_modality_dropout: float = 0.1  # Dropout ratio for cross attention
    observation_horizon: int = 2  # Observation horizon
    random_horizon_masking: bool = True  # Randomize observation input length
    add_pos_embedding_to_state: bool = False  # Positional embedding for the state
    stem_num_blocks: int = 1  # Number of blocks for stem transformer's cross and self attention

    crossattn_latent_image: int = 16  # Latent dimension for cross attention (image)
    crossattn_latent_state: int = 16  # Latent dimension for cross attention (state)

    image_input_dim: int = 512  # Input dimension for the image encoder
    image_output_dim: int = embed_dim  # Output dimension for the image encoder
    image_widths: list = [128]  # Widths of the layers for the image encoder
    image_num_of_copy: int = 1  # Number of copies for the image encoder

    state_input_dim: int = 0  # Placeholder, should be overwritten based on the environment state dimension
    state_output_dim: int = embed_dim  # Output dimension for the state encoder
    state_widths: list = [128]  # Widths of the layers for the state encoder

    # Head network
    head_input_dim: int = embed_dim  # Input dimension for the head network
    tanh_end: bool = True  # Whether to apply tanh to normalize action output
    action_dim: int = 0  # Placeholder, should be overwritten based on the environment action dimension
    action_horizon: int = 4  # Action horizon, should be overwritten based on the dataset
    dropout: bool = True  # Add dropout to the head network
    head_widths: list = [256, 128]  # Widths of the layers for the head network

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if (
            not any(k.startswith("observation.image") for k in self.input_shapes)
            and "observation.environment_state" not in self.input_shapes
        ):
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
