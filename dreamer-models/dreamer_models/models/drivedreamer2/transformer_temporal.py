# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Optional,Dict,Any

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import BasicTransformerBlock,TemporalBasicTransformerBlock,_chunked_feed_forward,FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.resnet import AlphaBlender

# from diffusers.models.transformer_temporal import TransformerSpatioTemporalModel

@dataclass
class TransformerTemporalModelOutput(BaseOutput):
    """
    The output of [`TransformerTemporalModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size x num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input.
    """

    sample: torch.FloatTensor


class TransformerTemporalModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        attention_type="default",
        num_embeds_ada_norm: Optional[int] = None,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,  #
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,  #
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,  #
                    norm_type=norm_type,  #
                    norm_elementwise_affine=norm_elementwise_affine,
                    attention_type=attention_type,
                )
                for d in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)
        
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        num_frames=1,
        cross_attention_kwargs=None,
        return_dict: bool = True,
    ):
        """
        The [`TransformerTemporal`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input hidden_states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        residual = hidden_states

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)

        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                # hidden_states.reshape(1, hidden_states.shape[0]*hidden_states.shape[1], hidden_states.shape[2]),
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states[None, None, :]
            .reshape(batch_size, height, width, channel, num_frames)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)

class TransformerTemporalLight(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        attention_type="default",
        num_embeds_ada_norm: Optional[int] = None,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,  #
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,  #
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,  #
                    norm_type=norm_type,  #
                    norm_elementwise_affine=norm_elementwise_affine,
                    attention_type=attention_type,
                )
                for d in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)
        
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        num_frames=1,
        cross_attention_kwargs=None,
        return_dict: bool = True,
        att_seq = None
    ):
        """
        The [`TransformerTemporal`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input hidden_states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        residual = hidden_states

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)  # b c n h w

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)  # bhw n c

        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        # FIXME only att i-1 and i+1 frames
        if att_seq is None:
            att_seq = []
            for i in range(num_frames):
                att_seq.append(i-1)
                att_seq.append((i+1)%num_frames)
        kv = hidden_states[:, att_seq]
        for block in self.transformer_blocks:
            hidden_list = []
            for i_frame in range(num_frames):
                q_ = hidden_states[:, i_frame:i_frame+1].reshape(batch_size, int(hidden_states.shape[0]/batch_size), hidden_states.shape[2])
                kv_ = kv[:, 2*i_frame: 2*i_frame+2].reshape(batch_size, int(hidden_states.shape[0]/batch_size), 2, hidden_states.shape[2]).reshape(batch_size, int(hidden_states.shape[0]/batch_size)*2, hidden_states.shape[2])
                if self.training and self.gradient_checkpointing:
                    hidden_list.append(torch.utils.checkpoint.checkpoint(
                        (block),
                        q_, None, kv_, None,
                        timestep, cross_attention_kwargs,
                        class_labels,
                        use_reentrant=True
                    ).permute(1,0,2))
                else:
                    hidden_list.append(block(
                        q_,
                        encoder_hidden_states=kv_,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                    ).permute(1,0,2))
            # ([1792, 12, 320])
            hidden_states = torch.stack(hidden_list, dim=1).permute(2,0,1,3).reshape(batch_size*height * width, num_frames, -1)
            del hidden_list

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states
            .reshape(batch_size, height, width, num_frames, channel)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)

class TransformerMVTemporalModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        attention_type="default",
        num_embeds_ada_norm: Optional[int] = None,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_temp_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,  #
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,  #
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,  #
                    norm_type=norm_type,  #
                    norm_elementwise_affine=norm_elementwise_affine,
                    attention_type=attention_type,
                )
                for d in range(num_layers)
            ]
        )

        self.transformer_mv_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,  #
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,  #
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,  #
                    norm_type=norm_type,  #
                    norm_elementwise_affine=norm_elementwise_affine,
                    attention_type=attention_type,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        num_frames=1,
        cross_attention_kwargs=None,
        return_dict: bool = True,
    ):
        """
        The [`TransformerTemporal`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input hidden_states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, channel, height, width = hidden_states.shape
        num_cams = batch_frames // num_frames

        residual = hidden_states
        
        hidden_states = hidden_states[None, :].reshape(num_cams, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(1, 2, 0, 3, 4)  # n c cam h w
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(num_frames * height * width, num_cams, channel)
        hidden_states = self.proj_in(hidden_states)
        
        # 2. MV Blocks only att i-1 and i+1 cams
        att_seq = []
        for i in range(num_cams):
            att_seq.append(i-1)
            att_seq.append((i+1)%num_cams)
        kv = hidden_states[:, att_seq]  # nhw 2 c
        for block in self.transformer_mv_blocks:
            hidden_list = []
            for i_cam in range(num_cams):
                q_ = hidden_states[:, i_cam:i_cam+1].reshape(num_frames, height * width, hidden_states.shape[2])
                kv_ = kv[:, 2*i_cam: 2*i_cam+2].reshape(num_frames, height * width * 2, hidden_states.shape[2])
                if self.training and self.gradient_checkpointing:
                    hidden_state_ = torch.utils.checkpoint.checkpoint(
                        (block),
                        q_, None, kv_, None,
                        timestep, cross_attention_kwargs,
                        class_labels,
                        use_reentrant=True
                    )
                else:
                    hidden_state_ = block(
                        q_,
                        encoder_hidden_states=kv_,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                    )
                hidden_list.append(hidden_state_.permute(1,0,2))
            hidden_states = torch.stack(hidden_list, dim=1).permute(2,0,1,3).reshape(num_frames * height * width, num_cams, -1)
        
        # 3. Temporal Blocks
        hidden_states = hidden_states.reshape(num_frames, height * width * num_cams, -1).permute(1, 0, 2)
        for block in self.transformer_temp_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
        hidden_states = hidden_states.permute(1, 0, 2).reshape(num_frames * height * width, num_cams, -1)
        
        # 4. Output
        hidden_states = self.proj_out(hidden_states)  # nhw cam c
        hidden_states = hidden_states.reshape(num_frames, height, width, num_cams, -1).permute(3, 0, 4, 1, 2).contiguous()
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)

class TransformerSpatioTemporalModel(nn.Module):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        out_channels (`int`, *optional*):
            The number of channels in the output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        attention_type: str = "default",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_type=attention_type
                )
                for d in range(num_layers)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0]

        if time_context.shape[1]==1:
            time_context = time_context_first_timestep[None, :].broadcast_to(
                height * width, batch_size, 1, time_context.shape[-1]
            )
        else:
            time_context = time_context_first_timestep[None, :].permute(2,1,0,3)[0][None].broadcast_to(height*width,batch_size,1,time_context.shape[-1])
        time_context = time_context.reshape(height * width * batch_size, 1, time_context.shape[-1])
        
        # time_context=time_context.reshape(-1,1,time_context.shape[-1])
        
        # time_context = time_context.reshape(time_context.shape[0] * batch_size, 1, time_context.shape[-1])

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)

@maybe_allow_in_graph
class MultiViewBasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        view_mix_inner_dim: Optional[int] = None,
    ):
        super().__init__()
        if view_mix_inner_dim is None:
            view_mix_inner_dim=dim
        self.is_res = dim == view_mix_inner_dim
        self.norm_in = nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(
            dim,
            dim_out=view_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(view_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=view_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(view_mix_inner_dim)
        self.ff = FeedForward(view_mix_inner_dim, activation_fn="geglu")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        num_cams: int,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        # batch_size = hidden_states.shape[0]

        batch_cams_frames, seq_length, channels = hidden_states.shape
        batch_frames = batch_cams_frames // num_cams
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states[None,None, :].reshape(batch_size,num_cams, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2,3, 1, 4)
        hidden_states = hidden_states.reshape(batch_size*num_frames * seq_length, num_cams, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states


        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None,None, :].reshape(batch_size, num_frames, seq_length, num_cams, channels)
        hidden_states = hidden_states.permute(0,3, 1, 2, 4)
        hidden_states = hidden_states.reshape(batch_size*num_cams * num_frames, seq_length, channels)

        return hidden_states

@maybe_allow_in_graph
class MultiViewBasicTransformerBlock_CA(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        view_mix_inner_dim: Optional[int] = None,
    ):
        super().__init__()
        if view_mix_inner_dim is None:
            view_mix_inner_dim=dim
        self.is_res = dim == view_mix_inner_dim
        self.norm_in = nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(
            dim,
            dim_out=view_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(view_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=view_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(view_mix_inner_dim)
        self.ff = FeedForward(view_mix_inner_dim, activation_fn="geglu")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        num_cams: int,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        # batch_size = hidden_states.shape[0]

        batch_cams_frames, seq_length, channels = hidden_states.shape
        batch_frames = batch_cams_frames // num_cams
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states[None,None, :].reshape(batch_size,num_cams, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2,3, 1, 4)
        hidden_states = hidden_states.reshape(batch_size*num_frames * seq_length, num_cams, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        q = norm_hidden_states[:,[0,2,3,5],:]
        kv = norm_hidden_states[:,[1,4],:]
        attn_output = self.attn1(q, encoder_hidden_states=kv)
        attn_output = torch.cat([attn_output,kv],dim=1)
        attn_output = attn_output[:,[0,4,1,2,5,3]]
        hidden_states = attn_output + hidden_states


        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None,None, :].reshape(batch_size, num_frames, seq_length, num_cams, channels)
        hidden_states = hidden_states.permute(0,3, 1, 2, 4)
        hidden_states = hidden_states.reshape(batch_size*num_cams * num_frames, seq_length, channels)

        return hidden_states

@maybe_allow_in_graph
class MultiViewBasicTransformerBlock_Text(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        view_mix_inner_dim: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        if view_mix_inner_dim is None:
            view_mix_inner_dim=dim
        self.is_res = dim == view_mix_inner_dim
        self.norm_in = nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(
            dim,
            dim_out=view_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(view_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=view_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )
        # 2. Cross-Attn
        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = nn.LayerNorm(view_mix_inner_dim)
            self.attn2 = Attention(
                query_dim=view_mix_inner_dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None
        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(view_mix_inner_dim)
        self.ff = FeedForward(view_mix_inner_dim, activation_fn="geglu")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        num_cams: int,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        # batch_size = hidden_states.shape[0]

        batch_cams_frames, seq_length, channels = hidden_states.shape
        batch_frames = batch_cams_frames // num_cams
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states[None,None, :].reshape(batch_size,num_cams, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2,3, 1, 4)
        hidden_states = hidden_states.reshape(batch_size*num_frames * seq_length, num_cams, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states
        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None,None, :].reshape(batch_size, num_frames, seq_length, num_cams, channels)
        hidden_states = hidden_states.permute(0,3, 1, 2, 4)
        hidden_states = hidden_states.reshape(batch_size*num_cams * num_frames, seq_length, channels)

        return hidden_states

@maybe_allow_in_graph
class MultiViewBasicTransformerBlock_Text_ALL(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        view_mix_inner_dim: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        if view_mix_inner_dim is None:
            view_mix_inner_dim=dim
        self.is_res = dim == view_mix_inner_dim
        self.norm_in = nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(
            dim,
            dim_out=view_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(view_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=view_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )
        # 2. Cross-Attn
        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = nn.LayerNorm(view_mix_inner_dim)
            self.attn2 = Attention(
                query_dim=view_mix_inner_dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None
        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(view_mix_inner_dim)
        self.ff = FeedForward(view_mix_inner_dim, activation_fn="geglu")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        num_cams: int,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        # batch_size = hidden_states.shape[0]

        batch_cams_frames, seq_length, channels = hidden_states.shape
        batch_frames = batch_cams_frames // num_cams
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states[None,None, :].reshape(batch_size,num_cams, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2,1,3, 4)
        hidden_states = hidden_states.reshape(batch_size*num_frames, num_cams*seq_length, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states
        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None,None, :].reshape(batch_size, num_frames, num_cams,seq_length, channels)
        hidden_states = hidden_states.permute(0,2, 1, 3, 4)
        hidden_states = hidden_states.reshape(batch_size*num_cams * num_frames, seq_length, channels)

        return hidden_states

@maybe_allow_in_graph
class MultiViewBasicTransformerBlock_Text_Neibor(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        view_mix_inner_dim: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        if view_mix_inner_dim is None:
            view_mix_inner_dim=dim
        self.is_res = dim == view_mix_inner_dim
        self.norm_in = nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(
            dim,
            dim_out=view_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(view_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=view_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )
        # 2. Cross-Attn
        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = nn.LayerNorm(view_mix_inner_dim)
            self.attn2 = Attention(
                query_dim=view_mix_inner_dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None
        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(view_mix_inner_dim)
        self.ff = FeedForward(view_mix_inner_dim, activation_fn="geglu")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        num_cams: int,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        # batch_size = hidden_states.shape[0]

        batch_cams_frames, seq_length, channels = hidden_states.shape
        batch_frames = batch_cams_frames // num_cams
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states[None,None, :].reshape(batch_size,num_cams, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2,1,3, 4)
        hidden_states = hidden_states.reshape(batch_size*num_frames, num_cams*seq_length, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states.reshape(batch_size*num_frames,num_cams,seq_length,channels)
        attn_output=[]
        for i in range(num_cams):
            current_view_hidden_states = norm_hidden_states[:,i]
            neibor_view_hidden_state = norm_hidden_states[:,[(i-1)%num_cams,(i+1)%num_cams]].reshape(batch_size*num_frames,2*seq_length,channels)
            current_view_output = self.attn1(current_view_hidden_states, encoder_hidden_states=neibor_view_hidden_state)
            attn_output.append(current_view_output)
        attn_output=torch.cat(attn_output,dim=1)
        hidden_states = attn_output + hidden_states
        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None,None, :].reshape(batch_size, num_frames, num_cams,seq_length, channels)
        hidden_states = hidden_states.permute(0,2, 1, 3, 4)
        hidden_states = hidden_states.reshape(batch_size*num_cams * num_frames, seq_length, channels)

        return hidden_states

@maybe_allow_in_graph
class MultiViewBasicTransformerBlock_Text_FTL(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        view_mix_inner_dim: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        if view_mix_inner_dim is None:
            view_mix_inner_dim=dim
        self.is_res = dim == view_mix_inner_dim
        

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(
            dim,
            dim_out=view_mix_inner_dim,
            activation_fn="geglu",
        )
        self.norm_ftl_in = nn.LayerNorm(dim)
        self.ff_ftl_in = FeedForward(
            dim,
            dim_out=view_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(view_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=view_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )
        # 2. Cross-Attn
        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = nn.LayerNorm(view_mix_inner_dim)
            self.attn2 = Attention(
                query_dim=view_mix_inner_dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None
        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(view_mix_inner_dim)
        self.ff = FeedForward(view_mix_inner_dim, activation_fn="geglu")
        
        self.norm_ftl_out = nn.LayerNorm(dim)
        self.ff_ftl_out = FeedForward(
            dim,
            dim_out=view_mix_inner_dim,
            activation_fn="geglu",
        )
        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def ftl(self, z, proj_mats):
        z = z.permute(0,2,1)
        b, c,l= z.size()
        N = proj_mats.size(2)

        z = z.reshape(b, N, -1)
        out = torch.bmm(proj_mats, z)

        out = out.reshape(b, c,l).permute(0,2,1).contiguous()
        return out
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        num_cams: int,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        proj_inv: Optional[torch.FloatTensor] = None,
        proj: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        # batch_size = hidden_states.shape[0]

        batch_cams_frames, seq_length, channels = hidden_states.shape
        batch_frames = batch_cams_frames // num_cams
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states[None,None, :].reshape(batch_size,num_cams, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2,1, 3, 4)
        hidden_states = hidden_states.reshape(batch_size*num_frames, num_cams,seq_length, channels)

        canonical_states = []
        for i in range(num_cams):
            sv_states = hidden_states[:,i]
            sv_states = self.ff_ftl_in(self.norm_ftl_in(sv_states))
            sv_states = self.ftl(sv_states,proj_inv[i])
            canonical_states.append(sv_states)
        hidden_states = torch.stack(canonical_states,dim=2)
        hidden_states = hidden_states.reshape(batch_size*num_frames*seq_length,num_cams,channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states
        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
      
        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output
        hidden_states = hidden_states.reshape(batch_size*num_frames,seq_length,num_cams,channels)
        sv_states = []
        for i in range(num_cams):
            can_states = hidden_states[:,:,i]
            can_states = self.ftl(can_states,proj[i])
            can_states = self.ff_ftl_out(self.norm_ftl_out(can_states))
            sv_states.append(can_states)

        hidden_states = torch.stack(sv_states,dim=1)
        hidden_states = hidden_states.reshape(batch_size,num_cams,num_frames,seq_length,channels)
        hidden_states = hidden_states.reshape(batch_size*num_cams * num_frames, seq_length, channels)

        return hidden_states


@maybe_allow_in_graph
class MultiViewBasicTransformerBlock_IMAGE(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        view_mix_inner_dim: Optional[int] = None,
    ):
        super().__init__()
        if view_mix_inner_dim is None:
            view_mix_inner_dim=dim
        self.is_res = dim == view_mix_inner_dim
        self.norm_in = nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(
            dim,
            dim_out=view_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(view_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=view_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(view_mix_inner_dim)
        self.ff = FeedForward(view_mix_inner_dim, activation_fn="geglu")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        num_cams: int,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        # batch_size = hidden_states.shape[0]

        batch_cams_frames, seq_length, channels = hidden_states.shape
        batch_frames = batch_cams_frames // num_cams
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states[None,None, :].reshape(batch_size,num_cams, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2,3, 1, 4)
        hidden_states = hidden_states.reshape(batch_size*num_frames, seq_length*num_cams, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states


        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None,None, :].reshape(batch_size, num_frames, seq_length, num_cams, channels)
        hidden_states = hidden_states.permute(0,3, 1, 2, 4)
        hidden_states = hidden_states.reshape(batch_size*num_cams * num_frames, seq_length, channels)

        return hidden_states

class TransformerSpatioTemporalModelMV(nn.Module):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        out_channels (`int`, *optional*):
            The number of channels in the output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        attention_type: str = "default",
        first_mv: bool=False
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.first_mv = first_mv
        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_type=attention_type
                )
                for d in range(num_layers)
            ]
        )
        self.mv_blocks = nn.ModuleList(
            [
                MultiViewBasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    )
                    for d in range(1)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        cam_only_indicator: Optional[torch.Tensor] = None,
        num_cams = 6,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames
        # num_frames = num_frames // num_cams

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0]

        if time_context.shape[1]==1:
            time_context = time_context_first_timestep[None, :].broadcast_to(
                height * width, batch_size, 1, time_context.shape[-1]
            )
        else:
            time_context = time_context_first_timestep[None, :].permute(2,1,0,3)[0][None].broadcast_to(height*width,batch_size,1,time_context.shape[-1])
        time_context = time_context.reshape(height * width * batch_size, 1, time_context.shape[-1])
        
        # time_context=time_context.reshape(-1,1,time_context.shape[-1])
        
        # time_context = time_context.reshape(time_context.shape[0] * batch_size, 1, time_context.shape[-1])

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, mv_block ,temporal_block in zip(self.transformer_blocks,self.mv_blocks, self.temporal_transformer_blocks):
            if self.first_mv:
                # print(1)
                hidden_states = mv_block(
                    hidden_states, 
                    num_frames=num_frames,
                    num_cams=num_cams,
                )

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            if not self.first_mv:
                assert 1==2
                hidden_states = mv_block(
                    hidden_states, 
                    num_frames=num_frames,
                    num_cams=num_cams,
                )

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)

class TransformerSpatioTemporalModelMV_Text(nn.Module):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        out_channels (`int`, *optional*):
            The number of channels in the output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        attention_type: str = "default",
        first_mv: bool=False
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.first_mv = first_mv
        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_type=attention_type
                )
                for d in range(num_layers)
            ]
        )
        self.mv_blocks = nn.ModuleList(
            [
                MultiViewBasicTransformerBlock_Text(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    )
                    for d in range(1)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        cam_only_indicator: Optional[torch.Tensor] = None,
        num_cams = 6,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_cams_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_cams = batch_cams_frames // num_frames
        batch_frames = batch_cams_frames//num_cams
        batch_size = batch_cams//num_cams
        # num_frames = num_frames // num_cams

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_cams, num_frames, -1, time_context.shape[-1]
        )[:, 0]
        view_context = encoder_hidden_states
        view_context_first_cam = view_context.reshape(batch_size,num_cams,num_frames,-1,view_context.shape[-1]).permute(0,2,1,3,4).reshape(batch_frames,num_cams,-1,view_context.shape[-1])[:,0]

        if time_context.shape[1]==1:
            time_context = time_context_first_timestep[None, :].broadcast_to(
                height * width, batch_cams, 1, time_context.shape[-1]
            )
            view_context = view_context_first_cam[None,:].broadcast_to(height*width,batch_frames,1,view_context.shape[-1])
        else:
            time_context = time_context_first_timestep[None, :].permute(2,1,0,3)[0][None].broadcast_to(height*width,batch_cams,1,time_context.shape[-1])
        time_context = time_context.reshape(height * width * batch_cams, 1, time_context.shape[-1])
        view_context = view_context.reshape(height*width*batch_frames,1,view_context.shape[-1])
        # time_context=time_context.reshape(-1,1,time_context.shape[-1])
        
        # time_context = time_context.reshape(time_context.shape[0] * batch_size, 1, time_context.shape[-1])

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_cams_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_cams, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, mv_block ,temporal_block in zip(self.transformer_blocks,self.mv_blocks, self.temporal_transformer_blocks):
            if self.first_mv:
                # print(1)
                hidden_states = mv_block(
                    hidden_states, 
                    num_frames=num_frames,
                    num_cams=num_cams,
                    encoder_hidden_states=view_context
                )

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            if not self.first_mv:
                assert 1==2
                hidden_states = mv_block(
                    hidden_states, 
                    num_frames=num_frames,
                    num_cams=num_cams,
                    encoder_hidden_states=view_context
                )

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_cams_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)

class TransformerSpatioTemporalModelMV_Text_ALL(nn.Module):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        out_channels (`int`, *optional*):
            The number of channels in the output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        attention_type: str = "default",
        first_mv: bool=False
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.first_mv = first_mv
        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_type=attention_type
                )
                for d in range(num_layers)
            ]
        )
        self.mv_blocks = nn.ModuleList(
            [
                MultiViewBasicTransformerBlock_Text_ALL(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    )
                    for d in range(1)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        cam_only_indicator: Optional[torch.Tensor] = None,
        num_cams = 6,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_cams_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_cams = batch_cams_frames // num_frames
        batch_frames = batch_cams_frames//num_cams
        batch_size = batch_cams//num_cams
        # num_frames = num_frames // num_cams

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_cams, num_frames, -1, time_context.shape[-1]
        )[:, 0]
        view_context = encoder_hidden_states
        view_context_first_cam = view_context.reshape(batch_size,num_cams,num_frames,-1,view_context.shape[-1]).permute(0,2,1,3,4).reshape(batch_frames,num_cams,-1,view_context.shape[-1])[:,0]

        if time_context.shape[1]==1:
            time_context = time_context_first_timestep[None, :].broadcast_to(
                height * width, batch_cams, 1, time_context.shape[-1]
            )
            # view_context = view_context_first_cam[None,:].broadcast_to(height*width,batch_frames,1,view_context.shape[-1])
        else:
            time_context = time_context_first_timestep[None, :].permute(2,1,0,3)[0][None].broadcast_to(height*width,batch_cams,1,time_context.shape[-1])
        time_context = time_context.reshape(height * width * batch_cams, 1, time_context.shape[-1])
        view_context = view_context_first_cam.reshape(batch_frames,1,view_context.shape[-1])
        # time_context=time_context.reshape(-1,1,time_context.shape[-1])
        
        # time_context = time_context.reshape(time_context.shape[0] * batch_size, 1, time_context.shape[-1])

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_cams_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_cams, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, mv_block ,temporal_block in zip(self.transformer_blocks,self.mv_blocks, self.temporal_transformer_blocks):
            if self.first_mv:
                # print(1)
                hidden_states = mv_block(
                    hidden_states, 
                    num_frames=num_frames,
                    num_cams=num_cams,
                    encoder_hidden_states=view_context
                )

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            if not self.first_mv:
                assert 1==2
                hidden_states = mv_block(
                    hidden_states, 
                    num_frames=num_frames,
                    num_cams=num_cams,
                    encoder_hidden_states=view_context
                )

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_cams_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)

class TransformerSpatioTemporalModelMV_Text_FTL(nn.Module):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        out_channels (`int`, *optional*):
            The number of channels in the output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        attention_type: str = "default",
        first_mv: bool=False
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.first_mv = first_mv
        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_type=attention_type
                )
                for d in range(num_layers)
            ]
        )
        self.mv_blocks = nn.ModuleList(
            [
                MultiViewBasicTransformerBlock_Text_FTL(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    )
                    for d in range(1)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        cam_only_indicator: Optional[torch.Tensor] = None,
        num_cams = 6,
        proj=None,
        proj_inv=None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_cams_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_cams = batch_cams_frames // num_frames
        batch_frames = batch_cams_frames//num_cams
        batch_size = batch_cams//num_cams
        # num_frames = num_frames // num_cams

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_cams, num_frames, -1, time_context.shape[-1]
        )[:, 0]
        view_context = encoder_hidden_states
        view_context_first_cam = view_context.reshape(batch_size,num_cams,num_frames,-1,view_context.shape[-1]).permute(0,2,1,3,4).reshape(batch_frames,num_cams,-1,view_context.shape[-1])[:,0]

        if time_context.shape[1]==1:
            time_context = time_context_first_timestep[None, :].broadcast_to(
                height * width, batch_cams, 1, time_context.shape[-1]
            )
            view_context = view_context_first_cam[None,:].broadcast_to(height*width,batch_frames,1,view_context.shape[-1])
        else:
            time_context = time_context_first_timestep[None, :].permute(2,1,0,3)[0][None].broadcast_to(height*width,batch_cams,1,time_context.shape[-1])
        time_context = time_context.reshape(height * width * batch_cams, 1, time_context.shape[-1])
        view_context = view_context.reshape(height*width*batch_frames,1,view_context.shape[-1])
        # time_context=time_context.reshape(-1,1,time_context.shape[-1])
        
        # time_context = time_context.reshape(time_context.shape[0] * batch_size, 1, time_context.shape[-1])

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_cams_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_cams, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, mv_block ,temporal_block in zip(self.transformer_blocks,self.mv_blocks, self.temporal_transformer_blocks):
            if self.first_mv:
                # print(1)
                hidden_states = mv_block(
                    hidden_states, 
                    num_frames=num_frames,
                    num_cams=num_cams,
                    encoder_hidden_states=view_context,
                    proj=proj,
                    proj_inv=proj_inv
                )

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            if not self.first_mv:
                assert 1==2
                hidden_states = mv_block(
                    hidden_states, 
                    num_frames=num_frames,
                    num_cams=num_cams,
                    encoder_hidden_states=view_context
                )

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_cams_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)



class TransformerSpatioTemporalModelMV_EMB(nn.Module):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        out_channels (`int`, *optional*):
            The number of channels in the output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        attention_type: str = "default",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_type=attention_type
                )
                for _ in range(num_layers)
            ]
        )
        view_mix_inner_dim = inner_dim
        self.mv_blocks = nn.ModuleList(
            [
                MultiViewBasicTransformerBlock_IMAGE(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    view_mix_inner_dim=view_mix_inner_dim
                    )
                    for _ in range(num_layers)
            ]
        )
        view_embed_dim = in_channels * 4
        self.view_pos_embed = TimestepEmbedding(in_channels, view_embed_dim, out_dim=in_channels)
        self.view_proj = Timesteps(in_channels, True, 0)
        self.view_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cam_only_indicator: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        num_cams: int = 6,
        return_dict: bool = True,

    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_cams_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        num_cams = cam_only_indicator.shape[-1]
        batch_cams = batch_cams_frames // num_frames
        batch_frames = batch_cams_frames // num_cams
        # num_frames = num_frames // num_cams

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_cams, num_frames, -1, time_context.shape[-1]
        )[:, 0]

        if time_context.shape[1]==1:
            time_context = time_context_first_timestep[None, :].broadcast_to(
                height * width, batch_cams, 1, time_context.shape[-1]
            )
        else:
            time_context = time_context_first_timestep[None, :].permute(2,1,0,3)[0][None].broadcast_to(height*width,batch_cams,1,time_context.shape[-1])
        time_context = time_context.reshape(height * width * batch_cams, 1, time_context.shape[-1])
        
        # time_context=time_context.reshape(-1,1,time_context.shape[-1])
        
        # time_context = time_context.reshape(time_context.shape[0] * batch_size, 1, time_context.shape[-1])

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_cams_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_cams, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        num_cams_emb = torch.arange(num_cams, device=hidden_states.device)
        num_cams_emb = num_cams_emb.repeat(batch_frames, 1)
        num_cams_emb = num_cams_emb.reshape(-1)
        v_emb = self.view_proj(num_cams_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        v_emb = v_emb.to(dtype=hidden_states.dtype)

        emb_t = self.time_pos_embed(t_emb)
        emb_v = self.view_pos_embed(v_emb)
        emb_t = emb_t[:, None, :]
        emb_v = emb_v[:, None, :]

        # 2. Blocks
        for block, mv_block ,temporal_block in zip(self.transformer_blocks,self.mv_blocks, self.temporal_transformer_blocks):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            hidden_states_mix_v = hidden_states
            hidden_states_mix_v = hidden_states_mix_v + emb_v

            hidden_states_mix_v = mv_block(
                hidden_states_mix_v, 
                num_frames=num_frames,
                num_cams=num_cams,
            )
            hidden_states = self.view_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix_v,
                image_only_indicator=cam_only_indicator,
            )
            
            hidden_states_mix_t = hidden_states
            hidden_states_mix_t = hidden_states_mix_t + emb_t

            hidden_states_mix_t = temporal_block(
                hidden_states_mix_t,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix_t,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_cams_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)
