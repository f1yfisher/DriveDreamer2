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

import torch
from torch import nn
from typing import Any, Dict, Optional, Tuple, Union

from diffusers.utils import is_torch_version
from diffusers.models.resnet import Downsample2D, Upsample2D,SpatioTemporalResBlock
from .transformer_temporal import TransformerSpatioTemporalModel


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    num_attention_heads,
    transformer_layers_per_block=1,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=True,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    attention_type="default",
):

    if down_block_type=="DownBlockSpatioTemporal":
        return DownBlockSpatioTemporal(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            temb_channels=temb_channels,
            add_downsample=add_downsample
        )
    elif down_block_type == "CrossAttnDownBlockSpatioTemporal":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlockSpatioTemporal")
        return CrossAttnDownBlockSpatioTemporal(in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        num_layers=num_layers,
        transformer_layers_per_block=transformer_layers_per_block,
        num_attention_heads=num_attention_heads,
        cross_attention_dim=cross_attention_dim,
        add_downsample=add_downsample,
        attention_type=attention_type,
        )

    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    num_attention_heads,
    transformer_layers_per_block=1,
    resolution_idx=None,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=True,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    attention_type="default",
):

    if up_block_type=="UpBlockSpatioTemporal":
        if resolution_idx is None:
            raise ValueError("resolution_idx must be specified for UpBlockSpatioTemporal")
        return UpBlockSpatioTemporal(
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            add_upsample=add_upsample
        )
    elif up_block_type == "CrossAttnUpBlockSpatioTemporal":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlockSpatioTemporal")
        if resolution_idx is None:
            raise ValueError("resolution_idx must be specified for CrossAttnUpBlockSpatioTemporal")
        return CrossAttnUpBlockSpatioTemporal(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            resnet_eps=resnet_eps,
            num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            add_upsample=add_upsample,
            attention_type=attention_type,
        )
    
    raise ValueError(f"{up_block_type} does not exist.")

class UNetMidBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        attention_type: str = "default",
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # there is always at least one resnet
        resnets = [
            SpatioTemporalResBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=1e-5,
            )
        ]
        attentions = []

        for i in range(num_layers):
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    attention_type=attention_type
                )
            )

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cam_only_indicator: Optional[torch.Tensor] = None,
        cross_attention_kwargs=None,
    ) -> torch.FloatTensor:
        
        hidden_states = self.resnets[0](
            hidden_states,
            temb,
            image_only_indicator=image_only_indicator,
        )

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
            else:
                
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
        

        return hidden_states

class CrossAttnDownBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        add_downsample: bool = True,
        attention_type: str = "default",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-6,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    attention_type=attention_type
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=1,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cam_only_indicator: Optional[torch.Tensor] = None,
        cross_attention_kwargs=None,

    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in blocks:
            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

class DownBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        add_downsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        image_only_indicator,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        image_only_indicator,
                    )
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

class UpBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        image_only_indicator,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        image_only_indicator,
                    )
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class CrossAttnUpBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        add_upsample: bool = True,
        attention_type: str = "default",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    attention_type=attention_type
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        cam_only_indicator: Optional[torch.Tensor] = None,
        cross_attention_kwargs=None,
    ) -> torch.FloatTensor:
        
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            
            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
