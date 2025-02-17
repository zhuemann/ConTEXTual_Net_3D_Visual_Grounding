# Modified by Yujin Oh, Mar-18-2024

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock
from monai.utils import ensure_tuple_rep
from transformers import LlamaTokenizer, LlamaModel

from .modules import ContextUnetrUpBlock, UnetOutUpBlock
from .sam import TwoWayTransformer
from .text_encoder import tokenize, TextContextEncoder
from .llama2.llama_custom import LlamaForCausalLM


class LangCrossAtt(nn.Module):
    "add documentaiton"

    def __init__(self, emb_dim):
        super(LangCrossAtt, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=1)  # vdim=vdimension
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, lang_rep, vision_rep):
        # print("inside cross attention!")

        # gets all dimensions to be used in the attention
        input_batch = vision_rep.size()[0]
        input_channel = vision_rep.size()[1]
        input_width = vision_rep.size()[2]
        input_height = vision_rep.size()[3]
        input_depth = vision_rep.size()[4]
        # print(f"input_width: {input_width}")
        # print(f"input_height: {input_height}")

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"vision rep siz: {vision_rep.size()}")
        # print(f"language rep: {lang_rep.size()}")
        # ision_rep = torch.zeros(vision_rep.size()).to(device, dtype=torch.float)

        # print(f"vision rep before: {vision_rep.size()}")
        # puts the vision representation into the right shape for attention mechanism
        vision_rep = torch.swapaxes(vision_rep, 0, 1)
        vision_rep_flat = torch.flatten(vision_rep, start_dim=2)
        vision_rep = torch.swapaxes(vision_rep_flat, 2, 0)

        # print(f"vision rep after: {vision_rep.size()}")

        # lang_rep = torch.unsqueeze(lang_rep, 1)
        # puts the language rep into the right shape for attention
        lang_rep = torch.swapaxes(lang_rep, 0, 1)
        # lang_rep = torch.swapaxes(lang_rep, 1, 2)

        # print(f"vision_rep dimensions: {vision_rep.size()}")
        # print(f"language_rep dimensions: {lang_rep.size()}")

        # does cross attention between vision and language
        att_matrix, attn_output_weights = self.multihead_attn(query=vision_rep, key=lang_rep, value=lang_rep)

        # att_matrix = self.sigmoid(att_matrix)
        att_matrix = self.tanh(att_matrix)
        # att_matrix = self.relu(att_matrix)

        vision_rep = vision_rep * att_matrix
        vision_rep = vision_rep.contiguous()

        # rearanges the output matrix to be the dimensions of the input
        out = vision_rep.view(input_width, input_height, input_depth, input_batch, input_channel)
        # print(f"out after the matrix reconstruction: {out.size()}")
        out = torch.swapaxes(out, 2, 4)
        out = torch.swapaxes(out, 1, 3)
        out = torch.swapaxes(out, 0, 2)
        out = torch.swapaxes(out, 0, 1)
        # out = torch.swapaxes(out, 0, 2)
        # out = torch.swapaxes(out, 1, 3)
        # out = torch.swapaxes(out, 2, 4)
        # out = torch.swapaxes(out, 0, 1)

        # print(f"final size: {out.size()}")
        return out


# LLMSeg
class ContextUNETR(nn.Module):
    def __init__(
            self,
            img_size: Sequence[int] | int,
            in_channels: int,
            out_channels: int,
            depths: Sequence[int] = (2, 2, 2, 2),
            feature_size: int = 24,
            norm_name: tuple | str = "instance",
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            normalize: bool = True,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            downsample="merging",
            context=False,
            args=None,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        Examples::
            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)
        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        self.context = context
        self.normalize = normalize

        self.encoder1 = UnetrBasicBlock(spatial_dims=spatial_dims,
                                        in_channels=in_channels,
                                        out_channels=feature_size,
                                        kernel_size=3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder2 = UnetrBasicBlock(spatial_dims=spatial_dims,
                                        in_channels=feature_size,
                                        out_channels=feature_size,
                                        kernel_size=3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder3 = UnetrBasicBlock(spatial_dims=spatial_dims,
                                        in_channels=feature_size,
                                        out_channels=2 * feature_size,
                                        kernel_size=3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder4 = UnetrBasicBlock(spatial_dims=spatial_dims,
                                        in_channels=2 * feature_size,
                                        out_channels=4 * feature_size,
                                        kernel_size=3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder10 = UnetrBasicBlock(spatial_dims=spatial_dims,
                                         in_channels=4 * feature_size,
                                         out_channels=8 * feature_size,
                                         kernel_size=3, stride=2, norm_name=norm_name, res_block=True)

        # decoder
        self.decoder4 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
                                            in_channels=feature_size * 8,
                                            out_channels=feature_size * 4,
                                            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True,
                                            add_channels=(
                                                args.n_prompts if args.align_score else 0 if self.context else 0))

        self.decoder3 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
                                            in_channels=feature_size * 4 - (
                                                args.n_prompts if args.align_score else 0 if self.context else 0),
                                            out_channels=feature_size * 2,
                                            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name,
                                            add_channels=(
                                                args.n_prompts if args.align_score else 0 if self.context else 0))

        self.decoder2 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
                                            in_channels=feature_size * 2 - (
                                                args.n_prompts if args.align_score else 0 if self.context else 0),
                                            out_channels=feature_size,
                                            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name,
                                            add_channels=(
                                                args.n_prompts if args.align_score else 0 if self.context else 0))

        self.decoder1 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
                                            in_channels=feature_size - (
                                                args.n_prompts if args.align_score else 0 if self.context else 0),
                                            out_channels=feature_size,
                                            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name,
                                            add_channels=(
                                                args.n_prompts if args.align_score else 0 if self.context else 0))

        # out
        self.out = UnetOutUpBlock(spatial_dims=spatial_dims,
                                  in_channels=feature_size,
                                  out_channels=out_channels,
                                  kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)

        lang_dimension = 4096
        self.lang_att1 = LangCrossAtt(emb_dim=384)
        self.lang_proj1 = nn.Linear(lang_dimension, 384)

        self.lang_att2 = LangCrossAtt(emb_dim=192)
        self.lang_proj2 = nn.Linear(lang_dimension, 192)

        self.lang_att3 = LangCrossAtt(emb_dim=96)
        self.lang_proj3 = nn.Linear(lang_dimension, 96)

        self.lang_att4 = LangCrossAtt(emb_dim=48)
        self.lang_proj4 = nn.Linear(lang_dimension, 48)

        self.lang_att5 = LangCrossAtt(emb_dim=48)
        self.lang_proj5 = nn.Linear(lang_dimension, 48)

        feature_size_list = [self.encoder1.layer.conv3.out_channels, self.encoder2.layer.conv3.out_channels,
                             self.encoder3.layer.conv3.out_channels, self.encoder4.layer.conv3.out_channels,
                             self.encoder10.layer.conv3.out_channels]

        # multiomdal text encoder
        if self.context:

            self.align_score = args.align_score

            if args.textencoder == 'llama2':
                self.txt_embed_dim = 4096
            elif args.textencoder == 'llama2_13b':
                self.txt_embed_dim = 5120
            else:
                self.txt_embed_dim = 512

            # interactive align module
            txt2vis, attntrans = [], []
            for i in range(len(depths) + 1):
                txt2vis.append(nn.Linear(self.txt_embed_dim, feature_size_list[i]))
                attntrans.append(TwoWayTransformer(depth=2,
                                                   embedding_dim=feature_size_list[i],
                                                   mlp_dim=feature_size * (2 ** i),
                                                   num_heads=8,
                                                   ))
            self.txt2vis = nn.Sequential(*txt2vis)
            self.attn_transformer = nn.Sequential(*attntrans)

            # clip text encoder
            self.text_encoder = TextContextEncoder(embed_dim=self.txt_embed_dim)
            self.context_length = args.context_length
            self.token_embed_dim = self.text_encoder.text_projection.shape[-1]
            self.contexts = nn.Parameter(torch.randn(args.n_prompts, self.context_length, self.token_embed_dim))
            self.max_length = 77
            for name, param in self.text_encoder.named_parameters():
                param.requires_grad_(False)

            # llama2
            if args.textencoder.find('llama') >= 0:
                self.text_encoder.llm = True
                if args.textencoder == 'llama2':
                    rep_llama = args.llama_rep
                # print("add in later one downloaded")
                # """
                # rep_llama = "/UserData/Zach_Analysis/git_multimodal/ro_llm/MM-LLM-RO/model/llama2/Llama-2-7b-hf/"
                print(f"path for llama: {rep_llama}")
                self.tokenizer = LlamaTokenizer.from_pretrained(rep_llama)
                """
                self.max_length = 200

                self.text_encoder.transformer  = LlamaForCausalLM.from_pretrained(
                        rep_llama,
                        # load_in_8bit=True, # Add this for using int8
                        torch_dtype=torch.float16,
                        device_map="cpu", #args.gpu "cpu"
                    ).model
                """
                self.llama_model = LlamaModel.from_pretrained(rep_llama)
                self.tokenizer._add_tokens(["<SEG>"], special_tokens=True)
                # self.text_encoder.transformer.resize_token_embeddings(len(self.tokenizer) + 1)
                # self.text_encoder.token_embedding = self.text_encoder.transformer.embed_tokens

                # for name, param in self.text_encoder.transformer.named_parameters():
                #    param.requires_grad_(False)
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad_(False)
                # """
                self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token as padding

    def load_from(self, weights):
        pass

    def interactive_alignment(self, hidden_states_out, report_in, x_in):

        tok_txt = []
        emb_txt = []
        emb_txt_t = []

        # prepare text tokens
        tok_txt = report_in
        # print(f"right before fail: {tok_txt}")
        # print(f"type: {type(tok_txt)}")
        # print(f"keys: {tok_txt.keys()}")

        emb_txt = self.text_encoder(tok_txt.to(x_in.device), self.contexts)
        print(f"right after embedding is obtained: {emb_txt.size()}")
        # projection
        report_l = []
        for i in self.txt2vis._modules.keys():
            print(f"i in projection: {i}")
            report_l.append(self.txt2vis._modules[i](emb_txt))
        print(f"report_l size: {len(report_l)}")
        print(f"report_l element 0 size: {report_l[0].size()}")
        print(f"report_l element 1 size: {report_l[1].size()}")
        print(f"report_l element 2 size: {report_l[2].size()}")
        print(f"report_l element 3 size: {report_l[3].size()}")

        # interactive alignment
        h_offset = 0
        for j, text_vis in enumerate(zip(report_l[h_offset:], hidden_states_out[h_offset:])):

            txt, vis = text_vis

            if len(report_in) != len(x_in):
                print(f"report in length: {report_in.size()}")
                print(f"x in size: {x_in.size()}")
                txt = torch.repeat_interleave(txt, vis.shape[0], dim=0)
                print(f"txt size inside attention: {txt.size()}")
            print(f"txt size inside attention: {txt.size()}")
            print(f"vis size inside attention: {txt.size()}")

            _, hidden_states_out[j + h_offset] = self.attn_transformer[j + h_offset](vis, None, txt)

        print(f"hiden_states_out 0 size: {hidden_states_out[0].size()}")
        print(f"hiden_states_out 1 size: {hidden_states_out[1].size()}")
        print(f"hiden_states_out 2 size: {hidden_states_out[2].size()}")
        print(f"hiden_states_out 3 size: {hidden_states_out[3].size()}")

        # print(f"emb_txt size: {emb_txt.size()}")
        # print(f"embed_txt_t: {emb_txt_t.size()}")
        return hidden_states_out, emb_txt, emb_txt_t

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in, report_in=None, target=None):

        # print(f"report in:{report_in}")

        # tokenized_input = self.tokenizer(report_in, padding=True, truncation=True, return_tensors="pt")
        # tokenized_input = {k: v.to(x_in.device) for k, v in report_in.items()}

        # outputs = self.llama_model(**report_in)
        # token_embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
        # print(f"first forward toekn sizen: {token_embeddings}")
        # outputs = self.llama_model(report_in.to(x_in.device))
        # print(f"outputs size: {outputs.size()}")
        # token_embeddings = outputs.last_hidden_state.squeeze()
        # print(f"my own token_embeddings: {token_embeddings}")
        # token_embeddings = self.llama_model.last_hidden_state.squeeze()  # Shape: (sequence_length, hidden_size)
        # print(f"test llama tokens: {token_embeddings.size()}")
        # print(f"lenght of input to text encoder: {report_in.size()}")
        # emb_txt = self.text_encoder(report_in.to(x_in.device), self.contexts)
        tokenized_input = {
            "input_ids": report_in.to(x_in.device),  # Move input_ids to the same device as x_in
            "attention_mask": torch.ones_like(report_in, dtype=torch.long, device=x_in.device),  # All ones
        }
        # print(f"Input IDs shape: {tokenized_input['input_ids'].shape}")
        # print(f"Attention mask shape: {tokenized_input['attention_mask'].shape}")

        # Ensure input_ids and attention_mask are padded to max_length = 512
        max_length = 200
        pad_token_id = 0  # Typically 0 for padding

        # Get the current length of input_ids
        current_length = tokenized_input["input_ids"].size(1)

        if current_length < max_length:
            # Pad input_ids
            padding = torch.full(
                (tokenized_input["input_ids"].size(0), max_length - current_length),  # Padding size
                pad_token_id,  # Padding value
                dtype=tokenized_input["input_ids"].dtype,
                device=tokenized_input["input_ids"].device,
            )
            tokenized_input["input_ids"] = torch.cat((tokenized_input["input_ids"], padding), dim=1)

            # Pad attention_mask
            padding_mask = torch.zeros(
                (tokenized_input["attention_mask"].size(0), max_length - current_length),
                dtype=tokenized_input["attention_mask"].dtype,
                device=tokenized_input["attention_mask"].device,
            )
            tokenized_input["attention_mask"] = torch.cat((tokenized_input["attention_mask"], padding_mask), dim=1)
        elif current_length > max_length:
            # Truncate sequences that are too long
            tokenized_input["input_ids"] = tokenized_input["input_ids"][:, :max_length]
            tokenized_input["attention_mask"] = tokenized_input["attention_mask"][:, :max_length]

        # Pass to LlamaModel
        outputs = self.llama_model(**tokenized_input)

        # print(f"input tokens: {tokenized_input['input_ids'].size()}")
        # print(f"attention mask: {tokenized_input['attention_mask'].size()}")

        # Pass to LlamaModel
        # outputs = self.llama_model(**tokenized_input)
        # outputs = self.llama_model(input_ids=tokenized_input["input_ids"], attention_mask=tokenized_input["attention_mask"])
        token_embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]

        # print(f"embed text size: {token_embeddings.size()}")

        hidden_states_out = []

        # visual encoder
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(enc0)
        enc2 = self.encoder3(enc1)
        enc3 = self.encoder4(enc2)
        dec4 = self.encoder10(enc3)

        # print(f"x in size: {x_in.size()}")
        # print(f"enc0 size: {enc0.size()}")
        # print(f"enc1 size: {enc1.size()}")
        # print(f"enc2 size: {enc2.size()}")
        # print(f"enc3 size: {enc3.size()}")
        # print(f"dec4 size: {dec4.size()}")

        hidden_states_out.append(enc0)
        hidden_states_out.append(enc1)
        hidden_states_out.append(enc2)
        hidden_states_out.append(enc3)
        hidden_states_out.append(dec4)

        # multimodal alignment
        # hidden_states_out, _, _ = self.interactive_alignment(hidden_states_out, report_in, x_in)
        lang_proj1 = self.lang_proj1(token_embeddings)
        # print(f"lang project size: {lang_proj1.size()}")
        # print(f"vision rep: {dec4.size()}")
        hidden_states_out4 = self.lang_att1(lang_rep=lang_proj1, vision_rep=dec4)
        # print(f"fhidden_state out4: {hidden_states_out4.size()}")

        lang_proj2 = self.lang_proj2(token_embeddings)
        # print(f"lang project size: {lang_proj2.size()}")
        # print(f"vision rep: {enc3.size()}")
        hidden_states_out3 = self.lang_att2(lang_rep=lang_proj2, vision_rep=enc3)
        # print(f"fhidden_state out3: {hidden_states_out3.size()}")

        lang_proj3 = self.lang_proj3(token_embeddings)
        # print(f"lang project size: {lang_proj3.size()}")
        hidden_states_out2 = self.lang_att3(lang_rep=lang_proj3, vision_rep=enc2)
        # print(f"fhidden_state out 2: {hidden_states_out2.size()}")

        lang_proj4 = self.lang_proj4(token_embeddings)
        # print(f"lang project size: {lang_proj4.size()}")
        hidden_states_out1 = self.lang_att4(lang_rep=lang_proj4, vision_rep=enc1)
        # print(f"fhidden_state out 1: {hidden_states_out1.size()}")

        lang_proj5 = self.lang_proj5(token_embeddings)
        # print(f"lang project size: {lang_proj5.size()}")
        hidden_states_out0 = self.lang_att5(lang_rep=lang_proj5, vision_rep=enc0)
        # print(f"fhidden_state out 0: {hidden_states_out0.size()}")
        # out = self.lang_att1(lang_rep=emb_txt, vision_rep=dec4)
        # print(f"cross attention out size:{out.size()}")
        # visual decoder
        dec2 = self.decoder4(hidden_states_out4, hidden_states_out3)
        dec1 = self.decoder3(dec2, hidden_states_out2)
        dec0 = self.decoder2(dec1, hidden_states_out1)
        out = self.decoder1(dec0, hidden_states_out0)

        # dec2 = self.decoder4(hidden_states_out[4], hidden_states_out[3])
        # dec1 = self.decoder3(dec2, hidden_states_out[2])
        # dec0 = self.decoder2(dec1, hidden_states_out[1])
        # out = self.decoder1(dec0, hidden_states_out[0])

        logits = self.out(out)

        return logits



