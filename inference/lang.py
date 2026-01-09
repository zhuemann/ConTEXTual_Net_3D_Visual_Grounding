import torch
import torch.nn as nn


# import numpy as np

# import os
# import cv2

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
