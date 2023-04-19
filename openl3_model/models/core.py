import torch.nn as nn
import torch.nn.functional as F

from models import audio_encoders, text_encoders
from pathlib import Path

import torch

from models.audio_encoders import OpenL3Embeddings
from utils.file_io import load_yaml_file
class DualEncoderModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super(DualEncoderModel, self).__init__()

        settings = load_yaml_file(Path('bart.yaml'))
        if torch.cuda.is_available() and not settings['training']['force_cpu']:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.out_norm = kwargs.get("out_norm", None)
        self.audio_enc = getattr(audio_encoders, args[0], None)(**kwargs["audio_enc"])
        self.text_enc = getattr(text_encoders, args[1], None)(**kwargs["text_enc"])

        # Load pretrained weights for audio encoder

    def audio_branch(self, audio):
        audio_embeds = self.audio_enc(audio)

        if self.out_norm == "L2":
            audio_embeds = F.normalize(audio_embeds, p=2.0, dim=-1)

        return audio_embeds

    def text_branch(self, text):
        text_embeds = self.text_enc(text)

        if self.out_norm == "L2":
            text_embeds = F.normalize(text_embeds, p=2.0, dim=-1)

        return text_embeds

    def forward(self, audio, text):
        """
        :param audio: tensor, (batch_size, time_steps, Mel_bands).
        :param text: tensor, (batch_size, len_padded_text).
        """
        audio_embeds = self.audio_branch(audio)
        text_embeds = self.text_branch(text)

        # audio_embeds: [N, E]    text_embeds: [N, E]
        return audio_embeds, text_embeds
