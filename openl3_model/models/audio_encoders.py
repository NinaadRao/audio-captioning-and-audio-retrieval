import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartAttention


from models import init_weights


class OpenL3Embeddings(nn.Module):

    def __init__(self,  **kwargs):
        super(OpenL3Embeddings, self).__init__()


        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )



        self.fc2 = nn.Linear(512, kwargs["out_dim"], bias=True)

        # self.bn0.apply(init_weights)
        # self.cnn.apply(init_weights)
        # self.fc.apply(init_weights)
        # self.fc2.apply(init_weights)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :return: tensor, (batch_size, embed_dim).
        """
        #print(x.size())
        x = x.squeeze(1)
        #print(x.size())
        #
        # x = x.transpose(1, 3)
        # x = self.bn0(x)
        # x = x.transpose(1, 3)

        # x = self.cnn(x)
        #x = torch.mean(x, dim=3)  # (N, 2048, T/64)

        #(x1, _) = torch.max(x, dim=2)  # max across time
        #x2 = torch.mean(x, dim=2)  # average over time
        #x = x1 + x2  # (N, 2048)

        x = self.fc(x)  # (N, 2048)
        #print(x.size())
        x = self.fc2(x)  # (N, embed_dim)
        #print(x.size())
        return x
