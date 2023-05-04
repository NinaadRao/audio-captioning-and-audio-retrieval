import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartAttention
import torch

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
        #self.bn0 = nn.BatchNorm2d(512)
        
        self.fc2 = nn.Linear(512, 300, bias=True)

        # self.bn0.apply(init_weights)
        # self.cnn.apply(init_weights)
        # self.fc.apply(init_weights)
        # self.fc2.apply(init_weights)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :return: tensor, (batch_size, embed_dim).
        """
        #print('Before:',x.size())
        #x = x.squeeze(1)
        #print('After:',x.size())
        #
        #x = x.transpose(1, 3)
        #x = self.bn0(x)
        #x = x.transpose(1, 3)

        #x = self.cnn(x)
        #print("size of x",x.size())
        x2 = torch.mean(x, dim=1)  # (N, 2048, T/64) 32,287,512
        #print('after mean',x.size())
        #(x1, _) = torch.max(x, dim=1)  # max across time 32,1
        #print('after max',x1.size())
        #x2 = torch.mean(x, dim=2)  # average over time 32,1
        x =  x2  # (N, 2048)
        #print('size of x',x.size())
        #print(x.size())
        x = self.fc(x)  # (N, 2048)
        #print(x.size())
        x = self.fc2(x)  # (N, embed_dim)
        #print(x.size())
        return x
