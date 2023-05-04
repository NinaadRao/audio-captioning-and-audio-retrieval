import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartAttention


from models import init_weights
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm


class BARTAAC(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        
        
        # Audio encoder from t6b
        self.audio_enc = OpenL3Embeddings(out_dim=2048) # Fixed
        
        
        
        
        
        # Main model configuration
        bart_config = BartConfig(vocab_size=kwargs['vocab_size'],
                                encoder_layers=kwargs['encoder_layers'],
                                encoder_ffn_dim=kwargs['encoder_ffn_dim'],
                                encoder_attention_heads=kwargs['encoder_attention_heads'],
                                decoder_layers=kwargs['decoder_layers'],
                                decoder_ffn_dim=kwargs['decoder_ffn_dim'],
                                decoder_attention_heads=kwargs['decoder_attention_heads'],
                                activation_function=kwargs['activation_function'],
                                d_model=kwargs['d_model'],
                                dropout=kwargs['dropout'],
                                attention_dropout=kwargs['attention_dropout'],
                                activation_dropout=kwargs['activation_dropout'],
                                classifier_dropout=kwargs['classifier_dropout'],
                                max_length=kwargs['max_length'],
                                min_length=kwargs['min_length'],
                                early_stopping=kwargs['early_stopping'],
                                num_beams=kwargs['num_beams'],
                                length_penalty=kwargs['length_penalty'],
                                no_repeat_ngram_size=kwargs['no_repeat_ngram_size'])
        print(bart_config)
        
        # Other parameters
        audio_emb_size = kwargs['audio_emb_size']
        lm_emb_size = bart_config.d_model
        pretrained_lm = kwargs['pretrained']
        n_adapt_layers = kwargs['nb_layers']
        
        # Audio features to d_model embeddings
        if n_adapt_layers >= 1:
            audio_adapt_list = [nn.Linear(audio_emb_size, lm_emb_size)]
            for i_adapt in range(n_adapt_layers-1):
                audio_adapt_list.append(nn.ReLU(inplace=True))
                audio_adapt_list.append(nn.Linear(lm_emb_size, lm_emb_size))
            self.audio_adapt = nn.Sequential(*audio_adapt_list)
        else:
            self.audio_adapt = None
        
        if pretrained_lm is not None: # Bypass model configuration to load a pre-trained model (e.g. facebook/bart-base)
            self.bart_lm = BartForConditionalGeneration.from_pretrained(pretrained_lm)
        else:
            self.bart_lm = BartForConditionalGeneration(bart_config)
        
        # Freezing
        if kwargs['freeze_all']:
            for p in self.bart_lm.parameters():
                p.requires_grad = False
            for p in self.bart_lm.model.encoder.embed_positions.parameters():
                p.requires_grad = True
            for p in self.bart_lm.model.encoder.layers[0].self_attn.parameters():
                p.requires_grad = True
        if kwargs['freeze_dec']:
            for p in self.bart_lm.model.shared.parameters():
                p.requires_grad = False
            for p in self.bart_lm.model.decoder.parameters():
                p.requires_grad = False
            for p in self.bart_lm.lm_head.parameters():
                p.requires_grad = False
        if kwargs['freeze_enc']:
            for p in self.bart_lm.model.encoder.parameters():
                p.requires_grad = False
        if kwargs['freeze_attn']:
            for l in self.modules():
                if isinstance(l, BartAttention):
                    for p in l.parameters():
                        p.requires_grad = False
        if kwargs['freeze_mlp']:
            for l in self.bart_lm.modules():
                if isinstance(l, Linear):
                    for p in l.parameters():
                        p.requires_grad = False
        if kwargs['freeze_dec_attn']:
            for l in self.bart_lm.model.decoder.modules():
                if isinstance(l, BartAttention):
                    for p in l.parameters():
                        p.requires_grad = False
        if kwargs['freeze_dec_mlp']:
            for l in self.bart_lm.model.decoder.layers:
                for p in l.fc1.parameters():
                    p.requires_grad = False
                for p in l.fc2.parameters():
                    p.requires_grad = False
        if kwargs['freeze_dec_self_attn']:
            for l in self.bart_lm.model.decoder.layers:
                for p in l.self_attn.parameters():
                    p.requires_grad = False
        if kwargs['freeze_enc_mlp']:
            for l in self.bart_lm.model.encoder.layers:
                for p in l.fc1.parameters():
                    p.requires_grad = False
                for p in l.fc2.parameters():
                    p.requires_grad = False
        if kwargs['freeze_enc_attn']:
            for l in self.bart_lm.model.encoder.layers:
                for p in l.self_attn.parameters():
                    p.requires_grad = False
        
    # Custom implementation of the Bart forward function
    def forward(self,
                audio_features=None,
                cond_tokens=None,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
        ):
        
        audio_embs = self.audio_enc(audio_features)
        
        if self.audio_adapt is not None:
            audio_embs = self.audio_adapt(audio_embs)
        else:
            audio_embs = audio_features
        #print('shape of audio_embs',audio_embs.size())
        
        # Encoder pass
        encoder_outputs = self.bart_lm.model.encoder(
                    input_ids=None,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=audio_embs,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True)['last_hidden_state']
        
        encoder_outputs = encoder_outputs.squeeze(1)
        #print('Shape of encoder output',encoder_outputs.shape)
        
        return encoder_outputs




class OpenL3Embeddings(nn.Module):

    def __init__(self,  **kwargs):
        super(OpenL3Embeddings, self).__init__()


        # self.fc = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 512, bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5)
        # )



        # self.fc2 = nn.Linear(512, kwargs["out_dim"], bias=True)
        #self.bn0 = nn.BatchNorm2d(512)

        self.lstm = nn.Sequential(
            # Conv2D block1
            nn.LSTM(512, 512, 3,dropout=0.2,batch_first=True,bidirectional=True),
            

            
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Linear(1024, kwargs["out_dim"], bias=True)

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
        #x = torch.mean(x, dim=2)  # average over time
        #x = x1 + x2  # (N, 2048)

        # x = self.fc(x)  # (N, 2048)
        #print(x.size())
        # x = self.fc2(x)  # (N, embed_dim)
        #print('open l3',x.size())
        #x = x.unsqueeze(1)

        #x = x.transpose(1, 3)
        #x = self.bn0(x)
        #x = x.transpose(1, 3)

        x,_ = self.lstm(x)
        x = x[:,-1]
        #print('size of x',x.size())
        #x = torch.mean(x, dim=3) # (N, 2048, T/64)

        #(x1, _) = torch.max(x, dim=2)  # max across time
        #x2 = torch.mean(x, dim=2)  # average over time
        #x = x1 + x2  # (N, 2048)

        x = self.fc(x)  # (N, 2048)
        x = self.fc2(x)  # (N, embed_dim)
        #x = torch.mean(x,dim=1)

        #print(x.size())
        return x
