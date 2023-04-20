import copy
import os

import numpy as np
import torch
from tqdm import tqdm

from models import core


def init_model(params, vocab):
    params = copy.deepcopy(params)

    # Text encoder params
    params["text_enc"]["num_embed"] = len(vocab)

    if params["text_enc"]["init"] == "prior":
        id2vec = {vocab.key2id[key]: vocab.key2vec[key] for key in vocab.key2id}
        weights = np.array([id2vec[id] for id in range(vocab.id)])
        weights = torch.as_tensor(weights, dtype=torch.float)
        params["text_enc"]["weight"] = weights
    else:
        params["text_enc"]["weight"] = None

    # Audio encoder params
    if params["audio_enc"]["init"] == "prior":
        params["audio_enc"]["weight"] = torch.load(params["audio_enc"]["weight"])
    else:
        params["audio_enc"]["weight"] = None

    # Initialize dual encoders
    model = getattr(core, "DualEncoderModel", None)(params["audio_enc"]["name"], params["text_enc"]["name"], **params)

    return model


def train(model, data_loader, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    model.train()

    for batch_idx, data in enumerate(data_loader, 0):
        item_batch, audio_batch, text_batch = data

        audio_batch = audio_batch.to(device)
        text_batch = text_batch.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        audio_embeds, text_embeds = model(audio_batch, text_batch)
        loss = criterion(audio_embeds, text_embeds, item_batch)
        total_loss = loss.item()
        loss.backward()
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (batch_idx + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))
        batch_bar.update()
        optimizer.step()
    batch_bar.close()

def eval(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
    model.eval()

    eval_loss, eval_steps = 0.0, 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            item_batch, audio_batch, text_batch = data

            audio_batch = audio_batch.to(device)
            text_batch = text_batch.to(device)

            audio_embeds, text_embeds = model(audio_batch, text_batch)
            loss = criterion(audio_embeds, text_embeds, item_batch)

            eval_loss += loss.cpu().numpy()
            eval_steps += 1
            batch_bar.set_postfix(loss="{:.04f}".format(float(eval_loss / (batch_idx + 1))))
            batch_bar.update()
    batch_bar.close()
    return eval_loss / max(eval_steps, 1)


def restore(model, ckp_dir,epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_values = torch.load(os.path.join(ckp_dir, str(epoch)+".pth"),
                                              map_location=device)
    model.load_state_dict(model_values["model_state_dict"])
    return model
