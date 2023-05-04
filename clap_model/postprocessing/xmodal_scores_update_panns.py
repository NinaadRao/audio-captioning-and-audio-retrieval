import dbm
import json
import os
import shelve
from dbm import dumb

import nltk
import torch
import sys
import gc
sys.path.append("/home/ubuntu/dcase2023-audio-retrieval/")
from utils import criterion_utils, data_utils, model_utils

dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}

stopwords = nltk.corpus.stopwords.words("english")
DEVICE = "cuda:0"

# Trial info
trial_base = "/home/ubuntu/dcase2023-audio-retrieval/results/"
trial_series = "~"
trial_name = "~"
ckp_dir = "pann-qa-multi/"

# Model checkpoint directory
ckp_fpath = os.path.join(trial_base, ckp_dir)

# Load trial parameters
conf_fpath = os.path.join(ckp_fpath + "params.json")
with open(conf_fpath, "rb") as store:
    conf = json.load(store)
print("Load", conf_fpath)

# Load data
data_conf = conf["data_conf"]
#train_ds = data_utils.load_data(data_conf["train_data"])
val_ds = data_utils.load_data(data_conf["val_data"])
eval_ds = data_utils.load_data(data_conf["eval_data"])

# Restore model checkpoint
param_conf = conf["param_conf"]
model_params = conf[param_conf["model"]]
obj_params = conf["criteria"][param_conf["criterion"]]
model = model_utils.init_model(model_params, val_ds.text_vocab)
model = model_utils.restore(model, ckp_fpath, "best_model.pth")
print(model)

model.eval()

model = model.to(DEVICE)
# train_ds.to(DEVICE)
# val_ds.to(DEVICE)
# eval_ds.to(DEVICE)

for name, ds in zip(["val", "eval"], [val_ds, eval_ds]):
    torch.cuda.empty_cache()
    gc.collect()
    text2vec = {}
    for idx in ds.text_data.index:
        item = ds.text_data.iloc[idx]

        if ds.text_level == "word":
            text_vec = torch.as_tensor([ds.text_vocab(key) for key in item["tokens"] if key not in stopwords])
            text2vec[item["tid"]] = torch.unsqueeze(text_vec, dim=0)

        elif ds.text_level == "sentence":
            text_vec = torch.as_tensor([ds.text_vocab(item["tid"])])
            text2vec[item["tid"]] = torch.unsqueeze(text_vec, dim=0)

    # Compute pairwise cross-modal scores
    score_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")
    print(name)
    vec2embed = {}
    for tid in text2vec:
        td = text2vec[tid]
        td = td.to(DEVICE)
        text_embed = model.text_branch(td)[0]  # Encode text data
        vec2embed[tid] = text_embed
   
    torch.cuda.empty_cache()
    gc.collect()

    with shelve.open(filename=score_fpath, flag="n", protocol=2) as stream:
        for fid in ds.text_data["fid"].unique():
            print('fid',fid)
            group_scores = {}

            # Encode audio data
            audio_vec = torch.as_tensor(ds.audio_data[fid][()])
            audio_vec = torch.unsqueeze(audio_vec, dim=0)
            audio_vec = audio_vec.to(DEVICE)
            audio_embed = model.audio_branch(audio_vec)[0]

            for tid in text2vec:
                xmodal_S = criterion_utils.score(audio_embed, vec2embed[tid], obj_params["args"].get("dist", "dot_product"))
                group_scores[tid] = xmodal_S.item()
            stream[fid] = group_scores
    print("Save", score_fpath)
