import os
import pickle
import torch
import gc
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import XLNetTokenizer, XLNetModel

global_params = {
    "dataset_dir": "/home/ubuntu/dcase2023-audio-retrieval/data/Clotho",
    "audio_splits": ["development", "validation", "evaluation"]
}


model_name = "xlnet"
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetModel.from_pretrained("xlnet-base-cased")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# %%

text_embeds = {}

for split in global_params["audio_splits"]:

    text_fpath = os.path.join(global_params["dataset_dir"], f"{split}_text.csv")
    text_data = pd.read_csv(text_fpath)

    for i in text_data.index:
        tid = text_data.iloc[i].tid
        raw_text = text_data.iloc[i].raw_text

        print(split, tid, raw_text)

        inputs = tokenizer(raw_text, return_tensors="pt")
        inputs = inputs.to(device)

        output = model(**inputs).last_hidden_state
        output = output.to("cpu")

        text_embeds[tid] = output.detach().numpy()

        torch.cuda.empty_cache()
        gc.collect()

# Save text embeddings
embed_fpath = os.path.join(global_params["dataset_dir"], f"{model_name}_embeds.pkl")

with open(embed_fpath, "wb") as stream:
    pickle.dump(text_embeds, stream)

print("Save text embeddings to", embed_fpath)
