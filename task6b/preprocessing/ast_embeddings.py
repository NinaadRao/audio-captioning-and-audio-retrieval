import gc
import soundfile as sf
import os
import torch
from argparse import ArgumentParser     
import logging
import glob
import h5py
import pickle
import numpy as np
parser = ArgumentParser()
from transformers import AutoFeatureExtractor, ClapModel
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
import librosa


#model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
#feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#model     = model.to(device)
#feature_extractor = feature_extractor.to(device)
parser.add_argument('-path',default='/home/ubuntu/drive/data/Clotho/')
parser.add_argument('-output_path',default='/home/ubuntu/drive/data/openl3_embeddings/')
# parser.add_argument('--dataset_dir',default='10')
# parser.add_argument('-partition_number',default='1')

args, _ = parser.parse_known_args()



path = args.path


global_params = {
    "dataset_dir": args.path,
    "audio_splits": ["development", "validation", "evaluation"]
}

# Load audio info
audio_info = os.path.join(global_params["dataset_dir"], "audio_info.pkl")
with open(audio_info, "rb") as store:
    audio_fid2fname = pickle.load(store)["audio_fid2fname"]

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = model.to(device)


for split in global_params["audio_splits"]:
    count = 0
    fid2fname = audio_fid2fname[split]
    fname2fid = {fid2fname[fid]: fid for fid in fid2fname}

    audio_dir = os.path.join(global_params["dataset_dir"], split)
    audio_logmel = os.path.join(global_params["dataset_dir"], f"{split}_audio_logmels_ast.hdf5")
    print(audio_logmel,'this is where it will save')
    with h5py.File(audio_logmel, "w") as stream:

        for fpath in glob.glob(r"{}/*.wav".format(audio_dir)):
            # try:
            fname = os.path.basename(fpath)
            fid = fname2fid[fname]

            audio,sr = librosa.load(fpath,sr=16000)
            inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
            inputs = inputs.to(device)
            #emb = model.get_audio_features(**inputs)
            with torch.no_grad():
                logits = model(**inputs).logits

            #print('type of emb',type(emb))
            stream[fid] = logits.cpu().detach().numpy()  # [Time, Mel]
            print(fid, fname)
            torch.cuda.empty_cache()
            gc.collect()            


            # except:
            #     print("Error audio file:", fpath)
