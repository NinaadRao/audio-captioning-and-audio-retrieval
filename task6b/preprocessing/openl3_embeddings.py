import torchopenl3
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

# output_path = args.output_path
# data_partition = path.split('/')[-2]
# files = set(os.listdir(path))
# output_files = os.listdir(output_path+'embeddings/'+data_partition)
# output_files = ['.'.join(i.split('.')[:-1])+'.wav' for i in output_files]
# output_files = set(output_files)
# print(len(output_files),len(files))

# files_output = files - output_files
# print(len(files_output))
# files = list(files_output)
# files.sort()

# # partition = int(args.partition)

# # partition_number = int(args.partition_number)

# total_data = len(files)
# # each_partition = total_data//partition
# # if partition_number == partition:
# # 	files = files[(partition_number-1)*each_partition:]
# # else:
# # 	files = files[(partition_number-1)*each_partition:(partition_number)*each_partition]

# print('Number of files to do',len(files))
# for i in range(len(files)):
# 	if(i%10==0):
# 		print('Files done',i)
# 	if '.wav' in files[i]:
# 		audio,sr = sf.read(path+files[i])
# 		output_file_name = '.'.join(files[i].split('.')[:-1])
# 		emb, ts = torchopenl3.get_audio_embedding(audio, sr)
# 		# print('output_path',output_path+'embeddings/'+data_partition+'/'+output_file_name+'.pt')
# 		# torch.save(emb,output_path+'embeddings/'+data_partition+'/'+output_file_name+'.pt')
# 		# torch.save(ts,output_path+'ts/'+data_partition+'/'+output_file_name+'.pt')



for split in global_params["audio_splits"]:
    count = 0
    fid2fname = audio_fid2fname[split]
    fname2fid = {fid2fname[fid]: fid for fid in fid2fname}

    audio_dir = os.path.join(global_params["dataset_dir"], split)
    audio_logmel = os.path.join(global_params["dataset_dir"], f"{split}_audio_logmels_openl3.hdf5")
    model = torchopenl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",
                                                 embedding_size=512)
    print(audio_logmel,'this is where it will save')
    with h5py.File(audio_logmel, "w") as stream:

        for fpath in glob.glob(r"{}/*.wav".format(audio_dir)):
            # try:
            fname = os.path.basename(fpath)
            fid = fname2fid[fname]

            audio,sr = sf.read(fpath)
            emb, ts = torchopenl3.get_audio_embedding(audio, sr, model=model)

            stream[fid] = emb.cpu().data.numpy()  # [Time, Mel]
            print(fid, fname)
            


            # except:
            #     print("Error audio file:", fpath)
