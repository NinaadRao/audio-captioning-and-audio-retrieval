import os
import random
import time

import numpy
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
import wandb
wandb.login(key="6303eb738fdf6199f76497134963d53a7f8cd9be")
wandb.login(key="6303eb738fdf6199f76497134963d53a7f8cd9be")
from utils import criterion_utils, data_utils, model_utils

torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)


def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'scheduler_state_dict': scheduler.state_dict(),
         'loss': metric,
         'epoch': epoch},
        path+str(epoch)+'.pth'
    )

def exec_trial(conf, ckp_dir=None):
    data_conf = conf["data_conf"]
    param_conf = conf["param_conf"]
    model_type = conf["trial_base"]

    train_ds = data_utils.load_data(data_conf["train_data"])
    train_dl = DataLoader(dataset=train_ds, batch_size=param_conf["batch_size"],
                          shuffle=True, collate_fn=data_utils.collate_fn)

    val_ds = data_utils.load_data(data_conf["val_data"])
    val_dl = DataLoader(dataset=val_ds, batch_size=param_conf["batch_size"],
                        shuffle=True, collate_fn=data_utils.collate_fn)

    eval_ds = data_utils.load_data(data_conf["eval_data"])
    eval_dl = DataLoader(dataset=eval_ds, batch_size=param_conf["batch_size"],
                         shuffle=True, collate_fn=data_utils.collate_fn)

    model_params = conf[param_conf["model"]]
    model = model_utils.init_model(model_params, train_ds.text_vocab)
    print(model)

    obj_params = conf["criteria"][param_conf["criterion"]]
    objective = getattr(criterion_utils, obj_params["name"], None)(**obj_params["args"])

    optim_params = conf[param_conf["optimizer"]]
    optimizer = getattr(optim, optim_params["name"], None)(model.parameters(), **optim_params["args"])

    lr_params = conf[param_conf["lr_scheduler"]]
    lr_scheduler = getattr(optim.lr_scheduler, lr_params["name"], "ReduceLROnPlateau")(optimizer, **lr_params["args"])

    if ckp_dir is not None:
        model_state, optimizer_state = torch.load(os.path.join(ckp_dir, "17"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        torch.save({'model_state_dict':model_state},path+'audio_encoder'+'.pth')

    max_epoch = param_conf["num_epoch"] + 1
    best_results = 1e30
    
    

# Main
if __name__ == "__main__":
    # Load configuration
    with open("conf.yaml", "rb") as stream:
        conf = yaml.full_load(stream)

    # Configure ray-tune clusters



    def trial_name_creator(trial):
        trial_name = "_".join([conf["param_conf"]["model"], trial.trial_id])
        return trial_name


    def trial_dirname_creator(trial):
        trial_dirname = "_".join([time.strftime("%Y-%m-%d"), trial.trial_id])
        return trial_dirname


    # Execute trials - local_dir/exp_name/trial_name
    '''
    run = wandb.init(
        name=conf['trial_base'],  ## Wandb creates random run names if you skip this field
        reinit=True,  ### Allows reinaitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project="audio-captioning",  ### Project should be created in your wandb account
        config=conf  ### Wandb Config for your run
    )
    '''
    exec_trial(conf, ckp_dir=None)
