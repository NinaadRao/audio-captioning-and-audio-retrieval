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
        path+str(epoch)
    )

def exec_trial(conf, run, ckp_dir=None):
    data_conf = conf["data_conf"]
    param_conf = conf["param_conf"]

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
        model_state, optimizer_state = torch.load(os.path.join(ckp_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    max_epoch = param_conf["num_epoch"] + 1
    best_results = 0
    
    for epoch in numpy.arange(0, max_epoch):

        if epoch > 0:
            model_utils.train(model, train_dl, objective, optimizer)

        epoch_results = {}
        epoch_results["train_obj"] = model_utils.eval(model, train_dl, objective)
        epoch_results["val_obj"] = model_utils.eval(model, val_dl, objective)
        epoch_results["eval_obj"] = model_utils.eval(model, eval_dl, objective)
        curr_lr = float(optimizer.param_groups[0]['lr'])
        print("\nEpoch: {}/{}".format(epoch, max_epoch))
        print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(epoch_results["train_obj"], curr_lr))
        print("\tEval loss {:.04f}%\t Val Loss {:.04f}".format(epoch_results['eval_obj'], epoch_results['val_obj']))

        epoch_results["stop_metric"] = epoch_results["val_obj"]
        wandb.log(epoch_results)
        if epoch_results['val_obj'] > best_results:
            best_results = epoch_results['val_obj']
            save_model(model, optimizer, lr_scheduler, best_results, epoch, '/home/ubuntu/results/')
            artifact = wandb.Artifact('model_artifact_audio', type='model')
            best_model_path = '/home/ubuntu/results/'+str(epoch)
            artifact.add_file(best_model_path)




        # Reduce learning rate w.r.t validation loss
        lr_scheduler.step(epoch_results["stop_metric"])


        # Save the model to the trial directory: local_dir/exp_name/trial_name/checkpoint_<step>
        # with tune.checkpoint_dir(step=epoch) as ckp_dir:
        #     path = os.path.join(ckp_dir, "checkpoint")
        #     torch.save((model.state_dict(), optimizer.state_dict()), path)
        #
        # # Send the current statistics back to the Ray cluster
        # tune.report(**epoch_results)
    run.finish()


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
    run = wandb.init(
        name="openl3",  ## Wandb creates random run names if you skip this field
        reinit=True,  ### Allows reinaitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project="audio-captioning",  ### Project should be created in your wandb account
        config=conf  ### Wandb Config for your run
    )
    exec_trial(conf, run, ckp_dir=None)
