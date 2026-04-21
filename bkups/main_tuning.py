import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from scipy.io import loadmat
import glob
from datetime import datetime as dt
import argparse
from tqdm import tqdm
import random

#import numpy as np
#import sys
#import time
#import json

from transformer_modules import MilliTransNet

from functools import partial
from ray import tune, air
from ray import train as ray_train
import tempfile
import ray.cloudpickle as pickle
#from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler


# specify fixed parameters
INPUT_DIR = os.path.abspath(os.path.join('..', 'input_data_batches'))
currtime = [dt.now().day, dt.now().month, dt.now().year, dt.now().hour, dt.now().minute]
#MODEL_DIR = os.path.join('tuning_models',
#                         f'model_{currtime[0]}_{currtime[1]}_{currtime[2]}_{currtime[3]}_{currtime[4]}')
DATA_PARAMS = loadmat(os.path.join(INPUT_DIR, 'data_parameters.mat'))
NJOINTS = 25
JOINTS_IGNORED_IDX = [6, 10, 15, 19, 21, 23, 22, 24]  # joints ignored: wrists, feet, hands (21/23), thumbs (22/24)

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--nSeq', type=int, default=10, help='Number of sequences used in each trial [default: 10]')
parser.add_argument('--maxEpochs', type=int, default=10, help='Maximum number of epochs each trial will run for [default: 10]')
parser.add_argument('--n_gpus', type=int, default=1, help='Number of gpus which can be used [default: 1]')
FLAGS = parser.parse_args()

N_SEQ = FLAGS.nSeq
MAX_EPOCHS = FLAGS.maxEpochs
N_GPUS = FLAGS.n_gpus

DEVICE = "cuda:0" if torch.cuda.is_available() and N_GPUS != 0 else "cpu"

LR = tune.loguniform(1e-4, 1e-2)
WD = tune.loguniform(1e-4,1e-2)
MAX_NORM = tune.loguniform(0.1,10)
OUTPUT_PROCESSING = tune.choice(['simple','spl'])
SHAPE_CODE_SIZE = tune.choice([2 ** i for i in range(6, 9)])
N_LAYERS = tune.choice([1, 2])
N_HEADS = tune.choice([8, 16, 32])
N_DROPOUT = tune.choice([0.2, 0.3, 0.4])
FWD_EXPANSION = tune.choice([2, 3, 4])
MODEL_INIT = tune.choice(['kaiming_uniform','kaiming_normal','xavier_uniform','xavier_normal'])
OPTIMIZER = tune.choice(['Adam', 'SGD', 'RMSProp'])

JOINTS_IDX = torch.tensor([(i not in JOINTS_IGNORED_IDX) for i in list(range(NJOINTS))])
NRANGES = int(DATA_PARAMS['nRanges']) if 'nRanges' in DATA_PARAMS else 104
NDOPPLER = int(DATA_PARAMS['dopplerFFTSize']) if 'dopplerFFTSize' in DATA_PARAMS else 32
LONG_CH_SIZE = int(DATA_PARAMS['angleFFTSize']) if 'angleFFTSize' in DATA_PARAMS else 8
SHORT_CH_SIZE = 2
SEQ_LEN = int(DATA_PARAMS['seqLen']) if 'seqLen' in DATA_PARAMS else 32
# 1 means already scaled to (0,1) or (-1,1)
MAX_SIGNAL_VAL_H = float(DATA_PARAMS['max_signal_val_h']) if 'max_signal_val_h' in DATA_PARAMS else 1
MAX_SIGNAL_VAL_V = float(DATA_PARAMS['max_signal_val_v']) if 'max_signal_val_v' in DATA_PARAMS else 1
MAX_1D_VAL = float(DATA_PARAMS['max_1d_val']) if 'max_1d_val' in DATA_PARAMS else 1

def main(nSeq,max_epochs,n_gpus):
    #os.mkdir(MODEL_DIR)
    #os.system('zip code_bkup.zip *.py')
    #os.system(f'mv code_bkup.zip ./{MODEL_DIR}')
    config = {
        "LR": LR,
        "WD": WD,
        "MAX_NORM": MAX_NORM,
        "OUTPUT_PROCESSING": OUTPUT_PROCESSING,
        "SHAPE_CODE_SIZE": SHAPE_CODE_SIZE,
        "N_LAYERS": N_LAYERS,
        "N_HEADS": N_HEADS,
        "N_DROPOUT": N_DROPOUT,
        "FWD_EXPANSION": FWD_EXPANSION,
        "MODEL_INIT": MODEL_INIT,
        "OPTIMIZER": OPTIMIZER
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train),
        resources_per_trial={"cpu": 1, "gpu": n_gpus},
        config=config,
        num_samples=nSeq,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    best_trained_model = MilliTransNet.MilliTransNet(
        NRANGES, NDOPPLER, LONG_CH_SIZE, SHORT_CH_SIZE, SEQ_LEN, JOINTS_IDX,
        best_trial.config["SHAPE_CODE_SIZE"],
        best_trial.config["N_LAYERS"],
        best_trial.config["N_HEADS"],
        best_trial.config["N_DROPOUT"],
        best_trial.config["FWD_EXPANSION"],
        output_process=best_trial.config["OUTPUT_PROCESSING"],
        initializer=best_trial.config["MODEL_INIT"],
        optimizer=best_trial.config["OPTIMIZER"]
    )

    best_trained_model.to(DEVICE)
    #best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    #best_checkpoint_data = best_checkpoint.to_dict()
    #best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    best_ckpt_dir = best_trial.checkpoint.value
    model_state = torch.load(os.path.join(best_ckpt_dir,"model.pt"))
    best_trained_model.load_state_dict(model_state)

    #inf_result = get_inference(best_trained_model, DEVICE)
    #print(f"Best trial test set result: {inf_result}")

def train(config):
    milli_transnet = MilliTransNet.MilliTransNet(
        NRANGES, NDOPPLER, LONG_CH_SIZE, SHORT_CH_SIZE, SEQ_LEN, JOINTS_IDX,
        config["SHAPE_CODE_SIZE"],
        config["N_LAYERS"],
        config["N_HEADS"],
        config["N_DROPOUT"],
        config["FWD_EXPANSION"],
        output_process=config["OUTPUT_PROCESSING"],
        initializer=config["MODEL_INIT"],
        optimizer=config["OPTIMIZER"]
    )
    if torch.cuda.device_count() > 1: milli_transnet = nn.DataParallel(milli_transnet)
    milli_transnet.to(DEVICE)

    #checkpoint = session.get_checkpoint()
    checkpoint = ray_train.get_checkpoint()
    start_epoch = 0
    if checkpoint:
        with ray_train.get_checkpoint().as_directory() as ckpt_dir:
            model_state_dict = torch.load(os.path.join(ckpt_dir,"model.pt"), map_location=DEVICE)
            opt_state_dict = torch.load(os.path.join(ckpt_dir,"optimizer.pt"))
            milli_transnet.module.load_state_dict(model_state_dict)
            milli_transnet.get_optimizer().load_state_dict(opt_state_dict)
            start_epoch = torch.load(os.path.join(ckpt_dir,"extra_state.pt"))["epoch"] + 1
        #checkpoint_state = checkpoint.to_dict()
        #start_epoch = checkpoint_state["epoch"]
        #milli_transnet.load_state_dict(checkpoint_state["net_state_dict"])
        #milli_transnet.get_optimizer().load_state_dict(checkpoint_state["optimizer_state_dict"])

    for epoch in range(start_epoch,MAX_EPOCHS):
        run_one_epoch(milli_transnet,os.path.join(INPUT_DIR,'training'),is_learning=True,
                      lr=config["LR"],
                      wd=config["WD"],
                      max_norm=config["MAX_NORM"])
        curr_loss = run_one_epoch(milli_transnet,os.path.join(INPUT_DIR,'validation'),is_learning=False,
                                  lr=config["LR"],
                                  wd=config["WD"],
                                  max_norm=config["MAX_NORM"])

        with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
            checkpoint = None
            if ray_train.get_context().get_world_rank() == 0:
                torch.save(milli_transnet.module.state_dict(), os.path.join(tmp_ckpt_dir,"model.pt"))
                torch.save(milli_transnet.get_optimizer().state_dict(), os.path.join(tmp_ckpt_dir, "optimizer.pt"))
                torch.save({"epoch":epoch}, os.path.join(tmp_ckpt_dir, "extra_state.pt"))
                checkpoint = ray_train.Checkpoint.from_directory(tmp_ckpt_dir)

        #checkpoint_data = {
        #    "epoch": epoch,
        #    "net_state_dict": milli_transnet.state_dict(),
        #    "optimizer_state_dict": milli_transnet.get_optimizer().state_dict(),
        #}
        ##checkpoint = Checkpoint.from_dict(checkpoint_data)
        ##session.report({"loss": LOSS[-1]},checkpoint=checkpoint)
        #checkpoint = ray_train.Checkpoint.from_dict(checkpoint_data)

        ray_train.report({"loss": curr_loss}, checkpoint=checkpoint)

    print("Training complete")

def run_one_epoch(net, data_dir, is_learning,
                  lr=1e-3,
                  wd=None,
                  max_norm=None):
    net.set_trainable(is_learning)
    if is_learning:
        net.train()
    else:
        net.eval()

    data_files = glob.glob(os.path.join(data_dir,'batch*.mat'))
    random.shuffle(data_files)
    loss_epoch = 0
    n_batches = 0
    for data_item in data_files:
        curr_data_file = loadmat(data_item)
        curr_h_seq = torch.from_numpy((1/MAX_SIGNAL_VAL_H) * curr_data_file['h_seq']).to(torch.float32).to(DEVICE)
        curr_v_seq = torch.from_numpy((1/MAX_SIGNAL_VAL_V) * curr_data_file['v_seq']).to(torch.float32).to(DEVICE)
        curr_joints_seq = torch.from_numpy((1/MAX_1D_VAL) * curr_data_file['joints_seq']).to(torch.float32).to(DEVICE)

        #--------------------------------------------------------------------------------------------------------------#
        # validate data
        valid_data = (list(curr_h_seq.size())[1:] == [NRANGES, NDOPPLER, SHORT_CH_SIZE, LONG_CH_SIZE, SEQ_LEN] and
                      list(curr_v_seq.size())[1:] == [NRANGES, NDOPPLER, LONG_CH_SIZE, SHORT_CH_SIZE, SEQ_LEN] and
                      list(curr_joints_seq.size())[1:] == [NJOINTS, 3, SEQ_LEN])
        if not valid_data:
            print('Discarding this batch, invalid data')
            continue

        valid_sequence_idx = torch.sum(~torch.isnan(torch.flatten(curr_h_seq, start_dim=1)), dim=1) & \
                             torch.sum(~torch.isnan(torch.flatten(curr_v_seq, start_dim=1)), dim=1) & \
                             torch.sum(~torch.isnan(torch.flatten(curr_joints_seq, start_dim=1)), dim=1)
        curr_h_seq = curr_h_seq[valid_sequence_idx,:,:,:,:,:]
        curr_v_seq = curr_v_seq[valid_sequence_idx,:,:,:,:,:]
        curr_joints_seq = curr_joints_seq[valid_sequence_idx,:,:,:]

        # rescale range to (-1,1) from (0,1)
        curr_joints_seq[:,:,0,:] = 2 * curr_joints_seq[:,:,0,:] - 1

        # get JOINTS_IDX joints
        curr_joints_seq = curr_joints_seq[:,JOINTS_IDX,:,:]
        # -------------------------------------------------------------------------------------------------------------#

        n_batches += 1
        curr_batch_size = curr_joints_seq.size(0)
        valid_indices = torch.ones((curr_batch_size, SEQ_LEN), dtype=torch.float32).to(torch.device(DEVICE))  # currently, let's train for regularly sampled input

        curr_pred_joints_seq = net(curr_h_seq, curr_v_seq, valid_indices)
        curr_loss = net.get_loss(curr_joints_seq, curr_pred_joints_seq)
        loss_epoch += curr_loss.item()
        if is_learning:
            net.optimize(curr_loss,lr=lr,weight_decay=wd,max_norm=max_norm)

    if not n_batches:
        print('No batch has run successfully')
        return 1
    loss_epoch /= n_batches

    return loss_epoch

if __name__ == "__main__":
    main(N_SEQ,MAX_EPOCHS,N_GPUS)
