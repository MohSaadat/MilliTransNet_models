import torch
import numpy as np
import os
import sys
from scipy.io import loadmat
from datetime import datetime as dt
import argparse
import time
import json
from tqdm import tqdm
import random

from transformer_modules import MilliTransNet
from Utils.DataUtils import get_dataset
from Utils.get_device import get_inference_device_config

# Directories
INPUT_DIR = os.path.join('..','input_data')
TRAINING_DIR = os.path.join(INPUT_DIR,'training')
VALIDATION_DIR = os.path.join(INPUT_DIR,'validation')
os.makedirs('saved_models', exist_ok=True)
MODEL_DIR = os.path.join(
    'saved_models',
    f'model_{dt.now().day}_{dt.now().month}_{dt.now().year}_{dt.now().hour}_{dt.now().minute}'
)

# Data processing parameters
DATA_PARAMS = loadmat(os.path.join(INPUT_DIR,'data_parameters.mat'))
NJOINTS = 25
JOINTS_IGNORE_IDX = [6,10,15,19,21,23,22,24] # joints ignored: wrists, feet, hands (21/23), thumbs (22/24)

# Tracking training progress
ALL_LR = []
ALL_WD = []
TRAINING_LOSS = []
VALIDATION_LOSS = []

def initialize_run():
    '''
    Intialize the training+validation:
    * parse arguments
    * set device
    * create the config dictionary and return it
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size [default: 32]')
    parser.add_argument('--epochs', type=int, default=50, help='Epoch to run [default: 50]')
    parser.add_argument('--gpu', type=int, default=None, help='GPU to use [default: auto-select a free GPU; use -1 for CPU]')
    parser.add_argument('--gpu_threshold_gb', type=float, default=4.0, help='Minimum free GPU memory required for auto-selection [default: 4.0]')
    parser.add_argument('--restore_dir', type=str, default=None, help='Checkpoint restore directory - '
                                                                      'set None if checkpoint is not to be restored [default: None]')

    parser.add_argument('--base_lr', type=float, default=1e-3, help='Base learning rate [default: 0.001]')
    # we will decay lr once after every epoch: to get lr decreased to 12% of initial --> decay_rate = 0.96 for 50 epochs, decay_rate = 0.98 for 100 epochs...
    parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')

    parser.add_argument('--output_process', type=str, default='simple', help='...[default: simple]')
    parser.add_argument('--shape_code_size', type=int, default=128, help='...[default: 128]')
    parser.add_argument('--n_layers', type=int, default=2, help='...[default: 2]')
    parser.add_argument('--n_heads', type=int, default=16, help='...[default: 16]')
    parser.add_argument('--n_dropout', type=float, default=0.3, help='...[default: 0.3]')
    parser.add_argument('--forward_expansion', type=int, default=4, help='...[default: 4]')
    parser.add_argument('--model_init', type=str, default='kaiming_uniform', help='...[default: kaiming_uniform]')

    parser.add_argument('--optimizer', type=str, default='Adam', help='...[default: Adam]')
    parser.add_argument('--base_weight_decay', type=float, default=2e-4, help='...[default: 2e-4]')
    parser.add_argument('--grad_norm_lim', type=float, default=1.0, help='...<0.0 indicates no norm clipping> [default: 1.0]')

    flags = parser.parse_args()
    device_config = get_inference_device_config(
        device_index=flags.gpu,
        threshold_gb=flags.gpu_threshold_gb
    )
    config = {
        'BATCH_SIZE': flags.batch_size,
        'N_EPOCHS': flags.epochs,
        'REQUESTED_GPU': flags.gpu,
        'GPU': device_config['resolved_device_index'],
        'GPU_THRESHOLD_GB': flags.gpu_threshold_gb,
        'AVAILABLE_GPU_INDICES': device_config['available_device_indices'],
        'RESTORE_DIR': flags.restore_dir,
        'NJOINTS': NJOINTS,
        'BASE_LR': flags.base_lr,
        'DECAY_RATE': flags.decay_rate,
        'OUTPUT_PROCESSING': flags.output_process,
        'SHAPE_CODE_SIZE': flags.shape_code_size,
        'N_LAYERS': flags.n_layers,
        'N_HEADS': flags.n_heads,
        'N_DROPOUT': flags.n_dropout,
        'FWD_EXPANSION': flags.forward_expansion,
        'MODEL_INIT': flags.model_init,
        'OPTIMIZER': flags.optimizer,
        'BASE_WEIGHT_DECAY': flags.base_weight_decay,
        'GRAD_NORM_LIM': flags.grad_norm_lim
    }
    config['DEVICE'] = device_config['torch_device']
    print(
        f'Running on {config["DEVICE"]} '
        f'(requested gpu={config["REQUESTED_GPU"]}, resolved gpu={config["GPU"]})'
    )

    config['JOINTS_IDX'] = torch.tensor([(i not in JOINTS_IGNORE_IDX) for i in range(NJOINTS)])
    config['NRANGES'] = int(DATA_PARAMS['nRanges']) if 'nRanges' in DATA_PARAMS else 104
    config['NDOPPLER'] = int(DATA_PARAMS['dopplerFFTSize']) if 'dopplerFFTSize' in DATA_PARAMS else 32
    config['LONG_CH_SIZE'] = int(DATA_PARAMS['angleFFTSize']) if 'angleFFTSize' in DATA_PARAMS else 8
    config['SHORT_CH_SIZE'] = 2
    config['SEQ_LEN'] = int(DATA_PARAMS['seqLen']) if 'seqLen' in DATA_PARAMS else 32
    # 1 means already scaled to (0,1) or (-1,1)
    config['MAX_SIGNAL_VAL_H'] = float(DATA_PARAMS['max_signal_val_h']) if 'max_signal_val_h' in DATA_PARAMS else 1
    config['MAX_SIGNAL_VAL_V'] = float(DATA_PARAMS['max_signal_val_v']) if 'max_signal_val_v' in DATA_PARAMS else 1
    config['MAX_1D_VAL'] = float(DATA_PARAMS['max_1d_val']) if 'max_1d_val' in DATA_PARAMS else 1

    config['PARAMS'] = {
        # training parameters
        'BATCH SIZE':           config['BATCH_SIZE'],
        'EPOCHS':               config['N_EPOCHS'],
        'GPU':                  config['GPU'],
        'REQUESTED_GPU':        config['REQUESTED_GPU'],
        'GPU_THRESHOLD_GB':     config['GPU_THRESHOLD_GB'],
        'RESTORE_DIR':          config['RESTORE_DIR'],
        'BASE_LR':              config['BASE_LR'],
        'DECAY_RATE':           config['DECAY_RATE'],

        # model parameters
        'GAN':                  False,
        'OUTPUT_PROCESSING':    config['OUTPUT_PROCESSING'],
        'SHAPE_CODE_SIZE':      config['SHAPE_CODE_SIZE'],
        'N_LAYERS':             config['N_LAYERS'],
        'N_HEADS':              config['N_HEADS'],
        'N_DROPOUT':            config['N_DROPOUT'],
        'FWD_EXPANSION':        config['FWD_EXPANSION'],

        # training parameters
        'MODEL_INIT':           config['MODEL_INIT'],
        'OPTIMIZER':            config['OPTIMIZER'],
        'BASE_WEIGHT_DECAY':    config['BASE_WEIGHT_DECAY'],
        'GRAD_NORM_LIM':        config['GRAD_NORM_LIM'],

        # output data parameters
        'NJOINTS':              NJOINTS,
        'JOINTS_IGNORE_IDX':    JOINTS_IGNORE_IDX,

        # input data parameters
        'NRANGES':              config['NRANGES'],
        'NDOPPLER':             config['NDOPPLER'],
        'LONG_CH_SIZE':         config['LONG_CH_SIZE'],
        'SHORT_CH_SIZE':        config['SHORT_CH_SIZE'],
        'SEQ_LEN':              config['SEQ_LEN'],

        'MAX_SIGNAL_VAL_H':     config['MAX_SIGNAL_VAL_H'],
        'MAX_SIGNAL_VAL_V':     config['MAX_SIGNAL_VAL_V'],
        'MAX_1D_VAL':           config['MAX_1D_VAL']
    }

    ALL_LR.clear()
    ALL_LR.append(config['BASE_LR'])
    ALL_WD.clear()
    ALL_WD.append(config['BASE_WEIGHT_DECAY'])
    TRAINING_LOSS.clear()
    VALIDATION_LOSS.clear()
    return config

def train_n_validate(config):
    if os.path.exists(MODEL_DIR):
        print(f'Model directory {MODEL_DIR} already exists. Exiting to avoid overwriting previous runs.')
        sys.exit(1)
    os.mkdir(MODEL_DIR)
    os.system('zip code_bkup.zip *.py')
    os.system(f'mv code_bkup.zip ./{MODEL_DIR}')
    with open(os.path.join(MODEL_DIR, 'parameters.json'), 'w') as f:
        f.write(json.dumps(config['PARAMS'], default=str))

    if config['RESTORE_DIR']:
        milli_transnet = torch.load(
            os.path.join(config['RESTORE_DIR'],'milli_transnet_final.pth'),
            map_location=config['DEVICE']
        )
        milli_transnet.set_device(config['DEVICE'])
    else:
        milli_transnet = MilliTransNet.MilliTransNet(config['NRANGES'],config['NDOPPLER'],config['LONG_CH_SIZE'],config['SHORT_CH_SIZE'],config['SEQ_LEN'],
                                                     config['JOINTS_IDX'],
                                                     config['SHAPE_CODE_SIZE'],config['N_LAYERS'],config['N_HEADS'],config['N_DROPOUT'],config['FWD_EXPANSION'],
                                                     output_process=config['OUTPUT_PROCESSING'],
                                                     initializer=config['MODEL_INIT'],
                                                     optimizer=config['OPTIMIZER'])
        milli_transnet.set_device(config['DEVICE'])

    h_seq_train, v_seq_train, joints_seq_train = get_dataset(TRAINING_DIR, config)
    h_seq_valid, v_seq_valid, joints_seq_valid = get_dataset(VALIDATION_DIR, config)
    # for seqL = 32 (fixed by data pre-processing): we get
    # 1727 training sequences
    # 560 validation sequences

    #----------------------------------------------------#
    #import matplotlib.pyplot as plt
    #h_data, v_data = torch.flatten(h_seq_train).numpy(), torch.flatten(v_seq_train).numpy()
    #print([np.mean(h_data),np.std(h_data)])
    #print([np.mean(v_data),np.std(v_data)])
    #plt.hist(h_data, color='red', bins=100, label='Azim.')
    #plt.hist(v_data, color='blue', bins=100, label='Elev.')
    #plt.show()
    #exit()
    #----------------------------------------------------#

    curr_lr = config['BASE_LR']
    curr_wd = config['BASE_WEIGHT_DECAY']
    for n_epochs in range(config['N_EPOCHS']):
        print(f'Epoch: {n_epochs}')
        begin_time = time.time()
        run_one_epoch(milli_transnet,[h_seq_train,v_seq_train,joints_seq_train],config,is_learning=True,curr_lr=curr_lr,curr_wd=curr_wd)
        run_one_epoch(milli_transnet,[h_seq_valid,v_seq_valid,joints_seq_valid],config,is_learning=False,curr_lr=curr_lr,curr_wd=curr_wd)
        end_time = time.time()

        curr_lr = curr_lr * config['DECAY_RATE']
        curr_wd = curr_wd * config['DECAY_RATE']
        ALL_LR.append(curr_lr)
        ALL_WD.append(curr_wd)
        print(f'Learning rate updated to: {curr_lr}')
        print(f'Weight decay updated to: {curr_wd}')

        print(f'Training loss: {TRAINING_LOSS[-1]} \t Validation loss: {VALIDATION_LOSS[-1]}')
        print(f'Executed in {end_time-begin_time} seconds')
        if n_epochs % 10 == 0:
            print(f'Saving model to {MODEL_DIR}')
            torch.save(milli_transnet, os.path.join(MODEL_DIR, f'milli_transnet_ep{n_epochs}.pth'))
            np.savez(os.path.join(MODEL_DIR, 'loss.npz'),
                     TRAINING_LOSS=TRAINING_LOSS,
                     VALIDATION_LOSS=VALIDATION_LOSS)
            np.savez(os.path.join(MODEL_DIR, 'lr.npz'),
                     LEARNING_RATES=ALL_LR)

    print(f'Saving model to {MODEL_DIR}')
    torch.save(milli_transnet, os.path.join(MODEL_DIR, f'milli_transnet_final.pth'))
    np.savez(os.path.join(MODEL_DIR, 'loss.npz'),
             TRAINING_LOSS=TRAINING_LOSS,
             VALIDATION_LOSS=VALIDATION_LOSS)
    np.savez(os.path.join(MODEL_DIR, 'lr.npz'),
             LEARNING_RATES=ALL_LR)

def run_one_epoch(net,dataset,config,is_learning,curr_lr,curr_wd):
    net.set_trainable(is_learning)
    device = config['DEVICE']

    if is_learning:
        net.train()
    else:
        net.eval()

    h_seq, v_seq, joints_seq = dataset
    n_datapoints = joints_seq.size(0)
    data_perm = torch.randperm(n_datapoints)
    h_seq = h_seq[data_perm,:,:,:,:,:]
    v_seq = v_seq[data_perm,:,:,:,:,:]
    joints_seq = joints_seq[data_perm,:,:,:]
    n_batches = n_datapoints // config['BATCH_SIZE']

    loss_epoch = 0
    for b in tqdm(range(n_batches)):
        curr_h_seq = h_seq[b*config['BATCH_SIZE']:min(n_datapoints,(b+1)*config['BATCH_SIZE']),:,:,:,:,:].to(device)
        curr_v_seq = v_seq[b*config['BATCH_SIZE']:min(n_datapoints,(b+1)*config['BATCH_SIZE']),:,:,:,:,:].to(device)
        curr_joints_seq = torch.permute(joints_seq[b*config['BATCH_SIZE']:min(n_datapoints,(b+1)*config['BATCH_SIZE']),:,:,:],(0,3,1,2)).view(config['BATCH_SIZE'],config['SEQ_LEN'],-1).to(device)

        valid_indices = torch.ones((config['BATCH_SIZE'],config['SEQ_LEN']), dtype=torch.float32).to(device) # currently, let's train for regularly sampled input

        curr_pred_joints_seq = net(curr_h_seq,curr_v_seq,valid_indices)
        curr_loss = net.get_loss(curr_joints_seq,curr_pred_joints_seq)
        loss_epoch += curr_loss
        if is_learning:
            net.optimize(curr_loss,lr=curr_lr,weight_decay=curr_wd,max_norm=config['GRAD_NORM_LIM'])
    loss_epoch /= n_batches
    if is_learning:
        TRAINING_LOSS.append(loss_epoch.detach().cpu())
    else:
        VALIDATION_LOSS.append(loss_epoch.detach().cpu())

def main():
    config = initialize_run()
    train_n_validate(config)

if __name__=="__main__":
    main()
