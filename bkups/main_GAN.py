import torch
import numpy as np
import os
import sys
from scipy.io import loadmat
import glob
from datetime import datetime as dt
import argparse
import time
import json
from tqdm import tqdm

from dnn_modules import SkeletalDiscriminator
from transformer_modules import MilliTransNet

# fixed
INPUT_DIR = os.path.join('..','input_data')
TRAINING_DIR = os.path.join(INPUT_DIR,'training')
VALIDATION_DIR = os.path.join(INPUT_DIR,'validation')
currtime = [dt.now().day, dt.now().month, dt.now().year, dt.now().hour, dt.now().minute]
MODEL_DIR = os.path.join('saved_models', f'model_{currtime[0]}_{currtime[1]}_{currtime[2]}_{currtime[3]}_{currtime[4]}')
DATA_PARAMS = loadmat(os.path.join(INPUT_DIR,'data_parameters.mat'))
NJOINTS = 25
JOINTS_IGNORED_IDX = [6,10,15,19,21,23,22,24] # joints ignored: wrists, feet, hands (21/23), thumbs (22/24)

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='Batch size [default: 32]')
parser.add_argument('--epochs', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]: uses CPU if GPU not available or index is out of range')
parser.add_argument('--base_lr', type=float, default=1e-3, help='Base learning rate for model [default: 0.001]')
parser.add_argument('--base_d_lr', type=float, default=1e-3, help='Base learning rate for discriminator [default: 0.0001]')
parser.add_argument('--restore_dir', type=str, default=None, help='Checkpoint restore directory - '
                                                                  'set None if checkpoint is not to be restored [default: None]')

# we will decay lr once after every epoch: to get lr decreased to 12% of initial --> decay_rate = 0.96 for 50 epochs, decay_rate = 0.98 for 100 epochs...
parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate (network) [default:0.96]')
parser.add_argument('--decay_rate_d', type=float, default=0.90, help='Learning rate decay rate (discriminator) [default:0.90]')

parser.add_argument('--shape_code_size', type=int, default=128, help='...[default: 128]')
parser.add_argument('--n_layers', type=int, default=2, help='...[default: 2]')
parser.add_argument('--n_heads', type=int, default=16, help='...[default: 16]')
parser.add_argument('--n_dropout', type=float, default=0.3, help='...[default: 0.3]')
parser.add_argument('--forward_expansion', type=int, default=4, help='...[default: 4]')
parser.add_argument('--d_loss_w', type=float, default=1e-2, help='...[default: 1e-2]')
parser.add_argument('--model_init', type=str, default='kaiming_uniform', help='...[default: kaiming_uniform]')
parser.add_argument('--optimizer', type=str, default='Adam', help='...[default: Adam]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
N_EPOCHS = FLAGS.epochs
GPU = FLAGS.gpu
BASE_LR = FLAGS.base_lr
BASE_D_LR = FLAGS.base_d_lr
RESTORE_DIR = FLAGS.restore_dir
DECAY_RATE = FLAGS.decay_rate
DECAY_RATE_D = FLAGS.decay_rate_d
SHAPE_CODE_SIZE = FLAGS.shape_code_size
N_LAYERS = FLAGS.n_layers
N_HEADS = FLAGS.n_heads
N_DROPOUT = FLAGS.n_dropout
FWD_EXPANSION = FLAGS.forward_expansion
D_LOSS_W = FLAGS.d_loss_w
MODEL_INIT = FLAGS.model_init
OPTIMIZER = FLAGS.optimizer
DEVICE = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() and GPU in range(torch.cuda.device_count()) else 'cpu')
print(f'Running on {DEVICE}')

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

# Store processing and training parameters
PARAMS = {
    # training parameters
    'BATCH SIZE':           BATCH_SIZE,
    'EPOCHS':               N_EPOCHS,
    'GPU':                  GPU,
    'BASE_LR':              BASE_LR,
    'BASE_D_LR':            BASE_D_LR,
    'RESTORE_DIR':          RESTORE_DIR,
    'DECAY_RATE':           DECAY_RATE,
    'DECAY_RATE_D':         DECAY_RATE_D,

    # model parameters
    'GAN':                  True,
    'SHAPE_CODE_SIZE':      SHAPE_CODE_SIZE,
    'N_LAYERS':             N_LAYERS,
    'N_HEADS':              N_HEADS,
    'N_DROPOUT':            N_DROPOUT,
    'FWD_EXPANSION':        FWD_EXPANSION,
    'D_LOSS_W':             D_LOSS_W,
    'MODEL_INIT':           MODEL_INIT,
    'OPTIMIZER':            OPTIMIZER,

    # output data parameters
    'NJOINTS':              NJOINTS,
    'JOINTS_IGNORED_IDX':   JOINTS_IGNORED_IDX,

    # input data parameters
    'NRANGES':              NRANGES,
    'NDOPPLER':             NDOPPLER,
    'LONG_CH_SIZE':         LONG_CH_SIZE,
    'SHORT_CH_SIZE':        SHORT_CH_SIZE,
    'SEQ_LEN':              SEQ_LEN,

    'MAX_SIGNAL_VAL_H':     MAX_SIGNAL_VAL_H,
    'MAX_SIGNAL_VAL_V':     MAX_SIGNAL_VAL_V,
    'MAX_1D_VAL':           MAX_1D_VAL
}

curr_lr = BASE_LR
curr_d_lr = BASE_D_LR
ALL_LR = [(curr_lr,curr_d_lr)]
TRAINING_LOSS = []
VALIDATION_LOSS = []

def get_dataset(path):
    data_files = glob.glob(os.path.join(path,'data*.mat'))

    h_seq = torch.empty((0,NRANGES,NDOPPLER,SHORT_CH_SIZE,LONG_CH_SIZE,SEQ_LEN), dtype=torch.float32)
    v_seq = torch.empty((0,NRANGES,NDOPPLER,LONG_CH_SIZE,SHORT_CH_SIZE,SEQ_LEN), dtype=torch.float32)
    joints_seq = torch.empty((0,torch.sum(JOINTS_IDX),3,SEQ_LEN), dtype=torch.float32)
    for i in tqdm(range(len(data_files))):
        file = data_files[i]
        curr_data_file = loadmat(file)
        curr_h_seq = torch.from_numpy((1/MAX_SIGNAL_VAL_H) * curr_data_file['h_seq']).to(torch.float32)
        curr_v_seq = torch.from_numpy((1/MAX_SIGNAL_VAL_V) * curr_data_file['v_seq']).to(torch.float32)
        curr_joints_seq = torch.from_numpy((1/MAX_1D_VAL) * curr_data_file['joints_seq']).to(torch.float32)

        valid_data = (list(curr_h_seq.size())[1:] == [NRANGES,NDOPPLER,SHORT_CH_SIZE,LONG_CH_SIZE,SEQ_LEN] and
                      list(curr_v_seq.size())[1:] == [NRANGES,NDOPPLER,LONG_CH_SIZE,SHORT_CH_SIZE,SEQ_LEN] and
                      list(curr_joints_seq.size())[1:] == [NJOINTS,3,SEQ_LEN])
        if not valid_data:
            continue
        curr_joints_seq = curr_joints_seq[:,JOINTS_IDX,:,:]

        valid_sequence_idx = torch.sum(~torch.isnan(torch.flatten(curr_h_seq,start_dim=1)),dim=1) & \
                             torch.sum(~torch.isnan(torch.flatten(curr_v_seq, start_dim=1)), dim=1) & \
                             torch.sum(~torch.isnan(torch.flatten(curr_joints_seq, start_dim=1)), dim=1)
        curr_h_seq = curr_h_seq[valid_sequence_idx,:,:,:,:,:]
        curr_v_seq = curr_v_seq[valid_sequence_idx,:,:,:,:,:]
        curr_joints_seq = curr_joints_seq[valid_sequence_idx,:,:,:]

        # rescale range to (-1,1) from (0,1)
        curr_joints_seq[:, :, 0, :] = 2 * curr_joints_seq[:, :, 0, :] - 1

        h_seq = torch.cat((h_seq,curr_h_seq),axis=0)
        v_seq = torch.cat((v_seq,curr_v_seq),axis=0)
        joints_seq = torch.cat((joints_seq,curr_joints_seq),axis=0)

    return h_seq, v_seq, joints_seq

def train_n_validate():
    os.mkdir(MODEL_DIR)
    os.system('zip code_bkup.zip *.py')
    os.system(f'mv code_bkup.zip ./{MODEL_DIR}')
    with open(os.path.join(MODEL_DIR, 'parameters.json'), 'w') as f:
        f.write(json.dumps(PARAMS, default=str))

    if RESTORE_DIR:
        milli_transnet = torch.load(os.path.join(RESTORE_DIR,'milli_transnet_final.pth'))
        discriminator = torch.load(os.path.join(RESTORE_DIR,'discriminator_final.pth'))
    else:
        milli_transnet = MilliTransNet.MilliTransNet(NRANGES,NDOPPLER,LONG_CH_SIZE,SHORT_CH_SIZE,SEQ_LEN,
                                                     JOINTS_IDX,
                                                     SHAPE_CODE_SIZE, N_LAYERS, N_HEADS, N_DROPOUT, FWD_EXPANSION,
                                                     output_process=OUTPUT_PROCESSING,
                                                     initializer=MODEL_INIT,
                                                     optimizer=OPTIMIZER,
                                                     device=DEVICE)
        discriminator = SkeletalDiscriminator.Discriminator(JOINTS_IDX,
                                                            initializer=MODEL_INIT,
                                                            optimizer=OPTIMIZER)
    milli_transnet.set_device(DEVICE)
    discriminator.set_device(DEVICE)

    h_seq_train, v_seq_train, joints_seq_train = get_dataset(TRAINING_DIR)
    h_seq_valid, v_seq_valid, joints_seq_valid = get_dataset(VALIDATION_DIR)
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

    global curr_lr
    global curr_d_lr
    for n_epochs in range(N_EPOCHS):
        print(f'Epoch: {n_epochs}')
        begin_time = time.time()
        train_one_epoch(milli_transnet,discriminator,[h_seq_train,v_seq_train,joints_seq_train])
        validate_one_epoch(milli_transnet,discriminator,[h_seq_valid,v_seq_valid,joints_seq_valid])
        end_time = time.time()

        curr_lr = max(curr_lr * DECAY_RATE,1e-5)
        curr_d_lr = max(curr_d_lr * DECAY_RATE_D,1e-5)
        ALL_LR.append((curr_lr,curr_d_lr))
        print(f'Learning rates updated to: {curr_lr} and {curr_d_lr}')

        print(f'Training losses: Discriminator - {TRAINING_LOSS[-1][0]} \t GAN - {TRAINING_LOSS[-1][1]}')
        print(f'Validation losses: Discriminator - {VALIDATION_LOSS[-1][0]} \t GAN - {VALIDATION_LOSS[-1][1]}')
        print(f'Executed in {end_time-begin_time} seconds')
        if n_epochs % 10 == 0:
            print(f'Saving model to {MODEL_DIR}')
            torch.save(milli_transnet, os.path.join(MODEL_DIR, f'milli_transnet_ep{n_epochs}.pth'))
            torch.save(discriminator, os.path.join(MODEL_DIR, f'discriminator_ep{n_epochs}.pth'))
            np.savez(os.path.join(MODEL_DIR, 'loss.npz'),
                     TRAINING_LOSS=TRAINING_LOSS,
                     VALIDATION_LOSS=VALIDATION_LOSS)
            np.savez(os.path.join(MODEL_DIR, 'lr.npz'),
                     LEARNING_RATES=ALL_LR)

    print(f'Saving model to {MODEL_DIR}')
    torch.save(milli_transnet, os.path.join(MODEL_DIR, f'milli_transnet_final.pth'))
    torch.save(discriminator, os.path.join(MODEL_DIR, f'discriminator_final.pth'))
    np.savez(os.path.join(MODEL_DIR, 'loss.npz'),
             TRAINING_LOSS=TRAINING_LOSS,
             VALIDATION_LOSS=VALIDATION_LOSS)
    np.savez(os.path.join(MODEL_DIR, 'lr.npz'),
             LEARNING_RATES=ALL_LR)

def train_one_epoch(net,net_d,dataset):
    net.train()
    net_d.train()

    h_seq, v_seq, joints_seq = dataset
    n_datapoints = joints_seq.size(0)
    data_perm = torch.randperm(n_datapoints)
    h_seq = h_seq[data_perm, :, :, :, :, :]
    v_seq = v_seq[data_perm, :, :, :, :, :]
    joints_seq = joints_seq[data_perm, :, :, :]
    n_batches = n_datapoints // BATCH_SIZE

    d_loss_epoch = 0
    gan_loss_epoch = 0
    for b in tqdm(range(n_batches)):
        # get the batch data
        curr_h_seq = h_seq[b * BATCH_SIZE:min(n_datapoints, (b + 1) * BATCH_SIZE), :, :, :, :, :].to(
            DEVICE)
        curr_v_seq = v_seq[b * BATCH_SIZE:min(n_datapoints, (b + 1) * BATCH_SIZE), :, :, :, :, :].to(
            DEVICE)
        curr_joints_seq = torch.permute(joints_seq[b * BATCH_SIZE:min(n_datapoints, (b + 1) * BATCH_SIZE), :, :, :],
                                        (0, 3, 1, 2)).view(BATCH_SIZE, SEQ_LEN, -1).to(DEVICE)

        valid_indices = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.float32).to(
            DEVICE)  # currently, let's train for regularly sampled input

        # train the discriminator
        net.set_trainable(False)
        net_d.set_trainable(True)
        curr_pred_joints_seq_d = net(curr_h_seq,curr_v_seq,valid_indices)
        d_out_true = net_d(torch.reshape(curr_joints_seq,(BATCH_SIZE*SEQ_LEN,torch.sum(JOINTS_IDX)*3)))
        d_loss_true = net_d.get_loss(torch.ones(BATCH_SIZE*SEQ_LEN,1,dtype=torch.float32).to(DEVICE),d_out_true)
        d_out_pred = net_d(torch.reshape(curr_pred_joints_seq_d,(-1,torch.sum(JOINTS_IDX)*3)))
        d_loss_pred = net_d.get_loss(torch.zeros(BATCH_SIZE*SEQ_LEN,1,dtype=torch.float32).to(DEVICE),d_out_pred)
        net_d.optimize(0.5*torch.add(d_loss_true,d_loss_pred),lr=curr_d_lr)
        d_loss_epoch += 0.5*torch.add(d_loss_true,d_loss_pred)
        #print(f'Training discriminator, loss: {0.5*torch.add(d_loss_true,d_loss_pred)}')
        net_d.set_trainable(False)

        # train network
        net.set_trainable(True)
        curr_pred_joints_seq = net(curr_h_seq,curr_v_seq,valid_indices)
        net_loss = net.get_loss(curr_joints_seq,curr_pred_joints_seq)
        d_out_n = net_d(torch.reshape(curr_pred_joints_seq,(BATCH_SIZE*SEQ_LEN,torch.sum(JOINTS_IDX)*3)))
        d_loss_n = net_d.get_loss(torch.ones(BATCH_SIZE*SEQ_LEN,1,dtype=torch.float32).to(DEVICE),d_out_n)
        gan_loss = torch.add(net_loss,torch.mul(d_loss_n,D_LOSS_W))
        net.optimize(gan_loss,lr=curr_lr)
        gan_loss_epoch += gan_loss
        #print(f'Training network, losses: ({d_loss_n},{net_loss})')

    d_loss_epoch /= n_batches
    gan_loss_epoch /= n_batches

    TRAINING_LOSS.append((d_loss_epoch,gan_loss_epoch))

def validate_one_epoch(net,net_d,dataset):
    net.set_trainable(False)
    net_d.set_trainable(False)
    net.eval()
    net_d.eval()

    h_seq, v_seq, joints_seq = dataset
    n_datapoints = joints_seq.size(0)
    data_perm = torch.randperm(n_datapoints)
    h_seq = h_seq[data_perm, :, :, :, :, :]
    v_seq = v_seq[data_perm, :, :, :, :, :]
    joints_seq = joints_seq[data_perm, :, :, :]
    n_batches = n_datapoints // BATCH_SIZE

    d_loss_epoch = 0
    gan_loss_epoch = 0
    for b in tqdm(range(n_batches)):
        curr_h_seq = h_seq[b * BATCH_SIZE:min(n_datapoints, (b + 1) * BATCH_SIZE), :, :, :, :, :].to(
            torch.device(DEVICE))
        curr_v_seq = v_seq[b * BATCH_SIZE:min(n_datapoints, (b + 1) * BATCH_SIZE), :, :, :, :, :].to(
            torch.device(DEVICE))
        curr_joints_seq = torch.permute(joints_seq[b * BATCH_SIZE:min(n_datapoints, (b + 1) * BATCH_SIZE), :, :, :],
                                        (0, 3, 1, 2)).view(BATCH_SIZE, SEQ_LEN, -1).to(torch.device(DEVICE))

        valid_indices = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.float32).to(
            torch.device(DEVICE))  # currently, let's train for regularly sampled input

        # validate net
        curr_pred_joints_seq = net(curr_h_seq, curr_v_seq, valid_indices)
        net_loss = net.get_loss(curr_joints_seq, curr_pred_joints_seq)
        d_out_n = net_d(torch.reshape(curr_pred_joints_seq, (BATCH_SIZE * SEQ_LEN, torch.sum(JOINTS_IDX) * 3)))
        d_loss_n = net_d.get_loss(torch.ones(BATCH_SIZE * SEQ_LEN, 1).to(DEVICE), d_out_n)
        gan_loss = torch.add(net_loss, torch.mul(d_loss_n, D_LOSS_W))
        gan_loss_epoch += gan_loss
        #print(f'Validating network, losses: ({d_loss_n},{net_loss})')

        # validate discriminator
        d_out_true = net_d(torch.reshape(curr_joints_seq,(BATCH_SIZE*SEQ_LEN,torch.sum(JOINTS_IDX)*3)))
        d_loss_true = net_d.get_loss(torch.ones(BATCH_SIZE * SEQ_LEN, 1).to(DEVICE),d_out_true)
        d_out_pred = net_d(torch.reshape(curr_pred_joints_seq,(BATCH_SIZE*SEQ_LEN,torch.sum(JOINTS_IDX)*3)))
        d_loss_pred = net_d.get_loss(torch.zeros(BATCH_SIZE*SEQ_LEN,1).to(DEVICE),d_out_pred)
        d_loss_epoch += 0.5*torch.add(d_loss_true,d_loss_pred)
        #print(f'Validating discriminator, loss: {0.5*torch.add(d_loss_true,d_loss_pred)}')

    d_loss_epoch /= n_batches
    gan_loss_epoch /= n_batches

    VALIDATION_LOSS.append((d_loss_epoch,gan_loss_epoch))

if __name__=="__main__":
    train_n_validate()
