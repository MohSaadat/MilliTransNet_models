import os
import torch
import glob
import numpy as np
from scipy.io import loadmat
import json
from tqdm import tqdm
import argparse
import time

INPUT_DIR = os.path.join('..','input_data')
TESTING_DIR = os.path.join(INPUT_DIR,'test')
RESTORE_DIR = os.path.join('saved_models','model_modified_inp_proc24')

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu',type=int,default=7,help='GPU to use [default: 7]')
FLAGS = parser.parse_args()
GPU = FLAGS.gpu

# Pull up parameters
with open(os.path.join(RESTORE_DIR,'parameters.json'),'r') as f:
    PARAMS = json.load(f)

NJOINTS = int(PARAMS['NJOINTS'])
JOINTS_IGNORED_IDX = [int(i) for i in PARAMS['JOINTS_IGNORED_IDX']]
JOINTS_IDX = torch.tensor([(i not in JOINTS_IGNORED_IDX) for i in list(range(NJOINTS))])

NRANGES = int(PARAMS['NRANGES'])
NDOPPLER = int(PARAMS['NDOPPLER'])
LONG_CH_SIZE = int(PARAMS['LONG_CH_SIZE'])
SHORT_CH_SIZE = int(PARAMS['SHORT_CH_SIZE'])
SEQ_LEN = int(PARAMS['SEQ_LEN'])

MAX_SIGNAL_VAL_H = float(PARAMS['MAX_SIGNAL_VAL_H']) if 'MAX_SIGNAL_VAL_H' in PARAMS else 1
MAX_SIGNAL_VAL_V = float(PARAMS['MAX_SIGNAL_VAL_V']) if 'MAX_SIGNAL_VAL_V' in PARAMS else 1
MAX_1D_VAL = float(PARAMS['MAX_1D_VAL']) if 'MAX_1D_VAL' in PARAMS else 1

DEVICE = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() and GPU in range(torch.cuda.device_count()) else 'cpu')
print(f'Running on {DEVICE}')

def get_dataset(path):
    data_files = glob.glob(os.path.join(path,'data*.mat'))

    h_seq = torch.empty((0,NRANGES,NDOPPLER,SHORT_CH_SIZE,LONG_CH_SIZE,SEQ_LEN), dtype=torch.float32)
    v_seq = torch.empty((0,NRANGES,NDOPPLER,LONG_CH_SIZE,SHORT_CH_SIZE,SEQ_LEN), dtype=torch.float32)
    joints_seq = torch.empty((0,torch.sum(JOINTS_IDX),3,SEQ_LEN), dtype=torch.float32)
    for i in tqdm(range(len(data_files))):
        file = data_files[i]
        curr_data_file = loadmat(file)
        if not (
                'h_seq' in curr_data_file
                and 'v_seq' in curr_data_file
                and 'joints_seq' in curr_data_file
        ):
            continue
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
        curr_joints_seq[:,:,0,:] = 2 * curr_joints_seq[:,:,0,:] - 1

        h_seq = torch.cat((h_seq,curr_h_seq),axis=0)
        v_seq = torch.cat((v_seq,curr_v_seq),axis=0)
        joints_seq = torch.cat((joints_seq,curr_joints_seq),axis=0)

    return h_seq, v_seq, joints_seq

def descale_n_convert(joints):
    '''
        seqL x n_joints x 3
    '''
    joints[:,:,0] = 0.5 * (joints[:,:,0] +1)
    params = loadmat(os.path.join(INPUT_DIR,'data_parameters.mat'))
    maxR = float(params['maxRange'])
    maxAzim = float(params['maxAzim'])
    maxElev = float(params['maxElev'])

    R = maxR * joints[:,:,0]
    phi = torch.deg2rad(maxAzim * joints[:,:,1])
    theta = torch.deg2rad(maxElev * joints[:,:,2])

    xz = torch.mul(R, torch.cos(theta))
    x = torch.mul(xz, torch.sin(phi)).unsqueeze(2)
    y = torch.mul(R, torch.sin(theta)).unsqueeze(2)
    z = torch.mul(xz, torch.cos(phi)).unsqueeze(2)

    joints_xyz = torch.cat((x, y, z), axis=2)

    return joints_xyz

def test():
    milli_transnet = torch.load(os.path.join(RESTORE_DIR,'milli_transnet_final.pth'), map_location=DEVICE)
    milli_transnet.set_trainable(False)
    milli_transnet.eval()

    print('Loading data...')
    h_seq, v_seq, joints_seq = get_dataset(TESTING_DIR)
    n_datapoints = joints_seq.size(0)
    data_perm = torch.randperm(n_datapoints)
    h_seq = h_seq[data_perm,:,:,:,:,:]
    v_seq = v_seq[data_perm,:,:,:,:,:]
    joints_seq = joints_seq[data_perm,:,:,:]

    # ----------------------------------------------------#
    #h_seq_train, _, _ = get_dataset(os.path.join('input_data','training'))

    #test_data = torch.flatten(h_seq).numpy()
    #train_data = torch.flatten(h_seq_train).numpy()

    #import matplotlib.pyplot as plt
    #fig = plt.figure(figsize=plt.figaspect(0.5))

    #ax1 = fig.add_subplot(1, 2, 1)
    #ax1.hist(test_data,color='red',bins=100)

    #ax2 = fig.add_subplot(1, 2, 2)
    #ax2.hist(train_data,color='blue',bins=100)

    #plt.show()
    #exit()
    # ----------------------------------------------------#

    print('Inferring...')
    true_data = np.empty((0,SEQ_LEN,torch.sum(JOINTS_IDX),3),dtype=np.float32)
    pred_data = np.empty((0,SEQ_LEN,torch.sum(JOINTS_IDX),3), dtype=np.float32)
    err = []
    inf_time = []
    for data_id in tqdm(range(n_datapoints)):

        curr_h_seq = h_seq[data_id,:,:,:,:,:].unsqueeze(0).to(torch.device(DEVICE))
        curr_v_seq = v_seq[data_id,:,:,:,:,:].unsqueeze(0).to(torch.device(DEVICE))
        curr_joints_seq = torch.permute(joints_seq[data_id,:,:,:],(2,0,1)) # we are not computing loss, so do not transfer to gpu
        valid_indices = torch.ones((1,SEQ_LEN),dtype=torch.float32).to(torch.device(DEVICE)) # current models have all been trained with full sampling

        begin_time = time.time()
        curr_pred_joints_seq = milli_transnet(curr_h_seq,curr_v_seq,valid_indices).detach().cpu().squeeze(0).view(SEQ_LEN,-1,3)
        end_time = time.time()
        inf_time.append(end_time-begin_time)

        curr_joints_seq = descale_n_convert(curr_joints_seq)
        curr_pred_joints_seq = descale_n_convert(curr_pred_joints_seq)

        curr_err = torch.sqrt(torch.sum((curr_pred_joints_seq-curr_joints_seq) ** 2, dim=-1)).mean()

        err.append(curr_err)
        true_data = np.concatenate((true_data,curr_joints_seq.unsqueeze(0).numpy()),axis=0)
        pred_data = np.concatenate((pred_data,curr_pred_joints_seq.unsqueeze(0).numpy()),axis=0)

        print(f'Mean error for this sequence: {curr_err}')
        print(f'Inference time for this sequence: {inf_time[-1]}')

    print(f'Errors for every batch: {err}')
    print(f'Mean error: {sum(err)/n_datapoints}')
    np.savez(os.path.join(RESTORE_DIR,'results.npz'),
             TRUE_DATA=true_data,
             PRED_DATA=pred_data,
             ERROR=err,
             INF_TIME=inf_time)

if __name__=="__main__":
    test()
