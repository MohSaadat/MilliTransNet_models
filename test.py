import os
import torch
import numpy as np
from scipy.io import loadmat
import json
from tqdm import tqdm
import argparse
import time
from Utils.DataUtils import get_dataset

INPUT_DIR = os.path.join('..','input_data')
TESTING_DIR = os.path.join(INPUT_DIR,'test')
DATA_PARAMS = loadmat(os.path.join(INPUT_DIR,'data_parameters.mat'))

def initialize_run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=7,help='GPU to use [default: 7]')
    parser.add_argument('--restore_dir', type=str, default=os.path.join('saved_models','model_modified_inp_proc24'),
                        help='Checkpoint restore directory [default: saved_models/model_modified_inp_proc24]')
    flags = parser.parse_args()

    with open(os.path.join(flags.restore_dir,'parameters.json'),'r') as f:
        params = json.load(f)

    joints_ignore_key = 'JOINTS_IGNORE_IDX' if 'JOINTS_IGNORE_IDX' in params else 'JOINTS_IGNORED_IDX'
    config = {
        'GPU': flags.gpu,
        'RESTORE_DIR': flags.restore_dir,
        'PARAMS': params,
        'NJOINTS': int(params['NJOINTS']),
        'JOINTS_IGNORE_IDX': [int(i) for i in params[joints_ignore_key]],
        'NRANGES': int(params['NRANGES']),
        'NDOPPLER': int(params['NDOPPLER']),
        'LONG_CH_SIZE': int(params['LONG_CH_SIZE']),
        'SHORT_CH_SIZE': int(params['SHORT_CH_SIZE']),
        'SEQ_LEN': int(params['SEQ_LEN']),
        'MAX_SIGNAL_VAL_H': float(params['MAX_SIGNAL_VAL_H']) if 'MAX_SIGNAL_VAL_H' in params else 1,
        'MAX_SIGNAL_VAL_V': float(params['MAX_SIGNAL_VAL_V']) if 'MAX_SIGNAL_VAL_V' in params else 1,
        'MAX_1D_VAL': float(params['MAX_1D_VAL']) if 'MAX_1D_VAL' in params else 1,
        'MAX_RANGE': float(DATA_PARAMS['maxRange']),
        'MAX_AZIM': float(DATA_PARAMS['maxAzim']),
        'MAX_ELEV': float(DATA_PARAMS['maxElev'])
    }
    config['JOINTS_IDX'] = torch.tensor([(i not in config['JOINTS_IGNORE_IDX']) for i in range(config['NJOINTS'])])
    config['DEVICE'] = torch.device(
        f'cuda:{config["GPU"]}'
        if torch.cuda.is_available() and config['GPU'] in range(torch.cuda.device_count())
        else 'cpu'
    )
    print(f'Running on {config["DEVICE"]}')
    return config

def descale_n_convert(joints, config):
    '''
        seqL x n_joints x 3
    '''
    joints[:,:,0] = 0.5 * (joints[:,:,0] +1)

    R = config['MAX_RANGE'] * joints[:,:,0]
    phi = torch.deg2rad(config['MAX_AZIM'] * joints[:,:,1])
    theta = torch.deg2rad(config['MAX_ELEV'] * joints[:,:,2])

    xz = torch.mul(R, torch.cos(theta))
    x = torch.mul(xz, torch.sin(phi)).unsqueeze(2)
    y = torch.mul(R, torch.sin(theta)).unsqueeze(2)
    z = torch.mul(xz, torch.cos(phi)).unsqueeze(2)

    joints_xyz = torch.cat((x, y, z), axis=2)

    return joints_xyz

def test(config):
    milli_transnet = torch.load(os.path.join(config['RESTORE_DIR'],'milli_transnet_final.pth'), map_location=config['DEVICE'])
    milli_transnet.set_trainable(False)
    milli_transnet.eval()

    print('Loading data...')
    h_seq, v_seq, joints_seq = get_dataset(TESTING_DIR, config)
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
    true_data = np.empty((0,config['SEQ_LEN'],torch.sum(config['JOINTS_IDX']),3),dtype=np.float32)
    pred_data = np.empty((0,config['SEQ_LEN'],torch.sum(config['JOINTS_IDX']),3), dtype=np.float32)
    err = []
    inf_time = []
    for data_id in tqdm(range(n_datapoints)):

        curr_h_seq = h_seq[data_id,:,:,:,:,:].unsqueeze(0).to(torch.device(config['DEVICE']))
        curr_v_seq = v_seq[data_id,:,:,:,:,:].unsqueeze(0).to(torch.device(config['DEVICE']))
        curr_joints_seq = torch.permute(joints_seq[data_id,:,:,:],(2,0,1)) # we are not computing loss, so do not transfer to gpu
        valid_indices = torch.ones((1,config['SEQ_LEN']),dtype=torch.float32).to(torch.device(config['DEVICE'])) # current models have all been trained with full sampling

        begin_time = time.time()
        curr_pred_joints_seq = milli_transnet(curr_h_seq,curr_v_seq,valid_indices).detach().cpu().squeeze(0).view(config['SEQ_LEN'],-1,3)
        end_time = time.time()
        inf_time.append(end_time-begin_time)

        curr_joints_seq = descale_n_convert(curr_joints_seq, config)
        curr_pred_joints_seq = descale_n_convert(curr_pred_joints_seq, config)

        curr_err = torch.sqrt(torch.sum((curr_pred_joints_seq-curr_joints_seq) ** 2, dim=-1)).mean()

        err.append(curr_err)
        true_data = np.concatenate((true_data,curr_joints_seq.unsqueeze(0).numpy()),axis=0)
        pred_data = np.concatenate((pred_data,curr_pred_joints_seq.unsqueeze(0).numpy()),axis=0)

        print(f'Mean error for this sequence: {curr_err}')
        print(f'Inference time for this sequence: {inf_time[-1]}')

    print(f'Errors for every batch: {err}')
    print(f'Mean error: {sum(err)/n_datapoints}')
    np.savez(os.path.join(config['RESTORE_DIR'],'results.npz'),
             TRUE_DATA=true_data,
             PRED_DATA=pred_data,
             ERROR=err,
             INF_TIME=inf_time)

def main():
    config = initialize_run()
    test(config)

if __name__=="__main__":
    main()
