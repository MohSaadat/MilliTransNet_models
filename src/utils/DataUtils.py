import glob
import os

import torch
from scipy.io import loadmat
from tqdm import tqdm


def get_dataset(path, config):
    data_files = glob.glob(os.path.join(path, 'data*.mat'))

    h_seq = torch.empty(
        (
            0,
            config['NRANGES'],
            config['NDOPPLER'],
            config['SHORT_CH_SIZE'],
            config['LONG_CH_SIZE'],
            config['SEQ_LEN']
        ),
        dtype=torch.float32
    )
    v_seq = torch.empty(
        (
            0,
            config['NRANGES'],
            config['NDOPPLER'],
            config['LONG_CH_SIZE'],
            config['SHORT_CH_SIZE'],
            config['SEQ_LEN']
        ),
        dtype=torch.float32
    )
    joints_seq = torch.empty(
        (0, torch.sum(config['JOINTS_IDX']), 3, config['SEQ_LEN']),
        dtype=torch.float32
    )
    for i in tqdm(range(len(data_files))):
        file = data_files[i]
        curr_data_file = loadmat(file)
        if not (
            'h_seq' in curr_data_file
            and 'v_seq' in curr_data_file
            and 'joints_seq' in curr_data_file
        ):
            continue

        curr_h_seq = torch.from_numpy(
            (1 / config['MAX_SIGNAL_VAL_H']) * curr_data_file['h_seq']
        ).to(torch.float32)
        curr_v_seq = torch.from_numpy(
            (1 / config['MAX_SIGNAL_VAL_V']) * curr_data_file['v_seq']
        ).to(torch.float32)
        curr_joints_seq = torch.from_numpy(
            (1 / config['MAX_1D_VAL']) * curr_data_file['joints_seq']
        ).to(torch.float32)

        valid_data = (
            list(curr_h_seq.size())[1:] == [
                config['NRANGES'],
                config['NDOPPLER'],
                config['SHORT_CH_SIZE'],
                config['LONG_CH_SIZE'],
                config['SEQ_LEN']
            ]
            and list(curr_v_seq.size())[1:] == [
                config['NRANGES'],
                config['NDOPPLER'],
                config['LONG_CH_SIZE'],
                config['SHORT_CH_SIZE'],
                config['SEQ_LEN']
            ]
            and list(curr_joints_seq.size())[1:] == [config['NJOINTS'], 3, config['SEQ_LEN']]
        )
        if not valid_data:
            continue

        curr_joints_seq = curr_joints_seq[:, config['JOINTS_IDX'], :, :]

        valid_sequence_idx = (
            torch.sum(~torch.isnan(torch.flatten(curr_h_seq, start_dim=1)), dim=1)
            & torch.sum(~torch.isnan(torch.flatten(curr_v_seq, start_dim=1)), dim=1)
            & torch.sum(~torch.isnan(torch.flatten(curr_joints_seq, start_dim=1)), dim=1)
        )
        curr_h_seq = curr_h_seq[valid_sequence_idx, :, :, :, :, :]
        curr_v_seq = curr_v_seq[valid_sequence_idx, :, :, :, :, :]
        curr_joints_seq = curr_joints_seq[valid_sequence_idx, :, :, :]

        # Rescale range to (-1, 1) from (0, 1).
        curr_joints_seq[:, :, 0, :] = 2 * curr_joints_seq[:, :, 0, :] - 1

        h_seq = torch.cat((h_seq, curr_h_seq), axis=0)
        v_seq = torch.cat((v_seq, curr_v_seq), axis=0)
        joints_seq = torch.cat((joints_seq, curr_joints_seq), axis=0)

    return h_seq, v_seq, joints_seq
