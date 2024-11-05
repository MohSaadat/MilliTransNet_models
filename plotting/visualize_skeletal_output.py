import numpy as np
import os
import Kinect_Skeleton
import json
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join('saved_models','model_30_1_2024_7_37')
with open(os.path.join(RESULTS_DIR,'parameters.json'),'r') as f:
    PARAMS = json.load(f)
NJOINTS = int(PARAMS['NJOINTS'])
SEQ_LEN = int(PARAMS['SEQ_LEN'])
N_DATA_TO_PLOT = min(8,SEQ_LEN)
JOINTS_IGNORED_IDX = [int(i) for i in PARAMS['JOINTS_IGNORED_IDX']]
JOINTS_IDX = [(i not in JOINTS_IGNORED_IDX) for i in list(range(NJOINTS))]
JOINTS_MULTIPLIER = 1.5

results = np.load(os.path.join(RESULTS_DIR,'results.npz'))
true_data = results['TRUE_DATA']
pred_data = results['PRED_DATA']
data_size = true_data.shape[0]

data_idx_to_plot = np.random.randint(data_size,size=N_DATA_TO_PLOT)
indx_in_seq = np.random.randint(SEQ_LEN,size=1)
kinect_skeleton = Kinect_Skeleton.Kinect_Skeleton(JOINTS_IDX)

for idx in data_idx_to_plot:
    fig = plt.figure(figsize=plt.figaspect(0.5))
    curr_true_data = np.squeeze(true_data[idx,indx_in_seq,:,:])
    curr_pred_data = np.squeeze(pred_data[idx,indx_in_seq,:,:])

    curr_true_data = JOINTS_MULTIPLIER * curr_true_data
    curr_pred_data = JOINTS_MULTIPLIER * curr_pred_data

    kinect_skeleton.plot(curr_true_data,curr_pred_data,fig)


