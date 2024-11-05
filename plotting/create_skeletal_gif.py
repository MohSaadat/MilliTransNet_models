import numpy as np
import os
import Kinect_Skeleton
import json
import imageio
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join('saved_models','model_30_1_2024_7_37')
with open(os.path.join(RESULTS_DIR,'parameters.json'),'r') as f:
    PARAMS = json.load(f)
NJOINTS = int(PARAMS['NJOINTS'])
SEQ_LEN = int(PARAMS['SEQ_LEN'])
JOINTS_IGNORED_IDX = [int(i) for i in PARAMS['JOINTS_IGNORED_IDX']]
JOINTS_IDX = [(i not in JOINTS_IGNORED_IDX) for i in list(range(NJOINTS))]
JOINTS_MULTIPLIER = 1.5
kinect_skeleton = Kinect_Skeleton.Kinect_Skeleton(JOINTS_IDX)
batch_indx = 0

results = np.load(os.path.join(RESULTS_DIR,'results.npz'))
true_data = results['TRUE_DATA'][batch_indx,:,:,:]
pred_data = results['PRED_DATA'][batch_indx,:,:,:]

fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

def update(idx):
    ax1.clear()
    ax2.clear()
    curr_true_data = np.zeros((NJOINTS,3))
    curr_pred_data = np.zeros((NJOINTS,3))
    curr_true_data[JOINTS_IDX,:] = true_data[idx,:,:]
    curr_pred_data[JOINTS_IDX,:] = pred_data[idx,:,:]
    for parent in kinect_skeleton.children:
        if not kinect_skeleton.joints_idx[parent]: continue

        for child in kinect_skeleton.children[parent]:
            if not kinect_skeleton.joints_idx[child]: continue

            ax1.plot3D(curr_true_data[[parent,child],0], curr_true_data[[parent,child],1],
                       curr_true_data[[parent,child],2], linewidth=3)
            ax2.plot3D(curr_pred_data[[parent,child],0], curr_pred_data[[parent,child],1],
                       curr_pred_data[[parent,child],2], linewidth=3)

filename = 'skeletal_out.gif'
with imageio.get_writer(filename,mode='I') as writer:
    for idx in range(SEQ_LEN):
        update(idx)
        plt.savefig('temp.png')
        image = imageio.imread('temp.png')
        writer.append_data(image)

os.remove('temp.png')
plt.show()
