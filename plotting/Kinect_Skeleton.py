from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

class Kinect_Skeleton():
    def __init__(self,joints_idx):
        self.names = [
            'spine base',       # 0
            'spine mid',        # 1
            'neck',             # 2
            'head',             # 3

            'shoulder left',    # 4
            'elbow left',       # 5
            'wrist left',       # 6
            'hand left',        # 7

            'shoulder right',   # 8
            'elbow right',      # 9
            'wrist right',      # 10
            'hand right',       # 11

            'hip left',         # 12
            'knee left',        # 13
            'ankle left',       # 14
            'foot left',        # 15

            'hip right',        # 16
            'knee right',       # 17
            'ankle right',      # 18
            'foot right',       # 19

            'spine shoulder',   # 20

            'hand tip left',    # 21
            'thumb left',       # 22

            'hand tip right',   # 23
            'thumb right'       # 24

        ]
        self.children = {
            0: [1,12,16],
            1: [20],
            2: [3],
            3: [],

            4: [5],
            5: [6],
            6: [7],
            7: [21,22],

            8: [9],
            9: [10],
            10: [11],
            11: [23,24],

            12: [13],
            13: [14],
            14: [15],
            15: [],

            16: [17],
            17: [18],
            18: [19],
            19: [],

            20: [4,8,2],

            21: [],
            22: [],
            23: [],
            24: []
        }
        self.joints_idx = joints_idx

    def plot(self,joints1,joints2,fig):
        joints1_full = np.zeros((25,3))
        joints1_full[self.joints_idx,:] = joints1
        joints2_full = np.zeros((25, 3))
        joints2_full[self.joints_idx,:] = joints2

        #fig = plt.figure(figsize=plt.figaspect(0.5))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        for parent in self.children:
            if not self.joints_idx[parent]: continue

            for child in self.children[parent]:
                if not self.joints_idx[child]: continue

                ax1.plot3D(joints1_full[[parent,child],0], joints1_full[[parent,child],1], joints1_full[[parent,child],2], linewidth=3)

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        for parent in self.children:
            if not self.joints_idx[parent]: continue

            for child in self.children[parent]:
                if not self.joints_idx[child]: continue

                ax2.plot3D(joints2_full[[parent, child], 0], joints2_full[[parent, child], 1], joints2_full[[parent, child], 2], linewidth=3)

        plt.show()