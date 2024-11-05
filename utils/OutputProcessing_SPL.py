import torch
import torch.nn as nn

class GetSkeletalOut(nn.Module):
    '''
        spine_base(0):      spine_mid(1), hip_left(12), hip_right(16)
        spine_mid(1):       spine_shoulder(4)
        neck(2):            head(3)
        head(3):

        shoulder_left(4):   elbow_left(5)
        elbow_left(5):      wrist_left(6)
        wrist_left(6):      hand_left(7)
        hand_left(7):       hand_tip_left(21), thumb_left(22)

        shoulder_right(8):  elbow_right(9)
        elbow_right(9):     wrist_right(10)
        wrist_right(10):    hand_right(11)
        hand_right(11):     hand_tip_right(23), thumb_right(24)

        hip_left(12):       knee_left(13)
        knee_left(13):      ankle_left(14)
        ankle_left(14):     foot_left(15)
        foot_left(15):

        hip_right(16):      knee_right(17)
        knee_right(17):     ankle_right(18)
        ankle_right(18):    foot_right(19)
        foot_right(19):

        spine_shoulder(20): shoulder_left(4), shoulder_right(8), neck(2)

        hand_tip_left(21):
        thumb_left(22):

        hand_tip_right(23):
        thumb_right(24):
    '''
    def __init__(self,
                 shape_code_size,
                 joints_idx,
                 hidden_dim=[64,16],
                 dout_per=0.3):
        super(GetSkeletalOut,self).__init__()
        self.__shape_code_size = shape_code_size
        self.__joints_idx = joints_idx
        self.__hidden_dim = hidden_dim
        self.__dout_per = dout_per
        self.__build_network()

    def __build_network(self):

        self.__down_coverter = nn.Sequential(
            *[
                nn.Linear(self.__shape_code_size,self.__hidden_dim[0]),
                nn.LeakyReLU(),
                nn.Dropout(self.__dout_per)
            ]
        )

        #----------------------------------------#

        # children of spine-base - hip-left, hip-right, spine-mid
        self.__spine_base_children = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0],self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1],9),
                nn.Tanh()
            ]
        )

        # child of hip-left & hip-right
        self.__knees = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0] + 6, self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1], 6),
                nn.Tanh()
            ]
        )

        # child of knee-left & knee-right
        self.__ankles = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0] + 6, self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1], 6),
                nn.Tanh()
            ]
        )

        # child of ankle_left & ankle-right
        self.__feet = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0]+6, self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1], 6),
                nn.Tanh()
            ]
        )

        # child of spine_mid
        self.__spine_shoulder = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0] + 3, self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1], 3),
                nn.Tanh()
            ]
        )

        # child of spine_shoulder: neck, shoulder-left, shoulder-right
        self.__spine_shoulder_children = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0] + 3, self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1], 9),
                nn.Tanh()
            ]
        )

        # child of neck
        self.__head = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0] + 3, self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1], 3),
                nn.Tanh()
            ]
        )

        # child of shoulder_left & shoulder-right
        self.__elbows = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0] + 6, self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1], 6),
                nn.Tanh()
            ]
        )

        # child of elbow_left & elbow-right
        self.__wrists = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0] + 6, self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1], 6),
                nn.Tanh()
            ]
        )

        # child of wrist-left & wrist-right
        self.__hands = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0] + 6, self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1], 6),
                nn.Tanh()
            ]
        )

        # child of hand-left & hand-right
        self.__hands_children = nn.Sequential(
            *[
                nn.Linear(self.__hidden_dim[0] + 6, self.__hidden_dim[1]),
                nn.BatchNorm1d(self.__hidden_dim[1]),
                nn.LeakyReLU(),

                nn.Linear(self.__hidden_dim[1], 12),
                nn.Tanh()
            ]
        )

        # ----------------------------------------#

    def forward(self, input):
        b, seqL, _ = input.size()
        input = torch.reshape(input, (-1,self.__shape_code_size))
        input = self.__down_coverter(input)

        c_spine_base = torch.zeros((b*seqL,3)).to(torch.device(input.device))

        c_spine_base_children = self.__spine_base_children(input)
        c_hip_left,c_hip_right,c_spine_mid = c_spine_base_children[:,:3], c_spine_base_children[:,3:6], c_spine_base_children[:,6:]

        c_knees = self.__knees(torch.cat((input,c_hip_left,c_hip_right),-1))
        c_knee_left,c_knee_right = c_knees[:,:3], c_knees[:,3:]

        c_ankles = self.__ankles(torch.cat((input, c_knee_left, c_knee_right), -1))
        c_ankle_left, c_ankle_right = c_ankles[:, :3], c_ankles[:, 3:]

        c_feet = self.__feet(torch.cat((input, c_ankle_left, c_ankle_right), -1))
        c_foot_left, c_foot_right = c_feet[:, :3], c_ankles[:, 3:]

        c_spine_shoulder = self.__spine_shoulder(torch.cat((input,c_spine_mid),-1))

        c_spine_shoulder_children = self.__spine_shoulder_children(torch.cat((input,c_spine_shoulder),-1))
        c_shoulder_left, c_shoulder_right, c_neck = c_spine_shoulder_children[:,:3], c_spine_shoulder_children[:,3:6], c_spine_shoulder_children[:,6:]

        c_head = self.__head(torch.cat((input,c_neck),-1))

        c_elbows = self.__elbows(torch.cat((input,c_shoulder_left,c_shoulder_right),-1))
        c_elbow_left, c_elbow_right = c_elbows[:,:3], c_elbows[:,3:]

        c_wrists = self.__wrists(torch.cat((input,c_elbow_left,c_elbow_right),-1))
        c_wrist_left, c_wrist_right = c_wrists[:,:3], c_wrists[:,3:]

        c_hands = self.__hands(torch.cat((input, c_wrist_left, c_wrist_right), -1))
        c_hand_left, c_hand_right = c_hands[:,:3], c_wrists[:,3:]

        c_hands_children = self.__hands_children(torch.cat((input,c_hand_left,c_hand_right),-1))
        c_thumb_left, c_thumb_right, c_hand_tip_left, c_hand_tip_right = \
            c_hands_children[:,:3], c_hands_children[:,3:6], c_hands_children[:,6:9], c_hands_children[:,9:]

        output = torch.cat((
            c_spine_base,
            c_spine_mid,
            c_neck,
            c_head,

            c_shoulder_left,
            c_elbow_left,
            c_wrist_left,
            c_hand_left,

            c_shoulder_right,
            c_elbow_right,
            c_wrist_right,
            c_hand_right,

            c_hip_left,
            c_knee_left,
            c_ankle_left,
            c_foot_left,

            c_hip_right,
            c_knee_right,
            c_ankle_right,
            c_foot_right,

            c_spine_shoulder,

            c_hand_tip_left,
            c_thumb_left,

            c_hand_tip_left,
            c_thumb_left

        ),
            -1)

        output = output[:,torch.repeat_interleave(self.__joints_idx,3)]
        output = torch.reshape(output, (b,-1,3*torch.sum(self.__joints_idx)))

        return output

if __name__=="__main__":
    skeletal_net = GetSkeletalOut(256,torch.ones(25).bool())
    shape_code = torch.rand(16,32,256)
    skeletal_out = skeletal_net(shape_code)
    print(skeletal_out.size())


