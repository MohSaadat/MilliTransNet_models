import torch
import torch.nn as nn

class GetSkeletalOut(nn.Module):
    '''
        Simple skeletal reconstruction: we'll try SPL if this does not work
    '''
    def __init__(self,
                 shape_code_size,
                 joints_idx,
                 hidden_dim=128,
                 dout_per=0.3):
        super(GetSkeletalOut,self).__init__()
        self.__shape_code_size = shape_code_size
        self.__joints_idx = joints_idx
        self.__hidden_dim = hidden_dim
        self.__dout_per = dout_per
        self.__build_network()

    def __build_network(self):
        self.__network = nn.Sequential(
            *[
                nn.Linear(self.__shape_code_size,self.__hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.__dout_per),

                nn.Linear(self.__hidden_dim,torch.sum(self.__joints_idx)*3),
                nn.Tanh()
            ]
        )

    def forward(self, input):
        b = input.size(0)
        input = torch.reshape(input, (-1,self.__shape_code_size))
        out = self.__network(input).view(b,-1,torch.sum(self.__joints_idx)*3)
        return out

if __name__=="__main__":
    skeletal_net = GetSkeletalOut(256, torch.ones(25).bool())
    shape_code = torch.rand(16, 32, 256)
    skeletal_out = skeletal_net(shape_code)
    print(skeletal_out.size())
