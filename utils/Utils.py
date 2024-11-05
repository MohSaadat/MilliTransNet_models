#import torch
import torch.nn as nn
import math

class Conv2DBlock(nn.Module):
    def __init__(self,
                 inChannel: int,
                 outChannel: int,
                 groups: int,
                 bias=True,
                 kernel=(4,4),
                 stride=(2,2),
                 padding=(0,0),
                 bn=False,
                 dout=False,
                 lRelu_slope=0.05,
                 dout_per=0.3):
        super(Conv2DBlock, self).__init__()

        self.__inChannel = inChannel
        self.__outChannel = outChannel
        self.__bias = bias
        self.__kernel = kernel
        self.__stride = stride
        self.__padding = padding
        self.__groups = groups
        self.__bn = bn
        self.__dout = dout
        self.__lRelu_slope = lRelu_slope
        self.__dout_per = dout_per
        self.__build_network()

    def __build_network(self):
        conv = nn.Conv2d(self.__inChannel, self.__outChannel,
                         kernel_size=self.__kernel,
                         stride=self.__stride,
                         padding=self.__padding,
                         groups=self.__groups,
                         bias=self.__bias)
        activation = nn.LeakyReLU(self.__lRelu_slope)
        batchnorm = nn.BatchNorm2d(self.__outChannel)
        dropout = nn.Dropout(self.__dout_per)
        net_list = [conv]
        if self.__bn:
            net_list.append(batchnorm)
        net_list.append(activation)
        if self.__dout:
            net_list.append(dropout)
        self.__features = nn.Sequential(*net_list)

    def get_out_size(self,in_dims):
        n_rows, n_cols = in_dims
        n_rows_out = math.floor((n_rows - self.__kernel[0] + 2 * self.__padding[0]) / self.__stride[0]) + 1
        n_cols_out = math.floor((n_cols - self.__kernel[1] + 2 * self.__padding[1]) / self.__stride[1]) + 1
        return n_rows_out, n_cols_out

    def forward(self, input):
        '''
        :param input:   4D: batch x grouped_channels x spatial1 x spatial2
        :return:        4D: batch x grouped_channels x spatial1 x spatial2
        '''
        return self.__features(input)

class Conv3DBlock(nn.Module):
    def __init__(self,
                 inChannel: int,
                 outChannel: int,
                 groups: int,
                 bias=True,
                 kernel=(4,2,2),
                 stride=(2,2,2),
                 padding=(0,0,0),
                 bn=False,
                 dout=False,
                 lRelu_slope=0.05,
                 dout_per=0.3):
        super(Conv3DBlock, self).__init__()

        self.__inChannel = inChannel
        self.__outChannel = outChannel
        self.__bias = bias
        self.__kernel = kernel
        self.__stride = stride
        self.__padding = padding
        self.__groups = groups
        self.__bn = bn
        self.__dout = dout
        self.__lRelu_slope = lRelu_slope
        self.__dout_per = dout_per
        self.__build_network()

    def __build_network(self):
        conv = nn.Conv3d(self.__inChannel, self.__outChannel,
                         kernel_size=self.__kernel,
                         stride=self.__stride,
                         padding=self.__padding,
                         groups=self.__groups,
                         bias=self.__bias)
        activation = nn.LeakyReLU(self.__lRelu_slope)
        batchnorm = nn.BatchNorm3d(self.__outChannel)
        dropout = nn.Dropout(self.__dout_per)
        net_list = [conv]
        if self.__bn:
            net_list.append(batchnorm)
        net_list.append(activation)
        if self.__dout:
            net_list.append(dropout)
        self.__features = nn.Sequential(*net_list)

    def get_out_size(self,in_dims):
        n_depth, n_rows, n_cols = in_dims
        n_depth_out = math.floor((n_depth - self.__kernel[0] + 2 * self.__padding[0]) / self.__stride[0]) + 1
        n_rows_out = math.floor((n_rows - self.__kernel[1] + 2 * self.__padding[1]) / self.__stride[1]) + 1
        n_cols_out = math.floor((n_cols - self.__kernel[2] + 2 * self.__padding[2]) / self.__stride[2]) + 1
        return n_depth_out, n_rows_out, n_cols_out

    def forward(self, input):
        '''
        :param input:   4D: batch x grouped_channels x spatial1 x spatial2
        :return:        4D: batch x grouped_channels x spatial1 x spatial2
        '''
        return self.__features(input)

class LinearBlock(nn.Module):
    def __init__(self,
                 inChannel: int,
                 outChannel: int,
                 dout: bool=False,
                 lRelu_slope: float=0.05,
                 dout_per: float=0.3):
        super(LinearBlock,self).__init__()
        self.__inChannel = inChannel
        self.__outChannel = outChannel
        self.__dout = dout
        self.__lRelu_slope = lRelu_slope
        self.__dout_per = dout_per
        self.__build_network()

    def __build_network(self):
        linear = nn.Linear(self.__inChannel,self.__outChannel)
        activation = nn.LeakyReLU(self.__lRelu_slope)
        dropout = nn.Dropout(self.__dout_per)

        if self.__dout:
            self.__network = nn.Sequential(*[linear,activation,dropout])
        else:
            self.__network = nn.Sequential(*[linear,activation])

    def forward(self,input):
        return self.__network(input)