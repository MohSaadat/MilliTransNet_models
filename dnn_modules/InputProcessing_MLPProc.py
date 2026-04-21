import torch
import torch.nn as nn
#import math
from dnn_modules.Utils import Conv2DBlock, LinearBlock

class _RD_processing(nn.Module):
    def __init__(self,
                 nRanges: int,
                 nDoppler: int,
                 groups: int,
                 hidden_channels=[2**2,2**2], # prev: [2**2,2**3,2**4]
                 outchannel=2**6, # prev: 2**8
                 lRelu_slope=0.05,
                 dout_per=0.3):
        super(_RD_processing, self).__init__()

        self.__nRanges = nRanges
        self.__nDoppler = nDoppler
        self.__groups = groups
        self.__hidden_channels = hidden_channels
        self.__outchannel = outchannel
        self.__lRelu_slope = lRelu_slope
        self.__dout_per = dout_per
        self.__build_network()

    def __build_network(self):
        self.__conv1 = Conv2DBlock(inChannel=1*self.__groups,
                                   outChannel=self.__hidden_channels[0]*self.__groups,
                                   groups=self.__groups,
                                   bias=True,
                                   kernel=(8,4),
                                   stride=(4,2),
                                   padding=(0,0),
                                   bn=False,
                                   dout=True,
                                   dout_per=self.__dout_per)
        self.__conv2 = Conv2DBlock(inChannel=self.__hidden_channels[0]*self.__groups,
                                   outChannel=self.__hidden_channels[1]*self.__groups,
                                   groups=self.__groups,
                                   bias=True,
                                   kernel=(4,4),
                                   stride=(2,2),
                                   padding=(0,0),
                                   bn=False,
                                   dout=True,
                                   dout_per=self.__dout_per)
        #self.__conv3 = Conv2DBlock(inChannel=self.__hidden_channels[1]*self.__groups,
        #                           outChannel=self.__hidden_channels[2]*self.__groups,
        #                           groups=self.__groups,
        #                           bias=True,
        #                           bn=False,
        #                           dout=True,dout_per=self.__dout_per)

        self.__flatten1 = nn.Flatten(start_dim=2)

        sp1, sp2 = self.__conv1.get_out_size([self.__nRanges,self.__nDoppler])
        self.__sp1, self.__sp2 = self.__conv2.get_out_size([sp1, sp2])
        #self.__sp1, self.__sp2 = self.__conv3.get_out_size([sp1, sp2])

        inChannel = self.__hidden_channels[1]*self.__sp1*self.__sp2
        self.__linear = LinearBlock(inChannel=inChannel,
                                     outChannel=self.__outchannel,
                                     dout=True,
                                     lRelu_slope=self.__lRelu_slope,
                                     dout_per=self.__dout_per)

    def get_outchannel(self):
        return self.__outchannel

    def forward(self, input):
        '''
            :param input:   4D: batch x nR x nD x nChL x nChH x seqL
            :return:        4D: batch x nChL*nChH*seqL x out_channel

            inp: b x nR x nD x chL x chH x seqL

            g (groups)  :   spatial1 * spatial2 * seqL
            the 2D conv will be executed independently on each member of this group
        '''
        #b, g, nR, nD = input.size() # b = _, g = groups, nR = nRanges, nD = nDoppler

        b, nR, nD, nChL, nChH, seqL = input.size()
        #input = torch.permute(input.view(b, nR, nD, nChL * nChH * seqL), (0, 3, 1, 2)) # b x g x nR x nD
        input = torch.permute(torch.reshape(input, (b, nR, nD, nChL * nChH * seqL)), (0, 3, 1, 2))
        g = nChL*nChH*seqL
        assert (g == self.__groups), "Group size is invalid"

        x = self.__conv1(input) # b x 4*g x nR' x nD'
        x = self.__conv2(x) # b x 16*g x nR'' x nD''
        #x = self.__conv3(x) # b x conv_out*g x nR''' x nD'''

        #x = torch.permute(x,(0,2,3,1)).view(b,self.__sp1,self.__sp2,g,self.__hidden_channels[1]) # b x nR''' x nD''' x g x conv_out
        x = torch.reshape(torch.permute(x, (0, 2, 3, 1)), (b, self.__sp1, self.__sp2, g, self.__hidden_channels[1]))
        x = torch.permute(x,(0,3,4,1,2)) # b x g x conv_out x nR''' x nD'''

        x = self.__flatten1(x) # b x g x (conv_out*nR'''*nD''')
        x = self.__linear(x)

        return x

class _CrossChannel(nn.Module):
    def __init__(self,
                 short_channel_size,
                 long_channel_size,
                 latent_hidden_channels=[2**5,2**5], # 2**4, 2**6, 2**4
                 self_hidden_channels=[2**5,2**5], # 2**4, 2**6, 2**4
                 outchannel=32,
                 lRelu_slope=0.05,
                 dout_per=0.3):
        super(_CrossChannel, self).__init__()
        self.__short_channel_size = short_channel_size
        self.__long_channel_size = long_channel_size
        self.__latent_hidden_channels = latent_hidden_channels
        self.__self_hidden_channels = self_hidden_channels
        self.__outchannel = outchannel
        self.__lRelu_slope = lRelu_slope
        self.__dout_per = dout_per
        self.__build_network()

    def __build_network(self):
        #latent_linear1 = LinearBlock(self.__long_channel_size,self.__self_hidden_channels[0],
        #                              dout=True,
        #                              lRelu_slope=self.__lRelu_slope)
        #latent_linear2 = LinearBlock(self.__self_hidden_channels[0], self.__self_hidden_channels[1],
        #                              dout=True,
        #                              lRelu_slope=self.__lRelu_slope)
        #latent_linear3 = LinearBlock(self.__self_hidden_channels[1], self.__outchannel,
        #                              dout=True,
        #                              lRelu_slope=self.__lRelu_slope,
        #                              dout_per=self.__dout_per)
        #self.__latent_network = nn.Sequential(*[latent_linear1,latent_linear2,latent_linear3])

        #self.__latent_network = LinearBlock(self.__long_channel_size,self.__outchannel)

        self.__latent_network = LinearBlock(self.__short_channel_size, self.__outchannel)

        #self_linear1 = LinearBlock(self.__long_channel_size, self.__self_hidden_channels[0],
        #                              dout=True,
        #                              lRelu_slope=self.__lRelu_slope)
        #self_linear2 = LinearBlock(self.__self_hidden_channels[0], self.__self_hidden_channels[1],
        #                              dout=True,
        #                              lRelu_slope=self.__lRelu_slope)
        #self_linear3 = LinearBlock(self.__self_hidden_channels[1], self.__outchannel,
        #                              dout=True,
        #                              lRelu_slope=self.__lRelu_slope,
        #                              dout_per=self.__dout_per)
        #self.__self_network = nn.Sequential(*[self_linear1,self_linear2,self_linear3])

        #self.__self_network = LinearBlock(self.__long_channel_size,self.__outchannel)

        self.__self_network = LinearBlock(self.__long_channel_size, self.__outchannel)

    def get_outchannel(self):
        return self.__outchannel

    def forward(self, input_h, input_v):
        '''
        :param input_h: batch x rd_processing.outchannel x seqL x nChL x nChH
        :param input_v: batch x rd_processing.outchannel x seqL x nChH x nChL
        :return:        batch x rd_processing.outchannel x seqL x nChH x nChH

        #-------------------------------------------------------------------#

        h_latent = self.__latent_network(input_h) # 1 x 512 x 32 x 2 x 8
        h_latent = torch.cat([input_h,h_latent],dim=3) # 1 x 512 x 32 x 4 x 8
        h_self = self.__self_network(h_latent) # 1 x 512 x 32 x 4 x 8
        h_expanded = torch.cat([h_latent,h_self],dim=3) # 1 x 512 x 32 x 8 x 8

        input_v_permuted = torch.permute(input_v,(0,1,2,4,3)) # 1 x 512 x 32 x 2 x 8

        v_latent_permuted = self.__latent_network(input_v_permuted)  # 1 x 512 x 32 x 2 x 8
        v_latent_permuted = torch.cat([input_v_permuted, v_latent_permuted], dim=3)  # 1 x 512 x 32 x 4 x 8
        v_self_permuted = self.__self_network(v_latent_permuted)  # 1 x 512 x 32 x 4 x 8
        v_expanded_permuted = torch.cat([v_latent_permuted,v_self_permuted], dim=3) # 1 x 512 x 32 x 8 x 8

        v_expanded = torch.permute(v_expanded_permuted,(0,1,2,4,3)) # 1 x 512 x 32 x 8 x 8

        #-------------------------------------------------------------------#

        h_v_expanded = torch.cat((input_h,v_latent), dim=-2) # [] x 4 x 8
        v_h_expanded = torch.cat((input_v,h_latent), dim=-2) # [] x 4 x 8

        h_self = self.__self_network(h_v_expanded) # [] x 4 x 8
        h_expanded = torch.cat((h_v_expanded,h_self), dim=-2) # [] x 8 x 8
        v_self = self.__self_network(v_h_expanded) # [] x 4 x 8
        v_expanded = torch.cat((v_h_expanded,v_self), dim=-2) # [] x 8 x 8

        v_expanded = v_expanded.transpose(-1,-2)

        return torch.mul(h_expanded,v_expanded)

        #-------------------------------------------------------------------#

        input_v = input_v.transpose(-1,-2)  # [] x 2 x 8

        h_latent = self.__latent_network(input_h) # [] x 2 x 16
        h_self = self.__self_network(input_h) # [] x 2 x 16
        v_latent = self.__latent_network(input_v) # [] x 2 x 16
        v_self = self.__self_network(input_v)  # [] x 2 x 16

        v_latent = v_latent.transpose(-1,-2) # [] x 64 x 2
        hv_latent = torch.matmul(v_latent,h_self)

        v_self = v_self.transpose(-1,-2) # [] x 64 x 2
        vh_latent = torch.matmul(v_self,h_latent)

        return torch.mul(hv_latent,vh_latent)

        '''

        h_latent = self.__latent_network(input_h.transpose(-1,-2)).transpose(-1,-2) # [] x -32-<v> x 16<h>
        h_self = self.__self_network(h_latent) # [] x -32-<v> x -32-<h>
        v_latent = self.__latent_network(input_v)  # [] x 16<v> x -32-<h>
        v_self = self.__self_network(v_latent.transpose(-1,-2)).transpose(-1,-2) # [] x -32-<v> x -32-<h>

        #combined_signal = torch.matmul(v_self,h_self) # [] x -32-<v> x -32-<h>
        combined_signal = torch.mul(v_self, h_self)  # [] x -32-<v> x -32-<h>

        return combined_signal

class _ReduceShapeCode(nn.Module):
    def __init__(self,
                 input_size,
                 shape_code_size,
                 hidden_dims=[2**9,2**8],
                 lRelu_slope=0.05,
                 dout_per=0.3):
        super(_ReduceShapeCode,self).__init__()

        self.__input_size = input_size
        self.__shape_code_size = shape_code_size
        self.__hidden_dims = hidden_dims
        self.__lRelu_slope = lRelu_slope
        self.__dout_per = dout_per
        self.__build_network()

    def __build_network(self):
        linear1 = LinearBlock(self.__input_size,self.__hidden_dims[0],dout=True,lRelu_slope=self.__lRelu_slope,dout_per=self.__dout_per)
        linear2 = LinearBlock(self.__hidden_dims[0],self.__hidden_dims[1], dout=True,lRelu_slope=self.__lRelu_slope,dout_per=self.__dout_per)
        linear3 = LinearBlock(self.__hidden_dims[1],self.__shape_code_size,dout=True,lRelu_slope=self.__lRelu_slope,dout_per=self.__dout_per)

        self.__network = nn.Sequential(*[linear1,linear2,linear3])

    def forward(self,combined_signal):
        '''
        :param combined_signal: b x seqL x input_size where input_size = rd_processing.outchannel * long_size * long_size
        :return: b x seqL x shape_code_size
        '''
        shape_code = self.__network(combined_signal)
        return shape_code

class CreateShapeCode(nn.Module):
    def __init__(self,
                 nRanges: int,
                 nDoppler: int,
                 long_channel_size: int,
                 short_channel_size: int,
                 seqL: int,
                 shape_code_size: int):
        super(CreateShapeCode, self).__init__()

        self.__nRanges = nRanges
        self.__nDoppler = nDoppler
        self.__long_channel_size = long_channel_size
        self.__short_channel_size = short_channel_size
        self.__seqL = seqL
        self.__shape_code_size = shape_code_size
        self.__build_network()

    def __build_network(self):
        groups = self.__long_channel_size * self.__short_channel_size * self.__seqL
        self.__rd_processing = _RD_processing(self.__nRanges,self.__nDoppler,groups,outchannel=2**6,lRelu_slope=0.05,dout_per=0.3)
        self.__cross_channel = _CrossChannel(self.__short_channel_size,self.__long_channel_size,outchannel=2**5,lRelu_slope=0.05,dout_per=0.3)
        input_size = self.__rd_processing.get_outchannel() * (self.__cross_channel.get_outchannel()**2)
        self.__reduce_shape_code = _ReduceShapeCode(input_size,self.__shape_code_size,lRelu_slope=0.05,dout_per=0.3)

    def __prepare_for_cross_channel(self,input):
        input = input.transpose(1,2)
        #input = input.view(-1,self.__rd_processing.get_outchannel(),self.__short_channel_size,self.__long_channel_size,self.__seqL)
        input = torch.reshape(input, (-1, self.__rd_processing.get_outchannel(), self.__short_channel_size,self.__long_channel_size, self.__seqL))
        input = torch.permute(input,(0,1,4,2,3))
        return input

    def __prepare_for_reduce_shape_code(self,input):
        input = torch.permute(input,(0,2,1,3,4))
        input = torch.reshape(input,(-1,self.__seqL,self.__rd_processing.get_outchannel()*self.__cross_channel.get_outchannel()*self.__cross_channel.get_outchannel()))
        return input

    def forward(self,input_h,input_v):
        '''
        :param input: b x nR x nD x nChL x nChH x seqL
        :return:
        '''
        # process range-doppler
        rd_out_h = self.__rd_processing(input_h) # b x groups x rd_processing.outchannel
        rd_out_v = self.__rd_processing(input_v) # b x groups x rd_processing.outchannel

        # cross-channel learning
        rd_out_h = self.__prepare_for_cross_channel(rd_out_h) # b x rd_processing.outchannel x seqL x short_length x long_length
        rd_out_v = self.__prepare_for_cross_channel(rd_out_v).transpose(3,4) # b x rd_processing.outchannel x seqL x long_length x short_length
        combined_signal = self.__cross_channel(rd_out_h,rd_out_v) # b x rd_processing.outchannel x seqL x cross_channel.outchannel x cross_channel.outchannel

        # generate shape code
        combined_signal = self.__prepare_for_reduce_shape_code(combined_signal) # b x seqL x rd_processing.outchannel*cross_channel.outchannel*cross_channel.outchannel
        shape_code = self.__reduce_shape_code(combined_signal) # b x seqL x reduce_shape_code.out_dim

        return shape_code

if __name__ == "__main__":
    b, nR, nD, short_size, long_size, seqL = 4, 104, 32, 2, 8, 32
    input_h = torch.rand((b, nR, nD, short_size, long_size, seqL), dtype=torch.float32)
    input_v = torch.rand((b, nR, nD, long_size, short_size, seqL), dtype=torch.float32)

    shape_code_learner = CreateShapeCode(nR,nD,long_size,short_size,seqL,shape_code_size=2**7)
    shape_code = shape_code_learner(input_h,input_v)
    print(shape_code.size())
