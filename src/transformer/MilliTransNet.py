import torch
import torch.nn as nn
import math
import TransformerModule
import InputProcessing_MLPProc as InputProcessing
#import InputProcessing_ConvProc as InputProcessing
#import TransformerUtils
import OutputProcessing_SPL
import OutputProcessing_Simple
from torch.nn.utils import clip_grad_norm_
import warnings

from plotting import Kinect_Skeleton as kskel

class MilliTransNet(nn.Module):
    def __init__(self,
                 nRanges: int,
                 nDoppler: int,
                 long_channel_size: int,
                 short_channel_size: int,
                 seqL: int,
                 joints_idx: list,
                 shape_code_size: int,
                 nLayers: int,
                 nHeads: int,
                 n_dropout: float,
                 forward_expansion: int,
                 output_process: str='spl',
                 initializer: str='kaiming_uniform',
                 optimizer: str='Adam'):
        super(MilliTransNet, self).__init__()

        self.__nRanges = nRanges
        self.__nDoppler = nDoppler
        self.__long_channel_size = long_channel_size
        self.__short_channel_size = short_channel_size

        self.__seqL = seqL
        self.__joints_idx = joints_idx
        self.__shape_code_size = shape_code_size
        self.__nLayers = nLayers
        self.__nHeads = nHeads
        self.__n_dropout = n_dropout
        self.__forward_expansion = forward_expansion

        self.__output_process = output_process
        self.__initializer = initializer
        self.__optimizer = optimizer

        self.__build_network()
        self.__initialize_model()
        self.__create_optimizer()

        self.__kskel = kskel.Kinect_Skeleton(self.__joints_idx)

    def __build_network(self):

        self.__input_processing = InputProcessing.CreateShapeCode(self.__nRanges,
                                                                  self.__nDoppler,
                                                                  self.__long_channel_size,
                                                                  self.__short_channel_size,
                                                                  self.__seqL,
                                                                  self.__shape_code_size)

        #self.__src_pos_encoder = TransformerUtils.PositionalEncoding(self.__shape_code_size,
        #                                                             self.__seqL,
        #                                                             start_token=False)
        #self.__tgt_pos_encoder = TransformerUtils.PositionalEncoding(self.__shape_code_size,
        #                                                             self.__seqL,
        #                                                             start_token=True)

        self.__transformer = TransformerModule.Transformer(self.__shape_code_size,
                                                           self.__seqL,
                                                           self.__nLayers,
                                                           self.__nHeads,
                                                           self.__n_dropout,
                                                           self.__forward_expansion)
        if self.__output_process == 'spl':
            self.__get_skeletal_out = OutputProcessing_SPL.GetSkeletalOut(self.__shape_code_size,
                                                                          self.__joints_idx)
        elif self.__output_process == 'simple':
            self.__get_skeletal_out = OutputProcessing_Simple.GetSkeletalOut(self.__shape_code_size,
                                                                             self.__joints_idx)
        else:
            raise ValueError('Invalid output processing scheme')

    def __initialize_model(self):
        for name, param in self.named_parameters():
            if 'weight' in name and \
                    (isinstance(param,nn.Conv2d) or \
                     isinstance(param,nn.Conv1d) or \
                     isinstance(param,nn.Linear)):
                if self.__initializer == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(param)
                elif self.__initializer == 'kaiming_normal':
                    nn.init.kaiming_normal_(param)
                elif self.__initializer == 'xavier_uniform':
                    nn.init.xavier_uniform_(param)
                elif self.__initializer == 'xavier_normal':
                    nn.init.xavier_normal_(param)
                else:
                    raise ValueError('Invalid initializer selected')

    def __get_all_trainables(self):
        trainable_list = []
        for p in self.parameters():
            if p.requires_grad:
                trainable_list.append(p)

        return trainable_list

    def __create_optimizer(self):
        if self.__optimizer == 'Adam':
            self.__optimizer_obj = torch.optim.Adam(self.__get_all_trainables(), lr=1e-3, weight_decay=1e-3)
        elif self.__optimizer == 'SGD':
            self.__optimizer_obj = torch.optim.SGD(self.__get_all_trainables(), lr=1e-3, weight_decay=1e-3)
        elif self.__optimizer == 'RMSProp':
            self.__optimizer_obj = torch.optim.RMSprop(self.__get_all_trainables(), lr=1e-3, weight_decay=1e-3)
        elif self.__optimizer == 'None':
            return None
        else:
            raise ValueError('Irrelevant optimizer selected')

    def get_optimizer(self): # public getter for optimizer object
        return self.__optimizer_obj

    def set_device(self,device):
        self.to(torch.device(device))

    def set_trainable(self, is_training):
        for p in self.parameters():
            p.requires_grad = is_training

    def optimize(self, loss, lr, weight_decay, max_norm):
        '''
        if self.__optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.__get_all_trainables(), lr=lr, weight_decay=1e-3)
        elif self.__optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.__get_all_trainables(), lr=lr, weight_decay=1e-3)
        elif self.__optimizer == 'RMSProp':
            optimizer = torch.optim.RMSProp(self.__get_all_trainables(), lr=lr, weight_decay=1e-3)
        else:
            raise ValueError('Irrelevant optimizer selected')
        '''
        if not self.__optimizer_obj or not self.__optimizer:
            warnings.warn('Returning without optimization; optimizer not created')
            return 

        for param_group in self.__optimizer_obj.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay

        self.__optimizer_obj.zero_grad()
        loss.backward()
        if max_norm != 0:
            clip_grad_norm_(self.parameters(), max_norm=max_norm)
        self.__optimizer_obj.step()

    def __sph2cart(self,joints):
        '''
            joints: b*seqL x n_joints x 3
        '''
        R = joints[:,:,0]
        phi = torch.deg2rad(joints[:,:,1])
        theta = torch.deg2rad(joints[:,:,2])

        xz = torch.mul(R,torch.cos(theta))
        x = torch.mul(xz,torch.sin(phi)).unsqueeze(2)
        y = torch.mul(R,torch.sin(theta)).unsqueeze(2)
        z = torch.mul(xz,torch.cos(phi)).unsqueeze(2)

        joints = torch.cat((x,y,z),axis=2)

        return joints

    def get_sph2cart(self,joints): # public wrapper for __sph2cart function
        return self.__sph2cart(joints)

    def get_loss(self,true_sk,pred_sk,loss_weights=[0.8,0.2]):
        '''
            b x seqL x [] --> b*seqL x n_joints x 3
        '''
        joints_idx = [i for i in range(len(self.__joints_idx)) if self.__joints_idx[i]]

        true_sk = torch.reshape(true_sk, (-1,self.__seqL,len(joints_idx),3))
        true_sk = torch.reshape(true_sk, (-1,len(joints_idx),3))
        true_sk = self.__sph2cart(true_sk)
        pred_sk = torch.reshape(pred_sk, (-1,self.__seqL,len(joints_idx),3))
        pred_sk = torch.reshape(pred_sk, (-1,len(joints_idx),3))
        pred_sk = self.__sph2cart(pred_sk)

        ext_batch = true_sk.size(0)

        true_sk_ = torch.zeros((ext_batch,len(self.__joints_idx),3), dtype=torch.float32).to(true_sk.device)
        true_sk_[:,joints_idx,:] = true_sk
        pred_sk_ = torch.zeros((ext_batch,len(self.__joints_idx),3), dtype=torch.float32).to(pred_sk.device)
        pred_sk_[:,joints_idx,:] = pred_sk

        # displacement loss
        #displacement_err  = \
        #    torch.sum((true_sk.view(-1,self.__seqL,torch.sum(self.__joints_idx),3) - pred_sk.view(-1,self.__seqL,torch.sum(self.__joints_idx),3)) ** 2, dim=-1).mean()
        #displacement_err = \
        #    torch.sum((true_sk-pred_sk)**2,dim=-1)[:,joints_idx].mean()
        displacement_err = \
            torch.sum((true_sk_ - pred_sk_) ** 2, dim=-1)[:,joints_idx].mean()

        # parent-child loss
        pc_loss = 0
        for parent in [value for value in self.__kskel.children if value in joints_idx]:
            for child in [value for value in self.__kskel.children[parent] if value in joints_idx]:
                pc_loss += torch.sum((true_sk_[:,parent,:]-pred_sk_[:,child,:])**2,dim=-1).mean()
        pc_loss /= len(joints_idx)

        #cosine_err = torch.mean(torch.mul(torch.acos(
        #    nn.CosineSimilarity(dim=3, eps=1e-6)(
        #        true_sk.view(-1, self.seqL, torch.sum(self.joints_idx), 3),
        #        pred_sk.view(-1, self.seqL, torch.sum(self.joints_idx), 3)
        #    )
        #),(1/math.pi)))

        #loss = torch.add(torch.mul(displacement_err,loss_weights[0]), torch.mul(cosine_err,loss_weights[1]))
        #loss = displacement_err
        loss = torch.add(torch.mul(displacement_err,loss_weights[0]),torch.mul(pc_loss,loss_weights[1]))

        return loss

    def forward(self,h_seq,v_seq,valid_indices):
        '''
            h_seq:          b x nR x nD x short_size x long_size x seqL
            v_seq:          b x nR x nD x long_size x short_size x seqL
            valid_indices:  b x seqL

        Code
        #-------------------------------------------------------------#

        b = h_seq.size(0)
        device = torch.device(h_seq.device)
        shape_code = self.__input_processing(h_seq,v_seq) # b x seqL x shape_code_size
        shape_code = self.__src_pos_encoder(shape_code,start_indx=0)

        curr_tgt_shape_code = torch.zeros((b,1,self.__shape_code_size),dtype=torch.float32).to(device)
        curr_tgt_valid_indices = torch.ones((b,1),dtype=torch.float32).to(device)
        for curr_out_idx in range(self.__seqL):

            curr_tgt_shape_code[:,-1,:] = self.__tgt_pos_encoder(curr_tgt_shape_code[:,-1,:].unsqueeze(1),start_indx=curr_out_idx).squeeze()
            curr_tgt_out = self.__transformer(shape_code,curr_tgt_shape_code,valid_indices,curr_tgt_valid_indices)

            curr_tgt_shape_code = torch.cat((curr_tgt_shape_code,curr_tgt_out[:,-1,:].unsqueeze(1)),axis=1)
            append_ind = torch.ones((b,1),dtype=torch.float32).to(device)
            curr_tgt_valid_indices = torch.cat((curr_tgt_valid_indices,append_ind),axis=-1)

        joints_seq_out = self.__get_skeletal_out(curr_tgt_shape_code[:,1:,:])

        return joints_seq_out
        '''
        shape_code = self.__input_processing(h_seq, v_seq)  # b x seqL x shape_code_size
        tgt_shape_code = self.__transformer(shape_code,valid_indices)
        joints_seq_out = self.__get_skeletal_out(tgt_shape_code)

        return joints_seq_out

if __name__=="__main__":
    b, nR, nD, short_size, long_size, seqL = 32, 104, 32, 2, 16, 48
    shape_code_size, nLayers, nHeads, n_dropout, forward_expansion = 128, 2, 16, 0.3, 4
    joints_idx = torch.tensor([(i not in [6,10,15,19,21,23,22,24]) for i in list(range(25))])
    device = 'cuda:0'

    h_seq = torch.rand((b,nR,nD,short_size,long_size,seqL),dtype=torch.float32).to(torch.device(device))
    v_seq = torch.rand((b,nR,nD,long_size,short_size,seqL),dtype=torch.float32).to(torch.device(device))
    gt_output = 2 * torch.rand((b,seqL,torch.sum(joints_idx)*3),dtype=torch.float32).to(torch.device(device)) -1
    milli_transnet = MilliTransNet(nR,
                                   nD,
                                   long_size,
                                   short_size,
                                   seqL,
                                   joints_idx,
                                   shape_code_size,
                                   nLayers,
                                   nHeads,
                                   n_dropout,
                                   forward_expansion,
                                   output_process='spl',
                                   optimizer='SGD')
    milli_transnet.set_device(device)

    valid_indices = torch.randint(0, 2, (b, seqL)).to(torch.device(device))
    h_seq[(valid_indices == 0).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, nR, nD, short_size,
                                                                                          long_size, 1)] = 0
    v_seq[(valid_indices == 0).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, nR, nD, long_size,
                                                                                          short_size, 1)] = 0

    milli_transnet.set_trainable(True)
    output = milli_transnet(h_seq, v_seq, valid_indices)
    loss = milli_transnet.get_loss(gt_output, output)
    milli_transnet.optimize(loss,lr=1e-3,weight_decay=1e-4)

    print(output.size())
    print(loss.size())

    #for name, params in milli_transnet.named_parameters():
    #    print(name)
    #for i, name_module in enumerate(milli_transnet.named_modules()):
    #    name, module = name_module
    #    if isinstance(module,nn.Tanh):
    #        print(i)
            #name, module = milli_transnet.named_modules()[i-1]
            #print(name)
            #print(module)
            #if 'get_skeletal_out' in name:
        #    print(module)
    #    print(params)
    #for p in milli_transnet.parameters():
    #    print(p)
    #for child in milli_transnet.children():
    #    print(child.register_forward_hook())
