import torch
import torch.nn as nn
from transformer_modules import TransformerUtils

class _Encoder(nn.Module):
    def __init__(self,
                 shape_code_size,
                 nLayers,
                 nHeads,
                 n_dropout,
                 forward_expansion):
        super(_Encoder,self).__init__()

        self.__shape_code_size = shape_code_size
        self.__nLayers = nLayers
        self.__nHeads = nHeads
        self.__n_dropout = n_dropout
        self.__forward_expansion = forward_expansion

        assert (self.__shape_code_size % self.__nHeads == 0), "Shape code size and number of heads must be divisible"

        self.__build_network()

    def __build_network(self):
        self.__layers = nn.ModuleList(
            [
                TransformerUtils.TransformerBlock(
                    shape_code_size=self.__shape_code_size,
                    nHeads=self.__nHeads,
                    n_dropout=self.__n_dropout,
                    forward_expansion=self.__forward_expansion
                )
                for _ in range(self.__nLayers)
            ]
        )
        self.__dropout = nn.Dropout(self.__n_dropout)

    def forward(self,x,mask):
        '''
        :param x: position encoded shape code --> b x seqL x shape_code_size
        :return:
        '''
        _, seqL, shape_code_size = x.size()
        out = self.__dropout(x)

        for layer in self.__layers:
            out = layer(out,out,out,mask)

        return out

class Transformer(nn.Module):
    def __init__(self,
                 shape_code_size,
                 seqL,
                 nLayers,
                 nHeads,
                 n_dropout,
                 forward_expansion):
        super(Transformer, self).__init__()

        self.__shape_code_size = shape_code_size
        self.__seqL = seqL
        self.__nLayers = nLayers
        self.__nHeads = nHeads
        self.__n_dropout = n_dropout
        self.__forward_expansion = forward_expansion
        self.__build_network()

    def __build_network(self):
        self.__src_pos_encoder = TransformerUtils.PositionalEncoding(self.__shape_code_size,
                                                                     self.__seqL,
                                                                     start_token=False)
        self.__tgt_pos_encoder = TransformerUtils.PositionalEncoding(self.__shape_code_size,
                                                                     self.__seqL,
                                                                     start_token=True)

        self.__encoder = _Encoder(
            self.__shape_code_size,
            self.__nLayers,
            self.__nHeads,
            self.__n_dropout,
            self.__forward_expansion
        )

    def forward(self,src,src_valid_idx):
        #device = torch.device(src.device)
        b, _, _ = src.size()

        # positional encoding on src --> generate src mask --> execute encoder
        src = self.__src_pos_encoder(src, start_indx=0)
        src_mask = (src_valid_idx != 0).unsqueeze(1).unsqueeze(2)
        tgt_shape_code = self.__encoder(src, src_mask)

        return tgt_shape_code

if __name__=="__main__":
    b, seqL, shape_code_size = 16, 32, 128
    nLayers = 6
    nHeads = 64
    n_dropout = 0.3
    forward_expansion = 4
    device = 'cuda:0'

    tgt_seqL = seqL # for our specific case, src and tgt will have equal seq lengths

    # during training
    src_shape_code = torch.rand((b,seqL,shape_code_size),dtype=torch.float32).to(torch.device(device))
    tgt_shape_code = torch.rand((b,tgt_seqL,shape_code_size),dtype=torch.float32).to(torch.device(device))
    src_valid_idx = torch.randint(0, 2, (b, seqL)).to(torch.device(device))

    # build
    transformer = Transformer(shape_code_size, seqL, nLayers, nHeads, n_dropout, forward_expansion).to(torch.device(device))

    # execute
    tgt_shape_code_out = transformer(src_shape_code,src_valid_idx)

    print(src_shape_code.size())
    print(tgt_shape_code.size())
    print(tgt_shape_code_out.size())
