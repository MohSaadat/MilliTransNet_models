import torch
import torch.nn as nn
import math
'''
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py
    https://www.youtube.com/watch?v=U0s0f995w14
    https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
    
    publicly used classes: TransformerBlock, DecoderBlock, PositionalEncoding
        
'''

class PositionalEncoding(nn.Module):
    '''
        https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
        shape_code_size ==> d_model in the original Transformer Architecture
        seqL ==> max_seq_len in the original Transformer Architecture
    '''
    def __init__(self,
                 shape_code_size,
                 seqL,
                 start_token=False):
        super(PositionalEncoding, self).__init__()
        self.__shape_code_size = shape_code_size
        self.__seqL = seqL
        self.__start_token = start_token

        pe = torch.zeros(self.__seqL, self.__shape_code_size)
        position = torch.arange(0, self.__seqL, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.__shape_code_size, 2).to(torch.float32) * -(math.log(10000.0) / self.__shape_code_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if self.__start_token:
            pe = torch.cat((
                torch.zeros(1,shape_code_size),
                pe
            ),
                           axis=0)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self,x,start_indx=0):
        return torch.add(x,self.pe[:, start_indx:start_indx+x.size(1), :])

class _MultiHeadAttention(nn.Module):
    '''
        https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
        shape_code_size ==> d_model in the original Transformer Architecture
    '''
    def __init__(self,
                 shape_code_size,
                 nHeads):
        super(_MultiHeadAttention, self).__init__()
        self.__shape_code_size = shape_code_size
        self.__nHeads = nHeads
        self.__d_k = shape_code_size // nHeads

        assert (self.__shape_code_size % self.__nHeads == 0), "shape_code_size must be divisible by nHeads"

        self.__build_network()

    def __build_network(self):
        self.__W_q = nn.Linear(self.__shape_code_size, self.__shape_code_size)
        self.__W_k = nn.Linear(self.__shape_code_size, self.__shape_code_size)
        self.__W_v = nn.Linear(self.__shape_code_size, self.__shape_code_size)
        self.__W_o = nn.Linear(self.__shape_code_size, self.__shape_code_size)

    def __scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.__d_k ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)

        return output

    def __split_heads(self, x):
        batch_size, seq_length, shape_code_size = x.size()
        return x.view(batch_size, seq_length, self.__nHeads, self.__d_k).transpose(1, 2)

    def __combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.__shape_code_size)

    def forward(self, Q, K, V, mask=None):
        Q = self.__split_heads(self.__W_q(Q))
        K = self.__split_heads(self.__W_k(K))
        V = self.__split_heads(self.__W_v(V))

        attn_output = self.__scaled_dot_product_attention(Q, K, V, mask)
        output = self.__W_o(self.__combine_heads(attn_output))
        return output

class _PositionWiseFeedForward(nn.Module):
    '''
            https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            shape_code_size ==> d_model in the original Transformer Architecture
            d_ff ==> same; 2048 in the paper
    '''
    def __init__(self,
                 shape_code_size,
                 d_ff):
        super(_PositionWiseFeedForward, self).__init__()
        self.__shape_code_size = shape_code_size
        self.__d_ff = d_ff
        self.__build_network()

    def __build_network(self):
        fc1 = nn.Linear(self.__shape_code_size, self.__d_ff)
        fc2 = nn.Linear(self.__d_ff, self.__shape_code_size)
        relu = nn.ReLU()
        self.__network = nn.Sequential(
            *[fc1,relu,fc2]
        )

    def forward(self, x):
        return self.__network(x)

class TransformerBlock(nn.Module):
    def __init__(self,
                 shape_code_size,
                 nHeads,
                 n_dropout,
                 forward_expansion):
        super(TransformerBlock, self).__init__()
        self.__shape_code_size = shape_code_size
        self.__nHeads = nHeads
        self.__n_dropout = n_dropout
        self.__forward_expansion = forward_expansion
        self.__build_network()

    def __build_network(self):
        self.__attention = _MultiHeadAttention(shape_code_size=self.__shape_code_size,nHeads=self.__nHeads)
        self.__norm1 = nn.LayerNorm(self.__shape_code_size)
        self.__norm2 = nn.LayerNorm(self.__shape_code_size)
        self.__ffn = _PositionWiseFeedForward(shape_code_size=self.__shape_code_size,d_ff=self.__shape_code_size*self.__forward_expansion)
        self.__dropout = nn.Dropout(self.__n_dropout)

    def forward(self, Q, K, V, mask):
        attention = self.__attention(Q, K, V, mask)

        x = self.__dropout(self.__norm1(attention+Q))
        forward = self.__ffn(x)
        out = self.__dropout(self.__norm2(forward+x))

        return out

#class DecoderBlock(nn.Module):
#    def __init__(self,
#                 shape_code_size,
#                 nHeads,
#                 n_dropout,
#                 forward_expansion):
#        super(DecoderBlock, self).__init__()
#        self.__shape_code_size = shape_code_size
#        self.__nHeads = nHeads
#        self.__n_dropout = n_dropout
#        self.__forward_expansion = forward_expansion
#        self.__build_network()

#    def __build_network(self):
#        self.__norm = nn.LayerNorm(self.__shape_code_size)
#        self.__attention = _MultiHeadAttention(shape_code_size=self.__shape_code_size,nHeads=self.__nHeads)
#        self.__transformer_block = TransformerBlock(shape_code_size=self.__shape_code_size,nHeads=self.__nHeads,n_dropout=self.__n_dropout,forward_expansion=self.__forward_expansion)
#        self.__dropout = nn.Dropout(self.__n_dropout)

#    def forward(self, x, V, K, src_mask, tgt_mask):
#        attention = self.__attention(x, x, x, tgt_mask)
#        Q = self.__dropout(self.__norm(attention + x))
#        out = self.__transformer_block(Q, K, V, src_mask)
#        return out

if __name__=="__main__":
    b, seqL, shape_code_size = 16, 32, 512
    nHeads = 8
    n_dropout = 0.2
    forward_expansion = 4

    shape_code = torch.rand((b,seqL,shape_code_size),dtype=torch.float32)
    start_token = torch.zeros((b,1,shape_code_size),dtype=torch.float32)
    shape_code = torch.cat((start_token,shape_code[:,:-1,:]),dim=1)
    shape_code = PositionalEncoding(shape_code_size,seqL,start_token=True)(shape_code)

    transformer_block = TransformerBlock(shape_code_size, nHeads, n_dropout, forward_expansion)
    #decoder_block = DecoderBlock(shape_code_size, nHeads, n_dropout, forward_expansion)

    src_mask = (torch.randint(0,2,(b,seqL)) != 0).unsqueeze(1).unsqueeze(2)
    #tgt_mask = torch.tril(torch.ones((seqL,seqL))).expand(
    #    b, 1, seqL, seqL
    #)

    transformer_block_out = transformer_block(shape_code,shape_code,shape_code, src_mask)
    #decoder_block_out = decoder_block(transformer_block_out,shape_code,shape_code, src_mask, tgt_mask)

    print(transformer_block_out.size())
    #print(decoder_block_out.size())