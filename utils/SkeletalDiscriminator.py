import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,
                 joints_idx,
                 hidden_dims=[64,16],
                 dout_per=0.3,
                 initializer='kaiming_uniform',
                 optimizer='Adam'):
        super(Discriminator,self).__init__()
        self.__joints_idx = joints_idx
        self.__hidden_dims = hidden_dims
        self.__dout_per = dout_per
        self.__initializer = initializer
        self.__optimizer = optimizer

        self.__build_model()
        self.__initialize_model()

        self.__loss = nn.BCELoss()

    def __build_model(self):
        self.__network = nn.Sequential(
            *[
                nn.Linear(torch.sum(self.__joints_idx)*3,self.__hidden_dims[0]),
                nn.LeakyReLU(),
                nn.Dropout(self.__dout_per),

                nn.Linear(self.__hidden_dims[0],self.__hidden_dims[1]),
                nn.LeakyReLU(),
                nn.Dropout(self.__dout_per),

                nn.Linear(self.__hidden_dims[1],1),
                nn.Sigmoid()
            ]
        )

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
                    raise ValueError('Irrelevant initializer selected')

    def __get_all_trainables(self):
        trainable_list = []
        for p in self.parameters():
            if p.requires_grad:
                trainable_list.append(p)

        return trainable_list

    def set_device(self,device):
        self.to(torch.device(device))

    def set_trainable(self, is_training):
        for p in self.parameters():
            p.requires_grad = is_training

    def optimize(self, loss, lr=1e-3):
        if self.__optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.__get_all_trainables(), lr=lr)
        elif self.__optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.__get_all_trainables(), lr=lr)
        elif self.__optimizer == 'RMSProp':
            optimizer = torch.optim.RMSProp(self.__get_all_trainables(), lr=lr)
        else:
            raise ValueError('Irrelevant optimizer selected')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def get_loss(self,true_labels,pred_labels):
        return self.__loss(pred_labels,true_labels)

    def forward(self,sk):
        '''
            sk: ex_b x 21/17*3
        '''
        return self.__network(sk)

if __name__=="__main__":
    joints_idx = torch.tensor([(i not in [21, 23, 22, 24]) for i in list(range(25))])
    device = 'cuda:0'
    discriminator = Discriminator(joints_idx)
    discriminator.set_device(device)

    sk_in = torch.rand(32*32,torch.sum(joints_idx)*3).to(torch.device(device))
    labels = torch.randint(0,2,(32*32,1), dtype=torch.float32).to(torch.device(device))

    pred_labels = discriminator(sk_in)
    loss = discriminator.get_loss(labels,pred_labels)
    discriminator.optimize(loss)

    print(pred_labels.size())
