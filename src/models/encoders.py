import torch
import torch.nn as nn



class EncoderMLP(nn.Module):
    def __init__(self, in_size = 50,in_chan = 1,  latent_dim=1, initw = True):
        super().__init__()

        self.l1 = nn.Linear(5600,1000*in_chan )
        self.l2 = nn.Linear(1000*in_chan,100*in_chan )
        self.l3 = nn.Linear(100*in_chan,latent_dim)

        self.relu= nn.ReLU()
        self.Softmax= nn.Softmax(dim=1)
        self.sigmoid= nn.Sigmoid()

        self.Tanh=nn.Tanh()

        if initw:

          #xavier_normal
          nn.init.kaiming_normal_(self.l1.weight)
          nn.init.kaiming_normal_(self.l2.weight)
          nn.init.kaiming_normal_(self.l3.weight)

    def forward(self, x):

    
      x = x.reshape(x.shape[0], -1)
      x = self.relu(self.l1(x))
      x = self.relu(self.l2(x))
      x = self.l3(x)

      return x
    
