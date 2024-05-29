import torch.nn as nn
import torch.nn.functional as F

#artifacts learning
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.features = nn.Sequential(
          nn.Conv2d(2,128,kernel_size=(11,1),stride=1,padding=(5,0)),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(128,64,kernel_size=(9,1),stride=1,padding=(4,0)),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64,32,kernel_size=(5,1),stride=1,padding=(2,0)),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Conv2d(32,2,kernel_size=(7,1),stride=(1,1),padding=(3,0)),
          nn.Tanh()
        )
    def forward(self, x):
        x = self.features(x)
        return x
