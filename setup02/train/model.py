import torch.nn as nn
import torch.nn.functional as F

#artifacts learning
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.features = nn.Sequential(
          nn.Conv1d(1,128,kernel_size=11,stride=1,padding=5),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Conv1d(128,64,kernel_size=9,stride=1,padding=4),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Conv1d(64,32,kernel_size=5,stride=1,padding=2),
          nn.BatchNorm1d(32),
          nn.ReLU(),
          nn.Conv1d(32,32,kernel_size=1,stride=1,padding=0),
          nn.BatchNorm1d(32),
          nn.ReLU(),
          nn.Conv1d(32,1,kernel_size=7,stride=1,padding=3),
        )
    def forward(self, x):
        x = self.features(x)
        return x
