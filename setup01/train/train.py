import numpy as np
import time
from model import Net1
import torch.optim as optim
import os                    
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# import ParaSetting ??

# torch.cuda.synchronize()
starttime = time.time()

os.environ["CUDA_VISIBLE_DEVICES"]="0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

datapath  = '../../data/'
modelpath = './model/'

os.makedirs(modelpath, exist_ok=True)
# os.makedirs(savepath, exist_ok=True)

## Hyperparameters
epoch_num = 20 #iteration number 
Nx = 128
bs = 16  # batch size
num_workers = 8
lr = 0.0005
# lr_update = 1
weight_decay = 0.000


class prepareData_train(Dataset):
    def __init__(self, train_or_test):
       
       self.files = os.listdir(datapath+train_or_test)
       self.train_or_test= train_or_test

    def __len__(self):
       
       return len(self.files)

    def __getitem__(self, idx):

        data = torch.load(datapath+self.train_or_test+'/'+self.files[idx])
        return data['k-space'],  data['label']

    
trainset = prepareData_train('train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,shuffle=True, num_workers=num_workers)

validationset = prepareData_train('validation')
validationloader = DataLoader(validationset, batch_size=bs,shuffle=True, num_workers=num_workers)

model = Net1().to(device)
print(model)

criterion1 = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


loss_train_list = []
loss_validation_list = []

for epoch in range(epoch_num):   
    model.train()
    loss_batch = []
    for i, data in enumerate(trainloader, 0):
       
        inputs = data[0].reshape(-1,2,Nx,10).to(device)
        labels = data[1].reshape(-1,2,Nx,1).to(device)

        outs = model(inputs)
        
        loss = criterion1(outs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch.append(loss.item())  
        if (i)%20==0:
            print('epoch:%d - %d, loss:%.10f'%(epoch+1,i+1,loss.item()))
    
    loss_train_list.append(round(sum(loss_batch) / len(loss_batch),10))
    print('\nTrain Loss: ',loss_train_list)

    model.eval()     # evaluation
    
    loss_batch = []
    print('testing...')
    for i, data in enumerate(validationloader, 0):
        
        inputs = data[0].reshape(-1,2,Nx,10).to(device)
        labels = data[1].reshape(-1,2,Nx,1).to(device)
    
        with torch.no_grad():
            outs = model(inputs)
        loss = criterion1(outs, labels)
        loss_batch.append(loss.item())
        

    loss_validation_list.append(round(sum(loss_batch) / len(loss_batch),10))
    print('Test Loss: ', loss_validation_list, '\n')

    if (epoch+1) == epoch_num:
        torch.save(model, os.path.join(modelpath, 'epoch-%d.pth' % (epoch+1)))

   
    if (epoch+1) % 4 == 0:
        # lr = min(2e-5,lr*lr_update) 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# torch.cuda.synchronize()
endtime = time.time()
print('Finished Training. Training time elapsed %.2fs.' %(endtime-starttime))

