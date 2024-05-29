# 2D for use of our 12/23 data with complex split, UTD normalization

import numpy as np
import time
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from model import Net1
import os


starttime = time.time()


device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

## Set Model Save Path
model_path = './model'
os.makedirs(model_path,exist_ok=True)


## Prepare Training Data
train_data_path = '../../data/12_2023/Train_data_grads_off/'

# shape: (# samples per rep, # reps, # channels (if EMI))
train_EMI = np.load(train_data_path+'EMI_coils_data.npy')
train_RX = np.load(train_data_path+'RX_coils_data.npy')

train_EMI = np.transpose(train_EMI, (1,0,2))

# extract + move real/imag components of signals into new channel
real, imag = train_EMI.real, train_EMI.imag
train_EMI = np.stack((real,imag),axis=1)
real, imag = train_RX.real, train_RX.imag
train_RX = np.stack((real,imag),axis=1)

# normalize and zero center data
div = np.expand_dims(np.max(abs(train_RX),axis=2),2)
train_RX /= div
sub = np.expand_dims(np.mean(train_RX,axis=2),2)
train_RX -= sub
div = np.expand_dims(np.max(abs(train_EMI),axis=2),3)
train_EMI /= div
sub = np.expand_dims(np.mean(train_EMI,axis=2),3)
train_EMI -= sub

train_emi, val_emi, train_rx, val_rx = train_test_split(train_EMI,train_RX,shuffle=True,test_size=0.2)

train_emi = torch.tensor(train_emi,dtype=torch.float) # shape: (8000, 1, 1251)
train_rx = torch.tensor(train_rx,dtype=torch.float) # shape: (8000, 1251)
val_emi = torch.tensor(val_emi,dtype=torch.float) # shape: (2000, 1, 1251)
val_rx = torch.tensor(val_rx,dtype=torch.float) # shape: (2000, 1251)


## Set Parameters
batch_size = 64
epochs = 5
lr = 0.01
num_workers = 8


## Load Datasets
train_ds = TensorDataset(train_emi,train_rx)
val_ds = TensorDataset(val_emi,val_rx)

train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)
val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)


model = Net1().to(device)
optimizer = optim.Adam(model.parameters(),lr=lr)
loss_fn = nn.MSELoss()


def train(dataloader,model,loss_fn):
    model.train()
    train_loss = 0
    num_batches = len(dataloader)
    for i, (emi,rx) in enumerate(train_dl):
        inputs = emi.to(device)
        labels = torch.unsqueeze(rx,3).to(device)
        
        outs = model(inputs)
        print(torch.max(outs),torch.min(outs),torch.mean(outs))
        loss = loss_fn(outs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return round(train_loss/num_batches,10)

def val(dataloader,model,loss_fn):
    model.eval()
    eval_loss = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for i, (emi,rx) in enumerate(dataloader):
            inputs = emi.to(device)
            labels = torch.unsqueeze(rx,3).to(device)

            outs = model(inputs)

            loss = loss_fn(outs,labels)
            eval_loss += loss.item()
    return round(eval_loss/num_batches,10)


train_loss = []
val_loss = []

## Training Loop
for e in range(epochs):
    tloss = train(train_dl,model,loss_fn)
    train_loss.append(tloss)

    vloss = val(val_dl,model,loss_fn)
    val_loss.append(vloss)

    print(f'Epoch: {e}, Train Loss: {tloss}, Val Loss: {vloss}')

    if e+1 == epochs:
        model_path = os.path.join(model_path,f'epoch-{e+1}.pth')
        torch.save(model,model_path)
        print(f'Model saved to {model_path}.')

endtime = time.time()
print(f'Training Complete! Training time elapsed {endtime-starttime}s\n')
