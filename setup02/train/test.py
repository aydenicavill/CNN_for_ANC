import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

## Set Model Save Path
model_path = './model/epoch-3.pth'
save_path = '../post/results/'


## Prepare Training Data
test_data_path = '../../data/12_2023/Test_data_grads_off/'

# shape: (# samples per rep, # reps, # channels (if EMI))
test_EMI = np.load(test_data_path+'EMI_coils_data.npy')
test_RX = np.load(test_data_path+'RX_coils_data.npy')

# normalize and zero center data
test_EMI = test_EMI - np.mean(test_EMI)
test_EMI = test_EMI / np.max(abs(test_EMI)) * 20
test_RX = test_RX - np.mean(test_RX)
test_RX = test_RX / np.max(abs(test_RX)) * 20

test_EMI = np.transpose(test_EMI, (1,2,0))

test_emi = torch.tensor(test_EMI,dtype=torch.float)
test_rx = torch.tensor(test_RX,dtype=torch.float)


# Set Parameters
batch_size = 256
num_workers = 8


# Load Datasets
test_ds = TensorDataset(test_emi,test_rx)
test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)


model = torch.load(model_path).to(device)
loss_fn = nn.MSELoss()

def test(dataloader,model,loss_fn):
    model.eval()
    test_loss = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for i, (emi,rx) in enumerate(dataloader):
            inputs = emi.to(device)
            labels = torch.unsqueeze(rx,1).to(device)

            outs = model(inputs)

            loss = loss_fn(outs,labels)
            test_loss += loss.item()
            print(f'batch {i} loss: {loss.item()}')
    return round(test_loss/num_batches,10)

loss = test(test_dl,model,loss_fn)
print(f'\nTest Loss: {loss}')


## Calculate R Ratio of First 10 Samples
percent_noise = 0.2 # percent of end of signal to use as noise

sig_len = test_RX.shape[1]
noise_i = int(np.ceil(sig_len*percent_noise))

inputs = test_emi[:10]
preds = model(inputs)
preds = np.squeeze(preds,axis=1).detach().numpy()

preds = preds[:,-noise_i:]
labels = test_RX[:10,-noise_i:]
print(np.max(preds),np.max(labels.real))

corrected = labels - preds
score = np.std(corrected,axis=0) / np.std(labels,axis=0)

print(f'Avg R Ratio for First 10 Samples: {np.mean(score)}\n')