import torch,math,random
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src import utils,models,train,data
import random

device = utils.get_device()
seed = 0
T_max = 20
utils.set_seed(seed=0,deterministic=True)

# My Y will show the patron [1,9] and assign tag 1 for ones which comply the propierty. 
X_train, Y_train = data.crear_dataset_patron_1_9(4000,T_max,seed=seed,device=device)
X_eval, Y_eval = data.crear_dataset_patron_1_9(1000,T_max,seed=seed,device=device)

print(device)
print(X_train)
acc = data.how_many_1_and_9_adyacentes(X_eval) / data.how_many_1_and_9(X_eval) 
print("ACC::::: ", acc)

epochs = 100
batch = 32
lr = 0.001
nheads = 8

dataset_train = TensorDataset(X_train,Y_train)
dataloader_train = DataLoader(dataset_train,shuffle=True,batch_size=batch)

dataset_eval = TensorDataset(X_eval,Y_eval)
dataloader_eval = DataLoader(dataset_eval,shuffle=True,batch_size=batch)

model = models.transformer_encoder(X_train,nheads,device)
opt = torch.optim.Adam(params=model.parameters(),lr=lr)
loss_fn = torch.nn.BCEWithLogitsLoss()

train.fit(model,device,dataloader_train,dataloader_eval,opt,loss_fn,epochs,early_stopping=20)
