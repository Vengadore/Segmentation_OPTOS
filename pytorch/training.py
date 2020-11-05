import torch
from tqdm import tqdm
import random

"""TO TRAIN"""
def train_data_gpu(model,optimizer,criterion,X,y,epochs = 10,batch_size = 1,shuffle=True,validation_data = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(epochs):
        if shuffle:
            pbar = tqdm(sorted(iter(X), key=lambda k: random.random()), ncols=100)
        else:
            pbar = tqdm(X,ncols=100)
        steps = X.shape[0]//batch_size
        i = 0
        running_loss = 0.0
        running_acc = 0.0
        for step in pbar:
            if i>steps:
                ""
            i+=batch_size
            X_batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            y_batch = torch.FloatTensor(y[i:i+batch_size]).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            running_acc += torch.sum(y_batch.gt(0.5) == outputs.gt(0.5)).cpu().detach().numpy();
            pbar.set_description("Accuracy: {:0.2f} Loss: {:0.2f} ".format(running_acc/i, running_loss/i))

        if validation_data != None:
            ""


