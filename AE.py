import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
import numpy as np
from sklearn.preprocessing import StandardScaler
class AutoEncoder(nn.Module):
    
    def __init__(self,n_features,hidden_nodes,dropout=None,VAE=False):
        
        '''PARAMETERS
           n_features: number of X variables in dataset
           hidden_nodes: number of nodes in hidden layer
           dropout: fraction of nodes to dropout (0 < dropout <1)'''
        
        super(AutoEncoder, self).__init__()
        self.n_features=n_features
        self.n_hidden = hidden_nodes
        self.encoder = nn.Linear(n_features,hidden_nodes)
        self.decoder = nn.Linear(hidden_nodes,n_features)
        self.output_layer = nn.Linear(n_features,n_features)
        self.dropout = dropout
        self.best_recon = None
        
        
    def forward (self,x):
        if self.dropout!=None:
            x = F.relu(F.dropout(self.encoder(x),p=self.dropout))
        else:
            x = F.relu(self.encoder(x))
        self.hidden_layer=x
        x = F.relu(self.decoder(x))
        x = self.output_layer(x)
        return x

def train_autoencoder(model, dataset, loss_func, optimizer, epochs=100, batch_size=1024,  
                      validation_tensor=None,y_val=None, lr_rate_scheduler = None, noise_factor=None, 
                      random_seed=None, MSE_stopping_threshold=0):
        '''PARAMETERS
           model: instantiated autoencoder
           dataset: torch tensor of X variables
           loss_func: instantiated loss function
           optimizer: instantiated optimizer
           validation_tensor: torch tensor of validation X variables
           y_val: numpy array of validation y values
           epochs: number of epochs
           lr_rate_scheduler: instantiated PyTorch learning rate scheduler
           batch_size: batch size
           noise_factor: magnitude of noise added to data
             for a denoising autoencoder (0 < noise_factor <=1)
           random_seed: random_seed
           stopping_MSE_threshold: MSE value after which autoencoder stops training'''

        #Set up
        if random_seed!=None:
            torch.manual_seed(random_seed)
        train_loader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True)

        if type(validation_tensor)==torch.Tensor:
            val = True
            val_numpy = validation_tensor.detach().numpy()
        else:
            val = False

        readout_batch_interval = 0.25*(dataset.shape[0]/batch_size)//1
        sc = StandardScaler()
        
        #Training
        for epoch in range(epochs):
            counter = 0
            print('\n\033[1mEpoch {}\033[0m\n'.format(epoch+1))
            for batch in train_loader:

                if noise_factor!=None:
                    batch = batch + noise_factor * torch.randn(*batch.shape)
                batch = torch.autograd.Variable(batch)
                optimizer.zero_grad()
                recon = model(batch)
                loss = loss_func(recon, batch)
                if counter%readout_batch_interval==0:
                    print('Batch {} Loss: {:.4f}'.format(counter, float(loss)))
                loss.backward()
                optimizer.step()
                counter+=1
            
            #Readout for each epoch
            if epoch==0:
                epoch_loss = loss_func(model(dataset), dataset)
                print('\nEPOCH {} LOSS: {:.4f}'.format(epoch+1, float(epoch_loss)))
            else:
                old_epoch_loss = epoch_loss
                epoch_loss = loss_func(model(dataset), dataset)
                print('\nEPOCH {} LOSS: {:.4f}'.format(epoch+1, float(epoch_loss)))
                
            if val == True:
                val_output = model(validation_tensor).detach().numpy()
                reconstruction_error = np.sqrt(np.power(val_output - val_numpy, 2)).sum(axis=1)
                reconstruction_error = sc.fit_transform(reconstruction_error.reshape(-1, 1))
                sklogit = LogisticRegression()
                if epoch==0:
                    sklogit.fit(reconstruction_error,y_val)
                    preds = sklogit.predict(reconstruction_error)
                    score = recall_score(y_val,preds)
                    model.best_recon=model.parameters()
                    model.best_pr = score
                    print('\nReconstruction error recall: {:.4f}'.format(score))
                else:
                    old_score = score
                    sklogit.fit(reconstruction_error,y_val)
                    preds = sklogit.predict(reconstruction_error)
                    score = recall_score(y_val,preds)
                    if score<old_score:
                        model.best_recon=model.parameters()
                        model.best_recall = score
                    print('\nReconstruction error recall {:.4f}'.format(score))
                    print('Change: {:.4f}%'.format(float((score-old_score)/old_score)))
            if type(lr_rate_scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
                lr_rate_scheduler.step(score) 
            if epoch_loss<=MSE_stopping_threshold:
                break