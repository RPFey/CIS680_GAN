import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset
from scipy import linalg
from scipy.stats import entropy
import tqdm
import cv2

# FashionMNIST Dataset
train_dataset = datasets.FashionMNIST(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root='./dataset', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
train_batch_size = 100
test_batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

class VAE(torch.nn.Module):
  def __init__(self, zdim):
    super(VAE,self).__init__()
    ################################
    # Please fill in your code here:

    ################################
    self.fc1 = nn.Linear(784, 400)
    self.mu_layer = nn.Linear(400, zdim)
    self.var_layer = nn.Linear(400, zdim)

    self.act = nn.ReLU()

    self.decoder = nn.Sequential(
      nn.Linear(zdim, 400),
      nn.ReLU(),
      nn.Linear(400, 784),
      nn.Sigmoid()
    )
    
  def encode(self, X:torch.Tensor):
    ################################
    # Please fill in your code here:

    ################################
    x = self.act(self.fc1(X))
    mean = self.act(self.mu_layer(x))
    log_var = self.act(self.var_layer(x))
    return mean, log_var

  def decode(self, X):
    ################################
    # Please fill in your code here:

    ################################
    X = self.decoder(X)
    X = X.reshape(-1, 1, 28, 28)
    return X

  def reparameterization(self, mean, log_var):
    ################################
    # Please fill in std, eps and z:
    # std = 
    # eps = 
    # z = 
    ################################
    B, C = mean.shape
    eps = torch.rand((B, C), device = mean.device)
    z = mean + eps * torch.exp(log_var)

    return z

  def forward(self, X):
    X = X.view(-1,784)
    mean, log_var = self.encode(X)
    z = self.reparameterization(mean, log_var)
    return self.decode(z), mean, log_var



# reparameterization's output is dynamic, for the test case, we use a fixed eps 
# and all the intermediate result is provided. 
# You could use these values to check if you get the final output z correct.
# Or you could add eps to the input when testing the reparameterization module. 
# (warm reminder: don't forget to change back, cause the dynamic reparameterization is the key to VAE) 

# TEST YOUR REPARAMETRIZATION FUNCTION with the values below
testcase_mean = torch.load('test_case_VAE/mean.pt')
testcase_log_var = torch.load('test_case_VAE/log_var.pt')
# check std
testcase_std = torch.load('test_case_VAE/std.pt')
# Since epsilon is random, use the deterministic value of epsilon provided below
testcase_eps = torch.load('test_case_VAE/eps.pt')
testcase_z = torch.load('test_case_VAE/z.pt' )

# Reconstruction error module
def reconstruction_error(model:VAE, test_loader):
    '''
    Argms: 
    Input:
        model: VAE model
        test_loader: Fashion-MNIST test_loader
    Output:
        avg_err: MSE 
    '''
    # set model to eval
    ##################
    # TODO:
    ##################
    model.eval()
    
    # Initialize MSE Loss(use reduction='sum')
    ##################
    # TODO:
    ##################
    recon_err = 0.
    
    recon_err = 0
    idx_counter = 0
    for i, (data, _ ) in enumerate(test_loader):
        data = data.to(device)
        # feed forward data to VAE
        ##################
        # TODO:
        ##################
        recon, mean, log_var = model(data)
        
        idx_counter += data.shape[0] # sum up the number of images in test_loader

        # flatten the reconstruction output
        ##################
        # TODO:
        ##################
        recon_flatten = recon.flatten(1)
        data_flatten = data.flatten(1)
        
        # accumulate the MSELoss acrossing the whole test set
        ##################
        # TODO:
        ##################
        mse = F.mse_loss(recon_flatten, data_flatten, reduction="sum")
        recon_err += mse.item()
    
    avg_err = recon_err/idx_counter
    return avg_err

# Return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    '''
    Compute reconstruction loss and KL divergence loss mentioned in pdf handout
    '''
    ################################
    # Please compute BCE and KLD:
    
    ################################
    criterion = nn.BCELoss(reduction = 'sum')
    BCE = criterion(recon_x, x)

    var = torch.exp(log_var)
    kl = 0.5 * (var + mu ** 2 - 1 - log_var)
    KLD = torch.sum(kl)

    totalloss = BCE + KLD

    return totalloss


#####################################################
# TEST CASE FOR VAE LOSS
#####################################################
testcase_loss_recon_x = torch.load('test_case_VAE/loss_recon_x.pt')
testcase_loss_x = torch.load('test_case_VAE/loss_x.pt')
testcase_loss_mu = torch.load('test_case_VAE/loss_mu.pt')
testcase_loss_log_var = torch.load('test_case_VAE/loss_log_var.pt')
testcase_loss_totalloss = torch.load('test_case_VAE/loss_totalloss.pt')

loss = loss_function(testcase_loss_recon_x, testcase_loss_x, testcase_loss_mu, testcase_loss_log_var)
print("test case loss value:", testcase_loss_totalloss.item())
print("computed loss value:", loss.item())

# Z dimension
ZDIM = 5

#Initialize VAE
vae = VAE(ZDIM)
vae.to(device)
#Initialize optimizer
optimizer = optim.Adam(vae.parameters(), lr = 1e-3)
#Initialize scheduler(optional)
scheduler = StepLR(optimizer, step_size=10, gamma=0.2)
#num of epochs 
num_epochs = 10
import pdb
train_loss_list = []
orig_image_list = []
recon_image_list = []
reconst_error = []


# Define Train loop 
def train(epochs, train_loader, test_loader):

  for epoch in range(epochs):
      vae.train()
      train_loss = 0
      print('Epoch:', epoch,'LR:', scheduler.get_lr())
      for batch_idx, (data, _) in enumerate(train_loader):
          data = data.cuda()
          optimizer.zero_grad()
          recon_batch, mean, log_var = vae(data)

          loss = loss_function(recon_batch, data, mean, log_var)
          
          loss.backward()
          train_loss += loss.item()
          optimizer.step()

          if batch_idx % 100 == 0:
              recon_err = reconstruction_error(vae, test_loader)
              reconst_error.append(recon_err)
              print('Train Epoch: {} {:.0f}% \tLoss: {:.6f} \tRecon_err: {}'.format(epoch+1, 100. * batch_idx / len(train_loader), loss.item() / len(data), recon_err))
          del data; del recon_batch; del mean; del log_var    
          
      train_loss_list.append(train_loss / len(train_loader.dataset))
      print('Epoch: {} Train loss: {:.4f}'.format(epoch+1, train_loss / len(train_loader.dataset)))
      scheduler.step() 

    #   recon_err = reconstruction_error(vae, test_loader)
    #   print('Epoch: {} Reconstruction Error: {:.4f}'.format(epoch+1, recon_err))
      if epoch%5==0:
        with torch.no_grad():
          
            x_batch =torch.randn(10*10, ZDIM)
            recon_batch = vae.decode(x_batch.to(device))

        orig_image_list.append(vutils.make_grid(x_batch, nrow=10 ,padding=2, normalize=True))
        recon_image_list.append(vutils.make_grid(recon_batch.view(recon_batch.shape[0], 1 , 28, 28).detach().cpu(),nrow=10 , padding=2, normalize=True))

  # save the training checkpoint
  checkpoint = {'vae': vae.state_dict()}
  torch.save(checkpoint, '/content/vae_{}.pt'.format(epochs))
# Run Train loop
train(num_epochs, train_loader, test_loader)

# Plot Train loss
plt.title("VAE Train Loss")
plt.plot(train_loss_list,label="train loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

