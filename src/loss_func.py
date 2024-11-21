import torch
import torch.nn as nn
from omegaconf import OmegaConf
import numpy as np

def mse_loss(input_img, outputs, expected_pred):

    z2_encoder, z2_phys, zroll = outputs
    z2_encoder = z2_encoder.reshape(-1, z2_encoder.shape[-1])
    z2_phys = z2_phys.reshape(-1, z2_phys.shape[-1])  
    zroll = zroll.reshape(-1, zroll.shape[-1])
    
    loss_MSE = nn.MSELoss()

    loss = loss_MSE(z2_encoder, z2_phys)
    total_loss = loss 

    return total_loss

def latent_loss(input_img, outputs, expected_pred):

    z2_encoder, z2_phys, zroll = outputs   
    z2_encoder = z2_encoder.reshape(-1, z2_encoder.shape[-1])
    z2_phys = z2_phys.reshape(-1, z2_phys.shape[-1])  
    zroll = zroll.reshape(-1, zroll.shape[-1])
    
    loss_MSE = nn.MSELoss()

    loss = loss_MSE(z2_encoder, z2_phys)
     

    KLD_loss = KL_divergence( z2_encoder)
    
    #Uncomment to use the other KL divergence different from N(0,1)

    #KLD_loss = KL_divergence_2( z2_encoder)
    total_loss = loss + KLD_loss

    return total_loss

def KL_divergence(z):

    mu = z.mean(0)
    logvar = torch.log(z.var(0))

    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return KLD_loss

def KL_divergence_2(z, mu_2 = 1.0, var_2 = 0.5):

    mu = z.mean(0)
    logvar = torch.log(z.var(0))
    return 0.5 * torch.sum( ((mu-mu_2).pow(2))/var_2 + logvar.exp()/var_2 - 1 - logvar + np.log(var_2) )

def latent_loss_multiple(input_img, outputs, expected_pred):

    d = 2

    z2_encoder, z2_phys, z_renorm = outputs

    z2_encoder = z2_encoder.reshape(-1, z2_encoder.shape[2])
    z2_phys = z2_phys.reshape(-1, z2_phys.shape[2])  
    z_renorm = z_renorm.reshape(-1, z_renorm.shape[2])
    
   
    loss_MSE = nn.MSELoss()
    loss = loss_MSE(z2_phys, z2_encoder) 

    KLD_4d= KL_divergence( z2_encoder )

    KLD_loss = KLD_4d

    total_loss = d*loss + KLD_loss


    if torch.isnan(loss):
        return KLD_loss
    if torch.isnan(KLD_loss):
        return loss
     
    return total_loss

def latent_loss_multiple_Kld(input_img, outputs, expected_pred):

    z2_encoder, z2_phys = outputs

    z2_encoder = z2_encoder.reshape(-1, 2)
    z2_phys = z2_phys.reshape(-1, 2)

    mu = z2_encoder.mean(0)
    logvar = torch.log(z2_encoder.var(0))

    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 

    total_loss = KLD_loss
    return total_loss

def getLoss(loss = None):

    if loss == None:        
        cfg = OmegaConf.load("config.yaml")
        loss = cfg.loss    
    
    if loss == "MSE":
        return mse_loss
    if loss == "latent_loss":
        return latent_loss
    if loss == "latent_loss_multiple":
        return latent_loss_multiple    
    if loss == "latent_loss_multiple_Kld":
        return latent_loss_multiple_Kld

    pass

if __name__ == "__main__":
    getLoss()