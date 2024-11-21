
import torch
import torch.nn as nn
from . import encoders
from . import PhysModels
    

class EndPhys(nn.Module):
    def __init__(self, in_size=50, latent_dim = 1, n_mask = 1, in_channels = 1,  dt = 0.2, pmodel = "Damped_oscillation", init_phys = None, initw = False):
        super().__init__()

        self.n_mask = n_mask
        self.in_channels = in_channels         

        self.encoder = encoders.EncoderMLP(in_size = in_size, in_chan=in_channels, latent_dim = latent_dim, initw = False)
              
        self.pModel = PhysModels.getModel(pmodel, init_phys)

        self.order = self.pModel.order
        self.dt = dt
        print("dt",dt)
    def forward(self, x):    
      
      order = self.order
      frames = x.clone()     

      for i in range(frames.shape[1]):     

          z_temp = self.encoder(frames[:,i,:,:,:])

          z_temp = z_temp.unsqueeze(1)
          z = z_temp if i == 0 else torch.cat((z,z_temp),dim=1)
    
     
      z2_phys = z[:,0:order,:]
      zroll = z[:,0:order,:]
      z2_encoder = z
      for i in range(frames.shape[1]-order):
          
          z_window = z2_phys[:,i:i+order,:]
          z_window2 = z[:,i:i+order,:]


          pred_window = self.pModel(z_window,self.dt)
          pred_window2 = self.pModel(z_window2,self.dt)
          
          z2_phys = torch.cat((z2_phys,pred_window),dim=1)
          zroll = torch.cat((zroll,pred_window2),dim=1)

      return  z2_encoder, z2_phys, zroll
    
    def get_masks(self):
        return self.masks
    

class EndPhysMultiple(nn.Module):
    def __init__(self, in_size=50, latent_dim = 1, n_mask = 1, in_channels = 1,  dt = 0.2, pmodel = "Damped_oscillation", init_phys = None, initw = False):
        super().__init__()

        self.n_mask = n_mask
        self.in_channels = in_channels 
        self.latent_dim =latent_dim 
        
        self.encoder = encoders.EncoderMLP(in_size = in_size, in_chan=1, latent_dim = latent_dim, initw = False)
        self.masks = None

        self.pModel = PhysModels.getModel(pmodel, init_phys)
        self.dt = dt
    def forward(self, x):    

      frames = x.clone()
      device = "cuda" if torch.cuda.is_available() else "cpu"  

      for i in range(frames.shape[1]):
          
          current_frame = frames[:,i,:,:,:]

          if self.latent_dim == 2:
            mask1 = current_frame[:,0:1,:,:]
            mask2 = current_frame[:,1:2,:,:]
            p1 = self.encoder(mask1)
            p2 = self.encoder(mask2)            

            z_temp = torch.cat((p1.unsqueeze(1),p2.unsqueeze(1)),dim=2)

          if self.latent_dim == 4:
             input = torch.sum(current_frame,1, keepdim=True)
             z_temp = self.encoder(input )
             z_temp = z_temp.unsqueeze(1)          
          
          z = z_temp if i == 0 else torch.cat((z,z_temp),dim=1)
                    
      z = z.squeeze(2)
      
      z_renorm = z
      z_renorm = z[:,0:2,:]
      z2_phys = z_renorm[:,0:2,:]
      
      for i in range(frames.shape[1]-2):      
          
          z_window = z2_phys[:,i:i+2,:]
          z_window2 = z[:,i:i+2,:]

          pred_window = self.pModel(z_window,self.dt)
          pred_window2 = self.pModel(z_window2,self.dt)
          
          z2_phys = torch.cat((z2_phys,pred_window),dim=1)
          z_renorm = torch.cat((z_renorm,pred_window2),dim=1)
      
      return  z, z2_phys, z_renorm
    
    def get_masks(self):
        return self.masks
    
