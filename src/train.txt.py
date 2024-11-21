import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from src import loss_func
from src import optimizer_Factory
from src import custom_plots as cp
from torch.autograd import Variable
import torch.optim as optim

from datetime import datetime

import wandb
import time


def train_epoch(model, loader,loss_fn, optimizer, device='cpu', print_loss = False):
    """
    Trains a neural network model for one epoch using the specified data loader and optimizer.
    Args:
    model (nn.Module): The neural network model to be trained.
    loader (DataLoader): The PyTorch Geometric DataLoader containing the training data.
    optimizer (torch.optim.Optimizer): The PyTorch optimizer used for training the model.
    device (str): The device used for training the model (default: 'cpu').
    Returns:
    float: The mean loss value over all the batches in the DataLoader.
    """
    model.to(device)
    model.train() # specifies that the model is in training mode
    running_loss = 0.
    total_loss = 0.
    
    #loss_fn = nn.MSELoss()
    for data in loader:

        input_Data, out_Data = data

        x0 = input_Data.to(device=device, dtype=torch.float)
        x1 = out_Data.to(device=device, dtype=torch.float)
        if False:
            max_values_dim1, _ = torch.max(x1, dim=1)
            max_values_dim2, _ = torch.max(max_values_dim1, dim=1)
            max_values_dim3, _ = torch.max(max_values_dim2, dim=1)
            max_values_dim3 = max_values_dim3.unsqueeze(1)

            print(max_values_dim3)


        # Zero gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(x0)
        # Compute the loss and its gradients
        loss  = loss_fn(x0 , outputs ,x1, print_loss )

        # l2_lambda = 0.001  # Regularization strength
        # l2_reg = torch.tensor(0.).to(device)
        # for param in model.parameters():
        #     l2_reg += torch.norm(param)

        # loss += l2_lambda * l2_reg
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    total_loss = running_loss/len(loader)
    return total_loss


#
def evaluate_epoch(model, loader,loss_fn, device='cpu'):
    with torch.no_grad():
        model.to(device)
        model.eval() # specifies that the model is in evaluation mode
        running_loss = 0.

        for data in loader:

            input_Data, out_Data = data
            
            x0 = input_Data.to(device=device, dtype=torch.float)
            x1 = out_Data.to(device=device, dtype=torch.float)
            
            outputs = model(x0)

            # Compute the loss             
            loss = loss_fn(x0 , outputs ,x1 )

            running_loss += loss.item()

        total_loss = running_loss/len(loader)

        if np.isnan(total_loss):
            print("Loss is NaN!")
            print(loss)
            print(len(loader))
            return 0
        
    return total_loss

def freeze(model, type): 
    for name, param in model.named_parameters():
        param.requires_grad = True
    if type == 'encoder':
        for name, param in model.named_parameters():
            if  'encoder' in name:  
                param.requires_grad = False
    if type == 'decoder':
        for name, param in model.named_parameters():
            if not 'decoder' in name: 
                param.requires_grad = False

    if type == 'encoder-decoder':
        for name, param in model.named_parameters():
            if name == 'pModel.alpha' or name == 'pModel.beta' : 
                param.requires_grad = False

    if type == 'pModel':
        for name, param in model.named_parameters():
                if name != 'pModel.k' and name != 'pModel.eq_distance' : 
                    param.requires_grad = False

    return model

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'best_model.pth.tar')

def train(model, train_loader, val_loader, type ='normal', init_phys = 1.0,loss_name=None):

    cfg = OmegaConf.load("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_epochs = cfg.train.epochs
    loss_fn = loss_func.getLoss(loss_name)
    #optimizer = optimizer_Factory.getOptimizer(model)    

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # if init_phys == 0:
    #     init_phys = 1.0
    
    # if init_phys < 0:

    #     init_phys = (init_phys*(-1))*0.1

    # elif init_phys > 1:
    #     init_phys = init_phys*0.5#np.ceil(np.log(init_phys)) * 0.5
    # else:
    #     init_phys = 2.0
    #init_phys = 1.0
    #init_phys = np.abs(init_phys)
    

    if hasattr(model, 'pModel') and hasattr(model, 'encoder'):
        optimizer = torch.optim.Adam([
                {'params': model.encoder.parameters(), 'name': 'encoder'},
                {'params':  model.pModel.parameters(), 'lr': init_phys,  'name': 'alpha' }#,'momentum': 0.9 ,  'name': 'alpha', 'alpha': 0.8},                     
            ], lr=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #'momentum': 0.8 ,
    # optimizer = torch.optim.Adam([
    #             {'params': model.encoder.parameters(), 'name': 'encoder'},
    #             {'params':  model.pModel.alpha, 'lr':init_phys , 'name': 'alpha'}, 
    #             {'params': model.pModel.beta, 'lr': init_phys, 'name': 'beta'}                       
    #         ], lr=1e-2)

    scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    model.to(device)
    

    now = datetime.now()

    dt_string = now.strftime("%d_%m_%y_%H")

    model_name = model.__class__.__name__

    if cfg.log_wandb:
        wandb.init(
                # set the wandb project where this run will be logged
                project="Vphysics-Project-IC",
                name = "exp_"+model_name+"_"+dt_string,
                
                # track hyperparameters and run metadata
                config={
                "learning_rate": cfg.optimize.lr,
                "architecture": model_name,
                "dataset": "NEURON",
                "epochs": cfg.train.epochs,
                }
            )

    log = []
    train_losses = []
    val_losses = []

    patience = 5 # patience for early stopping

    best_loss = float('inf')  # Initialize with a large value
    best_val_loss = float('inf')  # Initialize with a large value
    best_model_state = None

    # Model training
    try:
        train_loss = evaluate_epoch(model, train_loader, loss_fn, device=device)
    except:

        print("Loss is NaN! Epoch:", epoch)
        if cfg.log_wandb:
            wandb.finish()  
        model.load_state_dict(best_model_state)
        return model, log

    
    # Model validation
    val_loss = evaluate_epoch(model, val_loader, loss_fn, device=device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print("Initial Loss", "\t training loss:", train_loss,
                  "\t validation loss:",val_loss)

    dict_log = {"train_loss": train_loss, "validation_loss": val_loss}

    if hasattr(model, 'pModel'):
        for name, value in model.pModel.named_parameters():
            dict_log[name] = value[0].detach().cpu().numpy().item()

    if cfg.log_wandb:
        wandb.log(dict_log)
    log.append(dict_log)

    for epoch in range(1, num_epochs+1):       

        # if (epoch + 1) % (num_epochs//1) == 0:
        #     for param_group in optimizer.param_groups:
        #         if 'beta' in param_group['name'] or 'alpha' in param_group['name']:
        #             if param_group['lr'] > 0.05:
        #                 param_group['lr'] = param_group['lr']*0.1
                    
        # if (epoch + 1) == 10:
        #     for param_group in optimizer.param_groups:
        #         if 'beta' in param_group['name'] or 'alpha' in param_group['name']:
        #             #if param_group['lr'] > 0.05:
        #             param_group['lr'] = 0.5

        # Model training
        train_loss = train_epoch(model, train_loader, loss_fn,optimizer, device=device)
        #scheduler.step()
        # Model validation
        val_loss = evaluate_epoch(model, val_loader, loss_fn, device=device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)        

        dict_log = {"train_loss": train_loss, "validation_loss": val_loss}
        if hasattr(model, 'pModel'):
            for name, value in model.pModel.named_parameters():
                dict_log[name] = value[0].detach().cpu().numpy().item()
        log.append(dict_log)

        if cfg.log_wandb:
            wandb.log(dict_log)

        if np.isnan(train_loss) :            
            print("Loss is NaN! Epoch:", epoch)
            if cfg.log_wandb:
                wandb.finish()  
            model.load_state_dict(best_model_state)
            return model, log 
            
        
        
        if type == 'dynamic':
            if epoch % 2 == 0:
                model = freeze(model, 'encoder')
            if epoch % 2 == 1 or epoch > 3*num_epochs /4:
                model = freeze(model, 'encoder-decoder')

        # Early stopping
        try:
            if val_losses[-1]>=val_losses[-2]:
                early_stop += 1
            if early_stop == patience:
                print("Early stopping! Epoch:", epoch)
                if cfg.log_wandb:
                    wandb.finish()  
                break
            else:
                early_stop = 0
        except:
            early_stop = 0

        if  val_loss < best_val_loss:# and train_loss < best_loss:
            best_loss = train_loss
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

            torch.save(best_model_state, './best-model-parameters.pt')

            

            best_a = model.pModel.alpha[0].detach().cpu().numpy().item()
            best_b = model.pModel.beta[0].detach().cpu().numpy().item()
            

        if epoch%(num_epochs /10 )== 0 and cfg.log_wandb:
            print("epoch:",epoch, "\t training loss:", train_loss,
                  "\t validation loss:",val_loss)
            
 
    print("best model a",best_a)
    print("best last a", model.pModel.alpha[0].detach().cpu().numpy().item())

    print("best model b", best_b)
    print("best last b", model.pModel.beta[0].detach().cpu().numpy().item())

    model.load_state_dict(best_model_state)
   
    return model, log, [best_a, best_b]

def train_m(model, train_loader, val_loader, type ='normal', init_phys = 1.0,loss_name=None):

    cfg = OmegaConf.load("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_epochs = cfg.train.epochs
    loss_fn = loss_func.getLoss(loss_name)
    
    if init_phys == 0:
        init_phys = 1.0

    optimizer = torch.optim.Adam([
                {'params': model.encoder.parameters(), 'name': 'encoder'},
                {'params': model.renorm.parameters(), 'lr':0.1, 'name': 'renorm'},
                {'params': model.m, 'lr':0.1, 'name': 'm'},
                {'params': model.b, 'lr':0.1, 'name': 'b'},
                {'params':  model.pModel.k, 'lr': np.abs(init_phys), 'name': 'k'}, 
                {'params': model.pModel.eq_distance, 'lr': np.abs(0.05), 'name': 'eq_distance'}                       
            ], lr=1e-2)


    model.to(device)

    now = datetime.now()

    dt_string = now.strftime("%d_%m_%y_%H")

    model_name = model.__class__.__name__

    if cfg.log_wandb:
        wandb.init(
                # set the wandb project where this run will be logged
                project="Vphysics-Project-IC",
                name = "exp_"+model_name+"_"+dt_string,
                
                # track hyperparameters and run metadata
                config={
                "learning_rate": cfg.optimize.lr,
                "architecture": model_name,
                "dataset": "NEURON",
                "epochs": cfg.train.epochs,
                }
            )

    log = []
    train_losses = []
    val_losses = []

    patience = 5 # patience for early stopping

    best_loss = float('inf')  # Initialize with a large value
    best_val_loss = float('inf')  # Initialize with a large value
    best_model_state = None

    # Model training
    try:
        train_loss = evaluate_epoch(model, train_loader, loss_fn, device=device)
    except:

        print("Loss is NaN! Epoch:", epoch)
        if cfg.log_wandb:
            wandb.finish()  
        model.load_state_dict(best_model_state)
        return model, log


    # Model validation
    val_loss = evaluate_epoch(model, val_loader, loss_fn, device=device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print("Initial Loss", "\t training loss:", train_loss,
                  "\t validation loss:",val_loss)

    dict_log = {"train_loss": train_loss, "validation_loss": val_loss}

    if hasattr(model, 'pModel'):
        for name, value in model.pModel.named_parameters():
            if not ("encoder" in name):
                dict_log[name] = value[0].detach().cpu().numpy().item()

    if cfg.log_wandb:
        wandb.log(dict_log)
    log.append(dict_log)

    for epoch in range(1, num_epochs+1):
            
        if (epoch + 1) % 300 == 0:
            for param_group in optimizer.param_groups:
                if 'eq_distance' in param_group['name'] or 'k' in param_group['name']:
                    if param_group['lr'] > 0.005:
                        param_group['lr'] = param_group['lr']*0.1
                    if param_group['lr'] < 0.005:
                        param_group['lr'] = 0.005           
            

        
        if epoch % 100 == 0:
            print_loss = True
        else:
            print_loss = False
        # Model training
        #start_time = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn,optimizer, device=device, print_loss = print_loss)
        #end_time = time.time() 
        #print time in seconds
        #epoch_duration = end_time - start_time
        #print(f"Epoch {epoch + 1}/{num_epochs} took {epoch_duration:.2f} seconds")

        # Model validation
        val_loss = evaluate_epoch(model, val_loader, loss_fn, device=device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)        

        dict_log = {"train_loss": train_loss, "validation_loss": val_loss}
        if hasattr(model, 'pModel'):
            for name, value in model.pModel.named_parameters():
                dict_log[name] = value[0].detach().cpu().numpy().item()
        log.append(dict_log)

        if cfg.log_wandb:
            wandb.log(dict_log)

        if np.isnan(train_loss) :            
            print("Loss is NaN! Epoch:", epoch)
            if cfg.log_wandb:
                wandb.finish()  
            model.load_state_dict(best_model_state)
            return model, log 
            
        
        
        if type == 'dynamic':
            if epoch % 2 == 0:
                model = freeze(model, 'encoder')
            if epoch % 2 == 1 or epoch > 3*num_epochs /4:
                model = freeze(model, 'encoder-decoder')

        # Early stopping
        try:
            if val_losses[-1]>=val_losses[-2]:
                early_stop += 1
            if early_stop == patience:
                print("Early stopping! Epoch:", epoch)
                if cfg.log_wandb:
                    wandb.finish()  
                break
            else:
                early_stop = 0
        except:
            early_stop = 0

        if  val_loss < best_val_loss:
            best_loss = train_loss
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, './best-model-parameters.pt')

            best_a = model.pModel.k[0].detach().cpu().numpy().item()
            best_b = model.pModel.eq_distance[0].detach().cpu().numpy().item()
            

        if epoch%(num_epochs /10 )== 0 and cfg.log_wandb:
            print("epoch:",epoch, "\t training loss:", train_loss,
                  "\t validation loss:",val_loss)
            
        #if epoch == 2:
        #    return model, log
            
        

    if cfg.log_wandb:
        wandb.finish()

    

    print("best model a",best_a)
    print("best last a", model.pModel.k[0].detach().cpu().numpy().item())

    print("best model b", best_b)
    print("best last b", model.pModel.eq_distance[0].detach().cpu().numpy().item())

    model.load_state_dict(best_model_state)
   
    return model, log

def trainGAN(model, train_loader, val_loader, name, type ='normal'):

    cfg = OmegaConf.load("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_epochs = cfg.train.epochs
    loss_fn = loss_func.getLoss("adversarial_loss")
    #optimizer = optimizer_Factory.getOptimizer(model)    

    Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
    #create vectors for the training and validation loss
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=0.002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

    model.to(device)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%y_%H")
    model_name = model.__class__.__name__   

    wandb.init(
            # set the wandb project where this run will be logged
            project="Vphysics-Project",
            name = "exp_GAN_"+model_name+"_"+dt_string,
            config={
            "learning_rate": cfg.optimize.lr,
            "architecture": model_name,
            "dataset": "NEURON",
            "epochs": cfg.train.epochs,
            }
        )

    train_losses = []
    val_losses = []
    accuracy_list = []
    patience = 50 # patience for early stopping

    best_loss = float('inf')  # Initialize with a large value
    best_model_state = None

    for epoch in range(1, num_epochs+1):

        running_loss = 0.
        total_loss = 0.

        for data in train_loader:

            input_Data, out_Data = data

            z = input_Data.to(device=device, dtype=torch.float)
            real = out_Data.to(device=device, dtype=torch.float)

            optimizer_G.zero_grad()

            # Adversarial ground truths
            valid = Variable(Tensor(real.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(real.shape[0], 1).fill_(0.0), requires_grad=False)


            # Generate a batch of images
            gen_imgs = model.generator(z)
            # Loss measures generator's ability to fool the discriminator
            g_loss = loss_fn(model.discriminator(gen_imgs), valid)

        

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
           
            real_loss = loss_fn(model.discriminator(real), valid)
            fake_loss = loss_fn(model.discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            running_loss += d_loss.item()

        total_loss = running_loss/len(train_loader)
  
        train_losses.append(total_loss)

        
        wandb.log({"train_loss": total_loss})  
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_model_state = model.state_dict()

        if epoch%(num_epochs /10 )== 0:
            print("epoch:",epoch, "\t training loss:", total_loss)
    wandb.finish() 
        
    model.load_state_dict(best_model_state)
        #X = []
        #X.append( { 'x': range(1, num_epochs+1), 'y': train_losses, 'label': 'train_loss'} )
        #X.append({'x': range(1, num_epochs+1), 'y': val_losses, 'label': 'val_loss'} )
        #cp.plotMultiple( X,  'epoch', 'Loss','Performance', name, styleDark = True )
    return model, train_losses, val_losses, accuracy_list         

      
if __name__ == '__main__':
    train()