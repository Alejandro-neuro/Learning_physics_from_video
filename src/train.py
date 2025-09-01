import torch
import numpy as np
from omegaconf import OmegaConf
from src import loss_func
from datetime import datetime
import wandb
import os


def train_epoch(model, loader,loss_fn, optimizer, device='cpu'):
    """
    Trains a neural network model for one epoch using the specified data loader and optimizer.
    Args:
        model (nn.Module): The neural network model to be trained.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer used for training the model.
        device (str): The device used for training the model (default: 'cpu').
    Returns:
        float: The mean loss value over all the batches in the DataLoader.
    
    Examples:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> train_epoch(model, train_loader, optimizer, device='cuda')
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

        # Zero gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(x0)
        # Compute the loss and its gradients
        loss  = loss_fn(x0 , outputs ,x1 )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    total_loss = running_loss/len(loader)
    return total_loss


#
def evaluate_epoch(model, loader,loss_fn, device='cpu'):
    '''
    Evaluates the model on the validation set.
    Args:
        
        model (nn.Module): The neural network model to be evaluated.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the validation data.
        device (str): The device used for evaluating the model (default: 'cpu').
    Returns:
        float: The mean loss value over all the batches in the DataLoader.
    '''
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'best_model.pth.tar')

def train(model, train_loader, val_loader, lr_phys = 1.0,loss_name=None, experiment_name = None):

    cfg = OmegaConf.load("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_epochs = cfg.train.epochs

    if "dropped_ball" in experiment_name:
        num_epochs = 1000

    loss_fn = loss_func.getLoss(loss_name)

    if lr_phys <= 0:
        lr_phys = 1.0
    


    if hasattr(model, 'pModel') and hasattr(model, 'encoder'):
        optimizer = torch.optim.Adam([
                {'params': model.encoder.parameters(), 'name': 'encoder'},
                {'params':  model.pModel.parameters(), 'lr': lr_phys,  'name': 'alpha' }#,'momentum': 0.9 ,  'name': 'alpha', 'alpha': 0.8},                     
            ], lr=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    
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
                "dataset": "Delfys75",
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
    except Exception as e:

        print(f"Loss is NaN! on the 1st evaluation, {e}")
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
        

        # Model training
        train_loss = train_epoch(model, train_loader, loss_fn,optimizer, device=device)
        
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
            #create folder if it does not exist
            if experiment_name is not None:
                if not os.path.exists('.Results/'+experiment_name):
                    os.makedirs('.Results/'+experiment_name)

            torch.save(best_model_state, './Results/'+experiment_name+'/best_model.pt')            

            best_a = model.pModel.alpha[0].detach().cpu().numpy().item()
            best_b = model.pModel.beta[0].detach().cpu().numpy().item()
           

        if epoch%(num_epochs /10 )== 0 and cfg.log_wandb:
            print("epoch:",epoch, "\t training loss:", train_loss,
                  "\t validation loss:",val_loss)
            
 
    print("best model a",best_a)
    print("best last a", model.pModel.alpha[0].detach().cpu().numpy().item())

    print("best model b", best_b)
    print("best last b", model.pModel.beta[0].detach().cpu().numpy().item())

       
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
        
        train_loss = train_epoch(model, train_loader, loss_fn,optimizer, device=device)
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

        if  train_loss < best_loss:
            best_loss = train_loss            
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, './best-train-parameters.pt')   

        if epoch%(num_epochs /10 )== 0 and cfg.log_wandb:
            print("epoch:",epoch, "\t training loss:", train_loss,
                  "\t validation loss:",val_loss)  

    if cfg.log_wandb:
        wandb.finish()

    

    print("best model a",best_a)
    print("best last a", model.pModel.k[0].detach().cpu().numpy().item())

    print("best model b", best_b)
    print("best last b", model.pModel.eq_distance[0].detach().cpu().numpy().item())

    #model.load_state_dict(best_model_state)
   
    return model, log

    
if __name__ == '__main__':
    train()
