import torch
from omegaconf import OmegaConf



def getOptimizer(model):
    cfg = OmegaConf.load("config.yaml")

    if cfg.optimize.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimize.lr)
        return optimizer    
    if cfg.optimize.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimize.lr)
        return optimizer    


