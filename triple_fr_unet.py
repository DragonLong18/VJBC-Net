import argparse
from bunch import Bunch
from ruamel.yaml import safe_load
import models
from utils.helpers import get_instance, seed_torch,get_instance1
import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class TRIPLE_FR_UNet(nn.Module):
    def __init__(self):
        super(TRIPLE_FR_UNet, self).__init__()
        with open("./config.yaml", encoding="utf-8") as file:
            CFG = Bunch(safe_load(file))
        self.vessel_model = get_instance1(models, 'model', CFG)
        self.bifurcation_model = get_instance1(models, 'model', CFG)
        self.cross_model = get_instance1(models, 'model', CFG)
        checkpoint = torch.load('./pretrained_weights/DRIVE/checkpoint-epoch40.pth')
        self.vessel_model.load_state_dict(checkpoint['state_dict'])
    
    def forward(self, x):  
        y_vessel,vessel_desc=self.vessel_model(x)
        if self.training:
            vessel = F.sigmoid(y_vessel)
            x = (1+vessel)*x
        y_bifurcation,bif_desc = self.bifurcation_model(x)
        y_cross, cross_desc = self.cross_model(x)
        return y_vessel,y_bifurcation,y_cross,vessel_desc,bif_desc,cross_desc
    