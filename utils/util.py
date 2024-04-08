# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:05:22 2024

@author: LMK
"""

import torch
import os
import random
import numpy as np
from pathlib2 import Path
import shutil

def SetRandomSeed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def save_code(code_dir):
    this_dir = Path.cwd()
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", ".log", "*pycache*","*spyproject*","*pth","*pth*", \
        "*checkpoint*", "*data", "*result*", "*temp*","saved","*.pdf", "*.pptx"
    )
    shutil.copytree(this_dir, code_dir, ignore=ignore, dirs_exist_ok=True)   

def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + '/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/model_best.pth.tar')