#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:12:06 2022

@author: user
"""

import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from pathlib import Path
from sklearn import preprocessing
import torch.nn.functional as F
from pathlib import Path

class ThirdOctaveInfo():
    def __init__(self, device=torch.device("cpu")):
        self.n_labels = 29




