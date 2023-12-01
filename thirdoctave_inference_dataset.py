#félix import
import os
import argparse
import torch
import torch.nn as nn
import sys
from tqdm import tqdm

#modan import
import numpy as np 
import h5py
import sys
import os
from pathlib import Path
#import inference
import argparse
import numpy.lib.recfunctions as rfn
import time
import torch
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import librosa

if __name__ != '__main__':
    sys.path.append('./felix/')
    from felix.third_octave import ThirdOctaveTransform

else:
    from third_octave import ThirdOctaveTransform

class ThirdOctaveInference():
    def __init__(self):
        self.n_labels = 29

    def inference_from_scratch(self, file_name):
        tho_tr = ThirdOctaveTransform(sr=32000, fLen=4096, hLen=4000)
        wav_data, sr = librosa.load(file_name, sr=32000)
        spectral_data = tho_tr.wave_to_third_octave(wav_data, zeroPad=True)
        spectral_data = spectral_data.T
        return(spectral_data)
    
def detection(file_name):
    tho_tr = ThirdOctaveTransform(sr=32000, fLen=4096, hLen=4000)
    wav_data, sr = librosa.load(file_name, sr=32000)
    spectral_data = tho_tr.wave_to_third_octave(wav_data, zeroPad=True)
    spectral_data = spectral_data.T
    return(spectral_data)
