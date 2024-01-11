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
import librosa

#import inference
import argparse
import numpy.lib.recfunctions as rfn
import time
import torch
import datetime
import matplotlib.pyplot as plt
import pandas as pd

if __name__ != '__main__':
    sys.path.append('./censeModels/')
    from censeModels.model import *
    from censeModels.util import *
    from censeModels.data_loader import wav_to_npy_no_labels
    from censeModels.third_octave import ThirdOctaveTransform

else:
    from model import *
    from util import *
    from data_loader import wav_to_npy_no_labels
    from third_octave import ThirdOctaveTransform

class CenseDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data[0,:,:]
        self.len_data = data.shape[1]

    def __getitem__(self, idx):
        input_spec = torch.from_numpy(np.copy(self.data[idx]))
        return (input_spec)

    def __len__(self):
        return self.len_data

class FelixInference():
    def __init__(self, dataset=None):
        self.tho_tr = ThirdOctaveTransform(sr=32000, fLen=4096, hLen=4000)
        self.dataset = dataset

        if self.dataset == 'Grafic':
            # According to felix's code, it is supposed to be 101 but it's weird and it doesn't lead to good correlations. Also tried with 0, but not better.
            # Emppirically, +19dB seems ok.
            # self.db_offset = 101
            # self.db_offset = 0
            self.db_offset = 19
        elif self.dataset == 'Lorient1k':
            self.db_offset = 33.96
        elif self.dataset == 'CenseLorient':
            self.db_offset = 26
        else:
            self.db_offset = 0
        self.exp = "TVBCense_Fast0dB"

        self.settings = load_settings(Path('./censeModels/exp_settings/', self.exp+'.yaml'))
        modelName = get_model_name(self.settings)

        useCuda = torch.cuda.is_available() and not self.settings['training']['force_cpu']
        self.useCuda = useCuda
        if useCuda:
            print('Using CUDA.')
            dtype = torch.cuda.FloatTensor
            ltype = torch.cuda.LongTensor
        else:
            print('No CUDA available.')
            dtype = torch.FloatTensor
            ltype = torch.LongTensor

        # Model init.
        self.enc = VectorLatentEncoder(self.settings)
        self.dec = PresPredRNN(self.settings, dtype=dtype)
        if useCuda:
            self.enc = nn.DataParallel(self.enc).cuda()
            self.dec = nn.DataParallel(self.dec).cuda()

        # Pretrained state dict. loading
        self.enc.load_state_dict(load_latest_model_from('./censeModels/'+self.settings['model']['checkpoint_dir'], modelName+'_enc', useCuda=useCuda))
        self.dec.load_state_dict(load_latest_model_from('./censeModels/'+self.settings['model']['checkpoint_dir'], modelName+'_dec', useCuda=useCuda))
        self.dtype = dtype
        self.ltype = ltype

    def inference_from_scratch(self, file_name):
        wav_data, sr = librosa.load(file_name, sr=32000)
        #MT: test, to remove
        # wav_data = librosa.util.normalize(wav_data)
        spectral_data = self.tho_tr.wave_to_third_octave(wav_data, zeroPad=True)
        spectral_data = spectral_data.T

        #MT: normalization of third octaves
        #spectral_data = spectral_data - np.mean(spectral_data)

        #MT: added to have same third octaves as input as the ones used for training of félix's algorithm
        spectral_data = spectral_data + 94 - 26
        spectral_data = spectral_data + self.db_offset
        spectral_data = np.expand_dims(spectral_data, axis=0)

        presence, scores = inference(exp=self.exp, enc=self.enc, dec=self.dec, useCuda=self.useCuda, settings=self.settings, spectral_data=spectral_data, dtype=self.dtype, ltype=self.ltype, batch_size=480)

        print('XXXXXXXXXXXX CNN-TRAIN-SYNTH CLASSIFIER XXXXXXXXXXXX')
        print(file_name)
        to_plot = np.mean(scores, axis=0)
        print(f'TRAFFIC: {to_plot[0]}, VOICES: {to_plot[1]}, BIRDS: {to_plot[2]}')

        return(scores)
        # return(scores, np.mean(spectral_data))

#inspired from main function of inference.py
def inference(exp, enc, dec,  useCuda, settings, spectral_data, dtype, ltype, batch_size=480, verbose=False):

    # Load datasets
    dataSpec = spectral_data

    if verbose:
        print('Encoder: ', enc)
        print('Decoder: ', dec)
        print('Encoder parameter count: ', enc.module.parameter_count() if useCuda else enc.parameter_count())
        print('Decoder parameter count: ', dec.module.parameter_count() if useCuda else dec.parameter_count())
        print('Total parameter count: ', enc.module.parameter_count()+dec.module.parameter_count() if useCuda else enc.parameter_count()+dec.parameter_count())

    enc.eval()
    dec.eval()

    presence = np.empty((0,3))
    scores = np.empty((0,3))
    mydataset = CenseDataset(dataSpec)
    mydataloader = torch.utils.data.DataLoader(mydataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
    tqdm_it=tqdm(mydataloader, 
                desc='EVAL', disable=True)

    for x in tqdm_it:
        x = x.type(dtype)
        
        #pad 7 inputs because of the 7 missing outputs
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        if x.shape[-2] > 7:
            m = nn.ReflectionPad2d((0, 0, 7, 0))
        else:
            m = nn.ReplicationPad2d((0, 0, 7, 0))
        x = m(x)
        x = x.squeeze(0)
        x = x.squeeze(0)

        x = F.pad(x.unsqueeze(0).unsqueeze(0)+settings['data']['level_offset_db'], (0, 3))
        if useCuda:
            x = x.cuda()
        if 'Slow' in exp:
            encData = torch.zeros((x.size(0), x.size(2), 128)).type(dtype) # batch x seq_len x embedding_size
            for iSeq in range(x.size(2)):
                encData[:, iSeq, :] = enc(x[:, :, iSeq, :].squeeze(1))
            score = torch.sigmoid(dec(encData))
            #print(torch.sum(score))
        else:
            encData = torch.zeros((x.size(0), x.size(2)-7, 128)).type(dtype) # batch x seq_len x embedding_size
            for iSeq in range(x.size(2)-7):
                encData[:, iSeq, :] = enc(x[:, :, iSeq:iSeq+8, :].squeeze(1))
            score = torch.sigmoid(dec(encData))

        scores = np.concatenate((scores, np.array(score.squeeze(0).cpu().data)), axis=0)
        presence = np.concatenate((presence, np.array(score.squeeze(0).round().cpu().data)), axis=0)
    
    return(presence, scores)

# def detection(file_name):
#     tho_tr = ThirdOctaveTransform(sr=32000, fLen=4096, hLen=4000)
#     wav_data, sr = librosa.load(file_name, sr=32000)
#     spectral_data = tho_tr.wave_to_third_octave(wav_data, zeroPad=True)
#     spectral_data = spectral_data.T
#     spectral_data = np.expand_dims(spectral_data, axis=0)
#     # presence, scores = inference(exp="TVBCenseSensor_Fast", spectral_data=spectral_data, batch_size=480)
#     presence, scores = inference(exp="TVBCenseSensor_Fast", spectral_data=spectral_data, batch_size=480)
#     #print(scores)
#     return(scores)

