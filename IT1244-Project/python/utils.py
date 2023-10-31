import sys
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# Dataloader
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import pandas as pd

# Models
from models import CNN

def get_train_dataframe(directory):
    files = sorted(glob.glob(directory+'/*'))
    df = pd.DataFrame(columns=['idx', 'audio_file', 'classification'])
    for idx, audio_file in enumerate(files):
        audio_path = audio_file.split('/')[-1]
        classification = audio_path.split('_')[0]
        row = pd.DataFrame([{"idx":idx, "audio_file": audio_file, "classification": 1 if classification=='dog' else 0}])
        df = pd.concat([df, row])

    return df

def get_test_dataframe(directory):
    files = sorted(glob.glob(directory+'/*'))
    df = pd.DataFrame(columns=['idx', 'audio_file', 'classification'])
    for idx, audio_file in enumerate(files):
        audio_path = audio_file.split('/')[-1]
        classification = audio_path.split('_')[0]
        row = pd.DataFrame([{"idx":idx, "audio_file": audio_file, "classification": 1 if classification=='dog' else 0}])
        df = pd.concat([df, row])

    return df

def get_hyperparameters(path):
    try:
        with open(path, 'r') as file:
            hp= {}
            exec(file.read(), hp)

            hp_dict = {}
            hp_dict['learning_rate'] = hp.get('lr')
            hp_dict['target_length'] = hp.get('target_length')
            hp_dict['n_fft'] = hp.get('n_fft')
            hp_dict['hop_length'] = hp.get('hop')
            hp_dict['epochs'] = hp.get('epochs')
            hp_dict['batch_size'] = hp.get('batch_size')
            hp_dict['kfold_splits'] = hp.get('n_splits')

            return hp_dict
    except:
        print("Error reading hyperparameters")

class CatsAndDogsDataset(Dataset):
    def __init__(self, target_length, n_fft, hop, dataframe):
        self.df = dataframe
        self.n_fft = n_fft
        self.hop = hop
        self.target_length = target_length
 
    def __getitem__(self, index):
        audio_path = self.df.iloc[index]['audio_file']
        signal, sr = torchaudio.load(audio_path)
        modified_signal = self._process_signal(signal)
        classification = self.df.iloc[index]['classification']
        return modified_signal, classification
    
    def __len__(self):
        return len(self.df)
    
    def _process_signal(self, signal):
        # Right pad the audio files that are shorter than target, cut if longer.
        if len(signal) > self.target_length:
            print(signal.shape)
            signal = signal[:, :self.target_length]
            print(signal.shape)
        elif len(signal) < self.target_length:
            signal = F.pad(signal, [0, self.target_length-signal.shape[1]])
            
        spectrogram = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop)(signal)
        return spectrogram

