import os
import sys
import time
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
    path = os.path.join(directory, '*')
    modified_path = path.replace("/", os.path.sep)
    files = sorted(glob.glob(modified_path))
    df = pd.DataFrame(columns=['idx', 'audio_file', 'classification'])
    for idx, audio_file in enumerate(files):
        audio_path = audio_file.split('/')[-1]
        classification = audio_path.split('_')[0]
        row = pd.DataFrame([{"idx":idx, "audio_file": audio_file, "classification": 1 if classification=='dog' else 0}])
        df = pd.concat([df, row])

    return df

def get_test_dataframe(directory):
    path = os.path.join(directory, '*')
    modified_path = path.replace("/", os.path.sep)
    files = sorted(glob.glob(modified_path))
    df = pd.DataFrame(columns=['idx', 'audio_file', 'classification'])
    for idx, audio_file in enumerate(files):
        audio_path = audio_file.split('/')[-1]
        classification = audio_path.split('_')[0]
        row = pd.DataFrame([{"idx":idx, "audio_file": audio_file, "classification": 1 if classification=='dog' else 0}])
        df = pd.concat([df, row])

    return df

def get_hyperparameters(args):
    try:
        print("Fetching Hyperparameters:\n")
        hp_dict = {}
        hp_dict['learning_rate'] = float(args.learning_rate)
        hp_dict['dropout'] = float(args.dropout)
        hp_dict['batch_size'] = int(args.batch_size)
        # print(type(hp_dict['batch_size']), type(hp_dict['dropout']), type(hp_dict['learning_rate']))
        return hp_dict
    except Exception as e:
        print(f"Error reading hyperparameters: {e}")

class CatsAndDogsDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
 
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
        if len(signal) > 200000:
            print(signal.shape)
            signal = signal[:, :200000]
            print(signal.shape)
        elif len(signal) < 200000:
            signal = F.pad(signal, [0, 200000-signal.shape[1]])
            
        spectrogram = T.Spectrogram(n_fft=2048, hop_length=1024)(signal)
        return spectrogram

