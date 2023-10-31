import sys
import time
import argparse
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

# Model
from models import CNN
from utils import CatsAndDogsDataset, get_hyperparameters, get_test_dataframe

def test(args):
    loaded_state_dict = torch.load(args.state)
    hp = get_hyperparameters(args.hyperparameters)
    test_dataframe = get_test_dataframe(args.test_data)

    # Load the test dataloader
    test_dataset = CatsAndDogsDataset(target_length=hp['target_length'], n_fft=hp['n_fft'], hop=hp['hop_length'], dataframe=test_dataframe)

    # Load the model
    model = None
    if args.type== 'cnn':
        model = CNN(n_fft=hp['n_fft'], hop=hp['hop_length'])
        model.load_state_dict(loaded_state_dict)
    
    labels = {
        0: "Cat",
        1: "Dog"
    }
    model.eval()
    with torch.no_grad():
        correct = 0
        for idx in range(len(test_dataframe)):
            row = test_dataframe.iloc[idx]
            audio_path = row['audio_file']
            signal, sr = torchaudio.load(audio_path)
            if len(signal) > 200000:
                print(signal.shape)
                signal = signal[:, :200000]
                print(signal.shape)
            elif len(signal) < 200000:
                signal = F.pad(signal, [0, 200000-signal.shape[1]])
            
            spectrogram = T.Spectrogram(n_fft=2048, hop_length=1024)(signal)
            classification = row['classification']
            prediction = model(spectrogram.unsqueeze(0))
            if labels[torch.argmax(prediction,dim=1).item()] == labels[classification]:
                correct += 1
            print(f"Prediction: {labels[torch.argmax(prediction,dim=1).item()]}, Actual: {labels[classification]}")

    print(f"Total accuracy: {correct/len(test_dataframe)}")
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('-t','--type', type=str, help='Type of model', default='cnn')
    parser.add_argument('-s','--state', type=str, help='Location of model state', default='./models/CNN/cnn.pth')
    parser.add_argument('-d','--test_data', type=str, help='Location of test data', default='./Dataset/Audio Dataset/Cats and Dogs/test_data')
    parser.add_argument('-hp', '--hyperparameters', type=str, help='Hyperparameters location', default='./models/CNN/hyperparameters.py')
    args = parser.parse_args()

    test(args)