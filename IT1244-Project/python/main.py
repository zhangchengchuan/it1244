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

# Dataloader
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import pandas as pd

# Models
from models import CNN
from utils import CatsAndDogsDataset, get_train_dataframe, get_hyperparameters

# Warning: to mute deprecation notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module", lineno=1518)

def train(args):
    start = time.time()
    print("Training Started.")

    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters retrieved from hyperparameters.py
    hp = get_hyperparameters(args.hyperparameters)
    
    # Create dataloader class
    train_dataframe = get_train_dataframe(args.train_data)
    train_dataset = CatsAndDogsDataset(target_length=hp['target_length'], n_fft=hp['n_fft'], hop=hp['hop_length'], dataframe=train_dataframe)

    # Create model
    model = None
    if args.type == 'cnn':
        model = CNN(n_fft=hp['n_fft'], hop=hp['hop_length']).to(device)
    else:
        raise Exception(f"No such model type found: {args.type}")


    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])

    # Training the model
    kf = KFold(n_splits=hp['kfold_splits'], shuffle=True, random_state=10)
    all_losses = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset.df)):
        print(f"Fold: {fold}")
        train_subset = Subset(train_dataset, train_idx)
        train_loader = DataLoader(train_subset, batch_size=hp['batch_size'], shuffle=True)

        # Training loop for this fold
        losses = []
        for epoch in range(hp['epochs']):
            model.train()
            for feature, labels in train_loader:
                feature, labels = feature.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(feature)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
            print(loss.item())
            losses.append(loss.item())
        all_losses.append(losses)

    # Save the plots
    fig, axs = plt.subplots(hp['kfold_splits'], 1, figsize=(6,3*hp['kfold_splits'])) # number of splits = number of plots
    for idx, losses in enumerate(all_losses):
        axs[idx].plot(np.arange(hp['epochs']), losses)
        axs[idx].set_title(f'Training Loss over epochs for fold {fold}')
        axs[idx].set_xlabel('Loss')
        axs[idx].set_ylabel('Epochs')
    plt.tight_layout()
    fig.savefig('./plots/CNN/plots.png')

    # Save the model
    torch.save(model.state_dict(), './models/CNN/cnn.pth')

    # Save the losses
    file_path = './models/CNN/losses.txt'
    with open(file_path, 'w') as file:
        # Write each list element to the file, one element per line
        for idx in range(hp['kfold_splits']):
            file.write(f"----Fold {idx}----\n")
            for item in all_losses[idx]:
                file.write(f"{item}\n")

    print(f"Training Concluded. Time taken: {time.time()-start}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('-t','--type', type=str, help='Type of model', default='cnn')
    parser.add_argument('-d','--train_data', type=str, help='Location of training data', default='./Dataset/Audio Dataset/Cats and Dogs/data')
    parser.add_argument('-hp', '--hyperparameters', type=str, help='Hyperparameters location', default='./models/CNN/hyperparameters.py')
    args = parser.parse_args()
    train(args)