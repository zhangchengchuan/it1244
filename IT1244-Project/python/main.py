import sys
import os
import time
import argparse
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
from models import CNN, MLP, LSTM
from utils import CatsAndDogsDataset, get_train_dataframe, get_hyperparameters

# Seed
torch.manual_seed(0)

# Warning: to mute deprecation notifications
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module", lineno=1518)


def train(args):
    start = time.time()
    print("Training Started.")

    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Main folder location to store artifacts
    model_folder_location = args.folder_location
    hp = get_hyperparameters(args)

    # Create dataloader class
    train_dataframe = get_train_dataframe(args.train_data)
    train_dataset = CatsAndDogsDataset(dataframe=train_dataframe)

    # Create model
    model = None
    if args.type == 'cnn':
        model = CNN(dropout=hp['dropout']).to(device)
    elif args.type == 'base':
        model = MLP(dropout=hp['dropout']).to(device)
    elif args.type == 'lstm':
        model = LSTM(dropout=hp['dropout']).to(device)
    else:
        raise Exception(f"No such model type found: {args.type}")

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])

    # Training the model
    KFOLD_SPLITS = 4
    EPOCHS = 20
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=10)
    all_losses = []
    # Loop over each fold and train
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset.df)):
        print(f"Fold: {fold}")
        # Create dataloaders for each fold, the training subset and the validation subset
        train_subset = Subset(train_dataset, train_idx)
        train_loader = DataLoader(train_subset, batch_size=hp['batch_size'], shuffle=True)
        validation_subset = Subset(train_dataset, val_idx)
        validation_loader = DataLoader(validation_subset, batch_size=hp['batch_size'], shuffle=True)

        # Training loop for this fold
        train_losses = []
        val_losses = []
        # For each epoch, train and validate
        for epoch in range(EPOCHS):
            cur_training_loss = 0
            model.train()
            for feature, labels in train_loader:
                feature, labels = feature.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(feature)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                cur_training_loss += loss.item()
            epoch_training_loss = cur_training_loss / len(train_loader)
            print(f"Epoch Loss: {epoch_training_loss}")
            train_losses.append(epoch_training_loss)

            cur_validation_loss = 0
            model.eval()
            with torch.no_grad():
                for feature, labels in validation_loader:
                    feature, labels = feature.to(device), labels.to(device)
                    output = model(feature)
                    loss = loss_fn(output, labels)
                    cur_validation_loss += loss.item()
                epoch_validation_loss = cur_validation_loss / len(validation_loader)
                val_losses.append(epoch_validation_loss)
                # print(f"Epoch Validation Loss: {epoch_training_loss}")
        all_losses.append((train_losses, val_losses))

    # Save the plots
    fig, axs = plt.subplots(KFOLD_SPLITS, 1, figsize=(6, 3 * KFOLD_SPLITS))  # number of splits = number of plots
    for idx, losses in enumerate(all_losses):
        train_loss, val_loss = losses
        axs[idx].plot(np.arange(EPOCHS), train_loss, label="Training Loss")
        axs[idx].plot(np.arange(EPOCHS), val_loss, label="Validation Loss")
        axs[idx].set_title(f'Training and Validation Loss over {EPOCHS} epochs for fold {idx + 1}')
        axs[idx].set_xlabel('Epochs')
        axs[idx].set_ylabel('Loss')
        axs[idx].legend()
    plt.tight_layout()

    # Check if folders exists. if not create.
    if not os.path.exists(model_folder_location):
        os.makedirs(model_folder_location)
        print("Created folder for models, logs, plots.")

    # Model label for saving files
    label = f'{args.type}_lr-{args.learning_rate}_bs-{args.batch_size}_do-{args.dropout}'
    fig.savefig(os.path.join(model_folder_location, f'{label}_loss.png'))

    torch.save(model.state_dict(), os.path.join(model_folder_location, f'{label}.pth'))
    with open(os.path.join(model_folder_location, f'{label}_logs.txt'), 'w') as file:
        # Write each list element to the file, one element per line
        for idx in range(KFOLD_SPLITS):
            file.write(f"----Fold {idx}----\n")
            for item in all_losses[idx]:
                file.write(f"{item}\n")

    print(f"Training Concluded. Time taken: {time.time() - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('-t', '--type', type=str, help='Type of model', default='base')
    parser.add_argument('-d', '--train_data', type=str, help='Location of training data',
                        default='./Dataset/Audio Dataset/Cats and Dogs/data')
    parser.add_argument('-l', '--folder_location', type=str,
                        help='Location of folder for storing logs, plots, models and hyperparameters',
                        default='./models/')
    parser.add_argument('-lr', '--learning_rate', type=str, default='0.01')
    parser.add_argument('-bs', '--batch_size', type=str, default='128')
    parser.add_argument('-do', '--dropout', type=str, default='0.4')
    args = parser.parse_args()
    train(args)
