import os
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

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Model
from models import CNN, MLP, LSTM
from utils import CatsAndDogsDataset, get_hyperparameters, get_test_dataframe


def test(args):
    """
    This function is used to test the model on the test data, and calculate the metrics, such as F1, F2,
    Average Precision, etc., and save them to a file.
    :param args: Command line arguments
    :return: None
    """
    model_folder_location = args.folder_location
    hp = get_hyperparameters(args)
    label = f'{args.type}_lr-{args.learning_rate}_bs-{args.batch_size}_do-{args.dropout}'
    test_dataframe = get_test_dataframe(args.test_data)
    try:
        print(os.path.join(model_folder_location, f'{label}.pth'))
        loaded_state_dict = torch.load(os.path.join(model_folder_location, f'{label}.pth'))
    except Exception as e:
        print(f"Error loading model state: {e}")

    # Load the test dataloader
    test_dataset = CatsAndDogsDataset(dataframe=test_dataframe)

    # Load the model
    model = None
    if args.type == 'cnn':
        model = CNN(dropout=hp['dropout'])
        model.load_state_dict(loaded_state_dict)
    elif args.type == 'base':
        model = MLP(dropout=hp['dropout'])
        model.load_state_dict(loaded_state_dict)
    elif args.type == 'lstm':
        model = LSTM(dropout=hp['dropout'])
        model.load_state_dict(loaded_state_dict)
    else:
        raise Exception(f"No such model type found: {args.type}")

    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for data, target in test_dataset:
            data = data.unsqueeze(0)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted)
            all_labels.append(target)
            print(f"Prediction: {predicted.item()}, Actual: {target}")

    # Convert to tensors for convenience
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    # Calculate metrics
    confusion_matrix = torch.zeros(2, 2)
    for t, p in zip(all_labels.view(-1), all_preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    # Precision, Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    f2 = 5 * (precision * recall) / (4 * precision + recall)

    print("Confusion Matrix:")
    print(confusion_matrix)
    print("F1 Score:", f1.item())
    print("F2 Score:", f2.item())

    # Calculate Average Precision
    AP = average_precision_score(all_labels, all_preds)
    print("Average Precision (AP):", AP)

    # Save the metrics to a file
    with open(os.path.join(model_folder_location, f'{label}.txt'), 'w') as file:
        file.write(f"TP: {TP}\n")
        file.write(f"TN: {TN}\n")
        file.write(f"FP: {FP}\n")
        file.write(f"FN: {FN}\n")
        file.write(f"F1 Score: {f1.item()}\n")
        file.write(f"F2 Score: {f2.item()}\n")
        file.write(f"Mean Average Precision (MAP): {AP}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('-t', '--type', type=str, help='Type of model', default='base')
    parser.add_argument('-d', '--test_data', type=str, help='Location of test data',
                        default='./Dataset/Audio Dataset/Cats and Dogs/test_data')
    parser.add_argument('-l', '--folder_location', type=str,
                        help='Location of folder that has the model hyperparameters', default='./models/')
    parser.add_argument('-lr', '--learning_rate', type=str, default='0.01')
    parser.add_argument('-bs', '--batch_size', type=str, default='128')
    parser.add_argument('-do', '--dropout', type=str, default='0.4')
    args = parser.parse_args()
    test(args)
