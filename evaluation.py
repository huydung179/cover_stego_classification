import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from data_preparation import prepare_test
from model import LeNet


def parse_arguments():
    """
    Parse Paths for test data and saving the model.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Testing script for the model.")
    parser.add_argument('--data_dir', type=str, default='./data_test',
                        help='Directory for testing images.')
    parser.add_argument('--model_path', type=str, default='./output_models/Part_1_lenet.pt',
                        help='Path to of output model.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for testing.')
    return parser.parse_args()


def validate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             test_loader: torch.utils.data.DataLoader,
             device: torch.device) -> (float, float):
    """
    Validate the model and return average loss and accuracy.

    :param model: Trained PyTorch model
    :param criterion: Loss function used
    :param test_loader: DataLoader for validation data
    :param device: Computation device ('cuda' or 'cpu')
    :return: average validation loss, accuracy
    """
    model.eval()
    loss = 0
    outputs, targets = [], []

    # Evaluation without gradient calculation
    with torch.no_grad():
        for images, labels in tqdm(test_loader, ncols=80):
            # Predict and calculate batch loss
            images, labels = images.to(device), labels.float().to(device)
            output = model(images).squeeze()
            loss += criterion(output, labels).item()

            # Store predictions and targets for calculating accuracy
            outputs.append(output.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    # Average loss and accuracy calculation
    loss /= len(test_loader)
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    preds = (outputs > 0).astype(int)
    accuracy = np.mean(preds == targets)

    return loss, accuracy


def initialize_device() -> torch.device:
    """Initialize CUDA device if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_model(model: torch.nn.Module,
               criterion: torch.nn.Module,
               test_loader: torch.utils.data.DataLoader,
               device: torch.device) -> None:
    """
    Train the model and save when validation loss improves.

    :param model: PyTorch model to eval
    :param criterion: Loss function used for optimization
    :param test_loader: DataLoader for validation data
    :param device: Computation device ('cuda' or 'cpu')
    """

    # Validate
    valid_loss, valid_acc = validate(model, criterion, test_loader, device)

    print(f'Test loss: {valid_loss:.4f}, '
          f'Test accuracy: {valid_acc:.4f}')


def main():
    # Initialize the device, model, loss function, optimizer, and data loaders
    device = initialize_device()
    print(f'Using device: {device}')
    model = LeNet()
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loader = prepare_test(args.data_dir, args.batch_size)

    # Eval the model
    eval_model(model, criterion, test_loader, device)


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    main()
