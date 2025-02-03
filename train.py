import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import ( 
    save_checkpoint, 
    calculate_val_loss
)

DEVICE = "cpu"
NUM_EPOCHS = 4

label_mapping = {
    "No Impairment": 0,
    "Very Mild Impairment": 1,
    "Mild Impairment": 2,
    "Moderate Impairment": 3
}

# Inverse the dictionary to get the class names from class indices
class_names = {v: k for k, v in label_mapping.items()}

def train_fn(loader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module) -> float:
    model.train()
    train_loss = 0.0

    loop = tqdm(loader, total=len(loader), desc="Training", leave=True)
    for images, alz_class in loop:
        images = images.to(DEVICE)
        alz_class = alz_class.to(DEVICE)

        predictions = model(images)
        # #
        # _, predicted_labels = torch.max(predictions, 1)
        # predicted_class_names = [class_names[label.item()] for label in predicted_labels]
        # actual_class_names = [class_names[label.item()] for label in alz_class]
        # for pred_class, true_class in zip(predicted_class_names, actual_class_names):
        #     print(f"Pred class: {pred_class} | Actual class: {true_class}")
        # #
        loss = loss_fn(predictions, alz_class)
        train_loss += loss.item() * images.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    return train_loss / len(loader.dataset)

def test_fn(loader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    model.eval()
    test_loss = 0.0
    correct_pred = 0
    total_samples = 0

    loop = tqdm(loader, total = len(loader), desc='Testing', leave=True)

    with torch.no_grad():
        for image, alz_class in loop:
            image = image.to(DEVICE)
            alz_class = alz_class.to(DEVICE)

            # Forward pass
            predictions = model(image)

            # Compute loss
            loss = loss_fn(predictions, alz_class)
            test_loss += loss.item() * image.size(0)  # Accumulate loss

            # Calculate accuracy
            _, predicted_labels = torch.max(predictions, 1)  # Get the index of the maximum value in predictions (class with highest probability)
            correct_pred += (predicted_labels == alz_class).sum().item()  # Count correct predictions
            total_samples += image.size(0)

            loop.set_postfix(loss=loss.item())

    avg_loss = test_loss / total_samples
    accuracy = correct_pred / total_samples * 100 

    return avg_loss, accuracy

def training_loop(train_loader, model, optimizer, loss_fn):
    loss_per_epoch = []
    for epoch in range(0, NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn)
        loss_per_epoch.append(epoch_loss)
        save_checkpoint(
            model, optimizer, filename=f"checkpoint_epoch_{epoch + 1}.pth.tar"
        )
    epochs = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs, loss_per_epoch, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.grid()
    plt.show()



        