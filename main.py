import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from data import load_data
from train import training_loop, test_fn
from utils import load_checkpoint, save_model

LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_WORKERS = 6

LOAD_MODEL = False
DEVICE = "cpu"

def main():
    # model = models.vgg16(pretrained=True)
    # #model = models.resnet18(pretrained=True)  
    # model.classifier[6] = nn.Linear(model.classifier[6].in_features, 4)
    # #model.fc = nn.Linear(model.fc.in_features, 4)
    # model = model.to(DEVICE)

    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, test_loader = load_data(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # resume_epoch = 0
    # if LOAD_MODEL:
    #     print("Continuing from last checkpoint...")
    #     checkpoint_path = "checkpoint_epoch_2.pth.tar"  # Specify your checkpoint file path
    #     model, optimizer, resume_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)
    #     resume_epoch = resume_epoch or 0  # Default to 0 if loading fails
    #     print(f"Resuming from epoch {resume_epoch + 1}...")

    # training_loop(train_loader, model, optimizer, loss_fn)

    # save_model(model, filename="trained_resnet_model.pth")
    model_path = rf"C:\Users\vedan\Desktop\Uni Study\Alzheimer-detection-by-deep-learning\checkpoint_epoch_6.pth.tar"
    model = models.vgg16(pretrained=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    avg_loss, acc = test_fn(test_loader, model, loss_fn)
    print(f"avg loss: {avg_loss} acc: {acc}")


if __name__ == "__main__":
    main()
