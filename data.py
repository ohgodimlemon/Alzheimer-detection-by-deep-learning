import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
class MRIDataset(Dataset):

    def __init__(self, split_type: str, split_dir: str) -> None:
        self.split_type = split_type
        self.split_dir = split_dir
        self.img_paths = []
        for folder_name in os.listdir(self.split_dir):
            folder_path = os.path.join(self.split_dir, folder_name)
            self.img_paths.extend([os.path.join(folder_path, file) for file in os.listdir(folder_path)])
        print(f"num img for {split_type}: {len(self.img_paths)}")
            

    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:

        label_mapping = {
            "No Impairment": 0,
            "Very Mild Impairment": 1,
            "Mild Impairment": 2,
            "Moderate Impairment": 3
        }
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32)

        parent_dir = os.path.dirname(img_path)
        alzheimer_class = os.path.basename(os.path.normpath(parent_dir))

        #print(img.shape)

        return img, label_mapping.get(alzheimer_class, -1) 

def load_data(batch_size: int = 4, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    train_dataset = MRIDataset("train", rf"C:\Users\vedan\Desktop\Uni Study\Alzheimer-detection-by-deep-learning\alzheimer_dataset\Combined Dataset\train")
    test_dataset = MRIDataset("test", rf"C:\Users\vedan\Desktop\Uni Study\Alzheimer-detection-by-deep-learning\alzheimer_dataset\Combined Dataset\test")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, test_dataloader
# train - no impairment
# train - very mild impairment
# train - mild impairment
# train - moderate impairment

# test - no impairment
# test - very mild impairment
# test - mild impairment
# test - moderate impairment
        