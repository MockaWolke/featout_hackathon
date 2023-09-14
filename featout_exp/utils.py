from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from featout_exp import IMAGESIZE
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd

transformations = transforms.Compose([
    transforms.Resize(IMAGESIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
        self.transform = transformations

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Load image
        image_path = self.dataframe.iloc[index, 0]
        image = Image.open(str(image_path)).convert('RGB')

        # Get label
        label_str = self.dataframe.iloc[index, 1]
        label = 0 if label_str == "Cat" else 1

        # Apply transformations on the image
        if self.transform:
            image = self.transform(image)

        return image, label





def load_model(device = "cuda"):
    
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(device)
    
    return model


def load_data(kind, batchsize, shuffle):
    
    assert kind in ["train", "test"]
        
    path = Path(__file__).parent.parent / f"csvs/{kind}.csv"
    
    df = pd.read_csv(str(path), index_col= 0)
            
    dataset = CustomDataset(dataframe=df)

    loader = DataLoader(dataset, batch_size=batchsize, shuffle= shuffle)
    
    return loader
        