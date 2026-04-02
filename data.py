import kagglehub
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# CIFAKE Dataset specs:
# 32x32 pixels
# 3 channels RGB


def load_data():
    # load data from kagglehub
    url = "birdy654/cifake-real-and-ai-generated-synthetic-images"
    path = kagglehub.dataset_download(url)

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    train_dataset = ImageFolder(root=os.path.join(path,'train'),transform=transform)
    test_dataset = ImageFolder(root=os.path.join(path,'test'),transform=transform)
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size:  {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    return train_loader,test_loader
