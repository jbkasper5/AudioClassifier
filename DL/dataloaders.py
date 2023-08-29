import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

def getDataloaders(dataset, batch_size = 16):
    if dataset == "cifar10":
        train_loader = CIFAR(train = True, batch_size = batch_size)
        test_loader = CIFAR(train = False, batch_size = batch_size)
        return (train_loader.dataloader, test_loader.dataloader)
    elif dataset == "cnn_audio":
        data = CNNAudioDataloader(batch_size = batch_size)
        train_loader = DataLoader(data, batch_size = batch_size, shuffle = True)
        return (train_loader, None)
    elif dataset == "transformer_audio":
        data = TransformerAudioDataloader(batch_size = batch_size)
        train_loader = DataLoader(data, batch_size = batch_size, shuffle = True)
        return train_loader, None
    else:
        raise NotImplementedError()
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


class CIFAR(Dataset):
    def __init__(self, train, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        batch_size = batch_size
        if train:
            trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
            self.dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
        else:
            testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
            self.dataloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # dataiter = iter(self.trainloader)
        # images, labels = next(dataiter)
        # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
        # imshow(torchvision.utils.make_grid(images))

class CNNAudioDataloader(Dataset):
    def __init__(self, batch_size):
        self.im_path = "/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/DataCapture/Spectrograms"
        self.datapaths = []
        for direction in os.listdir(self.im_path):
            for gender in os.listdir(os.path.join(self.im_path, direction)):
                for image in os.listdir(os.path.join(self.im_path, direction, gender)):
                    if "female" == gender.lower():
                        continue
                    if direction.lower() == "up":
                        label = UP
                    elif direction.lower() == "down":
                        label = DOWN
                    elif direction.lower() == "left":
                        label = LEFT
                    elif direction.lower() == "right":
                        label = RIGHT
                    self.datapaths.append((os.path.join(direction, gender, image), label))
        self.transforms = torch.nn.Sequential(
            # transforms.Grayscale(),
            transforms.RandomHorizontalFlip(p = 0.5),
            # transforms.RandomRotation(degrees = 30),
        )

    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, idx):
        path = os.path.join(self.im_path, self.datapaths[idx][0])
        label = torch.tensor(self.datapaths[idx][1])
        image = Image.open(path).convert("RGB")
        image = np.array(self.transforms(image))
        im_resize = torch.tensor(image.reshape(3, image.shape[0], image.shape[1]))
        im_resize = im_resize.float()
        return (im_resize, label)

class TransformerAudioDataloader(Dataset):
    def __init__(self, batch_size):
        self.im_path = "/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/DataCapture/Spectrograms"
        self.datapaths = []
        for direction in os.listdir(self.im_path):
            for gender in os.listdir(os.path.join(self.im_path, direction)):
                for image in os.listdir(os.path.join(self.im_path, direction, gender)):
                    if "female" == gender.lower():
                        continue
                    if direction.lower() == "up":
                        label = UP
                    elif direction.lower() == "down":
                        label = DOWN
                    elif direction.lower() == "left":
                        label = LEFT
                    elif direction.lower() == "right":
                        label = RIGHT
                    self.datapaths.append((os.path.join(direction, gender, image), label))

    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, idx):
        path = os.path.join(self.im_path, self.datapaths[idx][0])
        label = torch.tensor(self.datapaths[idx][1])
        image = ImageOps.grayscale(Image.open(path).convert("RGB"))
        image = torch.tensor(np.array(image))
        return image, label