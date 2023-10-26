## Installs Necessary
The following code is required to be installed before proceeding
```
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
```
## Data Preprocessing
We first define the necessary transformations to apply to our dataset.
```
# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```
## Loading the Dataset
Here, we load the flower dataset and apply the defined transformations.
```
# Load dataset
trainset = datasets.ImageFolder(root='flower_data/train', transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
```
## Load Pre-trained AlexNet Model
We'll utilize the pre-trained AlexNet model and modify its final layer to match the number of classes in our flower dataset.
```
# Load the pre-trained AlexNet model
alexnet = torchvision.models.alexnet(pretrained=True)
num_classes = len(trainset.classes)
alexnet.classifier[6] = torch.nn.Linear(alexnet.classifier[6].in_features, num_classes)
```
## Fetch Dataset Labels
We obtain the human-readable flower names from a provided text file.
```
dataset_labels = pd.read_csv('Oxford-102_Flower_dataset_labels.txt', header=None)[0].str.replace("'", "").str.strip()
flower_names = dataset_labels.tolist()
```
## Predictions with AlexNet
We fetch a batch of images from the trainloader and make predictions using the modified AlexNet.
```
# Fetch a batch of data
dataiter = iter(trainloader)
images, labels = next(dataiter)

outputs = alexnet(images)
_, predicted = torch.max(outputs, 1)
predicted_flower_names = [flower_names[p.item()] for p in predicted]
```
## Visualization Function
We define a helper function to visualize the tensors as images.
```
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
```
## Display Predicted Images
Finally, we visualize the images and their predicted labels.
```
# Plot the images and their predicted labels
imshow(torchvision.utils.make_grid(images), title=[name for name in predicted_flower_names])
```
![Unknown-13](https://github.com/Carlbronge/FlowersDataSet/assets/143009718/58a9b744-12cc-45e0-b08e-96ddb893b7da)


