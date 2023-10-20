## Install
Installment Code of Matplotlib, Tensor, Color Channel and Image Display
```
import matplotlib.pyplot as plt
def plot(x,title=None):
    # Move tensor to CPU and convert to numpy
    x_np = x.cpu().numpy()

    # If tensor is in (C, H, W) format, transpose to (H, W, C)
    if x_np.shape[0] == 3 or x_np.shape[0] == 1:
        x_np = x_np.transpose(1, 2, 0)

    # If grayscale, squeeze the color channel
    if x_np.shape[2] == 1:
        x_np = x_np.squeeze(2)

    x_np = x_np.clip(0, 1)

    fig, ax = plt.subplots()
    if len(x_np.shape) == 2:  # Grayscale
        im = ax.imshow(x_np, cmap='gray')
    else:
        im = ax.imshow(x_np)
    plt.title(title)
    ax.axis('off')
    fig.set_size_inches(10, 10)
    plt.show()
```
## Loading the Dataset
This is the downloading and loading of the Flowers 102 Data Set
```
!wget https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt
!wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip 'flower_data.zip'
```
This is the trasnformation of loading the dataset using an Image Folder & Loading the data for Batching
```
import torch
from torchvision import datasets, transforms
import os
import pandas as pd

data_dir = '/content/flower_data/'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transform)
dataset_labels = pd.read_csv('Oxford-102_Flower_dataset_labels.txt', header=None)[0].str.replace("'", "").str.strip()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
```
Extracting the Images and Lables
```
images, labels = next(iter(dataloader))

print(f"Images tensor shape: {images.shape}")
print(f"Labels tensor shape: {labels.shape}")
```
Loading the Images and Lables
```
i = 50
plot(images[i],dataset_labels[i]);
```
![Unknown-6](https://github.com/Carlbronge/FlowersDataSet/assets/143009718/da5ae279-4f20-4925-95b9-6325e2cca2ef)
```
i = 66
plot(images[i],dataset_labels[i]);
```
![Unknown-7](https://github.com/Carlbronge/FlowersDataSet/assets/143009718/ee3d9d64-6043-4faf-9087-7dc00e07ae20)

```
i = 98
plot(images[i],dataset_labels[i]);
```
![Unknown-8](https://github.com/Carlbronge/FlowersDataSet/assets/143009718/42932a4f-d07b-4d2c-a7ac-12f1a41a3ce5)


