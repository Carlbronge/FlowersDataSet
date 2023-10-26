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
## Adjust the final layer of AlexNet
Modify the classifier to match the number of classes:
```
num_classes = len(trainset.classes)
alexnet.classifier[6] = torch.nn.Linear(alexnet.classifier[6].in_features, num_classes)
```
## Set up the optimizer and criterion
```
import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)
```
## Move the model to the appropriate device
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alexnet = alexnet.to(device)
```
## Training loop
```
num_epochs = 1

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = alexnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

print('Finished Training')
```
Epoch 1, Loss: 2.062453852227283
Finished Training
## Prediction and Visualization
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataiter = iter(trainloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

outputs = alexnet(images)
_, predicted = torch.max(outputs, 1)
predicted_flower_names = [flower_names[p.item()] for p in predicted]
true_flower_names = [flower_names[label.item()] for label in labels]

# Move images back to CPU for visualization
images = images.cpu()

imshow(torchvision.utils.make_grid(images), title=[f"True: {true}, Predicted: {pred}" for true, pred in zip(true_flower_names, predicted_flower_names)])
```
![Unknown-14](https://github.com/Carlbronge/FlowersDataSet/assets/143009718/ace6f8c2-2e44-4c4c-9d31-122c25502243)
