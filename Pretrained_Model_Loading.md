## Installs Necessary
The following code is required to be installed before proceeding
```
import torch
from torchvision import models, transforms
import requests
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
## Defining the Model
Defining Alexnet Model
```
alexnet = models.alexnet(pretrained=True).to(device)
labels = {int(key):value for (key, value) in requests.get('https://s3.amazonaws.com/mlpipes/pytorch-quick-start/labels.json').json().items()}
```
Transforming the Image for Use In the Model
```
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
```
Converting the Image to PIL
```
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
img = to_pil(img)
```
Classifying the Image with Alexnet
```
scores, class_idx = alexnet(img_t).max(1)
print('Predicted class:', labels[class_idx.item()])
```
Weights and Bias
```
w0 = alexnet.features[0].weight.data
w1 = alexnet.features[3].weight.data
w2 = alexnet.features[6].weight.data
w3 = alexnet.features[8].weight.data
w4 = alexnet.features[10].weight.data
w5 = alexnet.classifier[1].weight.data
w6 = alexnet.classifier[4].weight.data
w7 = alexnet.classifier[6].weight.data
```

