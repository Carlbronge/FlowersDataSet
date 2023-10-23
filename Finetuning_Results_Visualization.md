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
## Results Visualization
Normalizing the Numpy Array
```
def scale(img):
    max_value = img.max()
    min_value = img.min()
    normalized_array = (img - min_value) / (max_value - min_value)
    return normalized_array
```
```
def tensor_plot(img_t,index=0):
    numpy_array = img_t[index,:,:,:].cpu().numpy()
    numpy_array_transposed = numpy_array.transpose(1, 2, 0)
    numpy_array_transposed = scale(numpy_array_transposed)
    plt.imshow(numpy_array_transposed)
    plt.show()
```
Visualizing
```
tensor_plot(img_t)
```
![Unknown-9](https://github.com/Carlbronge/FlowersDataSet/assets/143009718/32e6bfca-0799-4fcd-9181-80c30247901e)

Convulving
```
f0 = F.conv2d(img_t, w0, stride=4, padding=2)
```
Plotting
```
i = 5
plt.imshow(f0[0,i,:,:].cpu().numpy())
```
![Unknown-10](https://github.com/Carlbronge/FlowersDataSet/assets/143009718/21e9544e-329d-467e-809b-929b313e979b)

