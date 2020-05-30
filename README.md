# Face-Generation-with-Variational-Autoencoders

Generating new faces from the Celebrity Faces dataset using Variational Autoencoders. The model was trained on TESLA GPU on Google Colab for 100 epochs which took a little over 8 hours to train on a celebrity dataset consisting of around 200K images divided into mini-batches of 64. The images were scaled to a size of 50x50 with one color channel due to shotage of VRAM provided by google colab.

## To use the trained model

Download the "face_generatorE100v5.pt" trained model from the <b>models</b> directory and load it into your python app using the following code:- (note, the folowing code is for users using Pytorch backend)

```
model = VAE()
model.load_state_dict(torch.load("./face_generatorE100v5.pt", map_location=torch.device("cpu")))
```

To generate sample images from the learnt distribution:-

```
# sampling a vector from a normal distribution
z = torch.randn(batch_size, 800) # batch_size := no.of images you want to generate, 800 -> dimension of latent space
output = model.decode(z.float()).detach()
# making a grid to display the images efficiently
grid = make_grid(output, nrow=8)
# displaying the image
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0), cmap='gray')
plt.show()
```

By the way don't forget the imports

```
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
```

### Improvements that can be made

U can make sliders that represent each of the 800 dimensions, and drag the slider which randomizer the latent space and you can see the faces changing. (It's fun :>)

### Output after 1st Epoch
![alt text](https://raw.githubusercontent.com/ppvalluri09/Face-Generation-with-Variational-Autoencoders/master/output/Version1Epochs1.png)

### Output after 25 Epochs
![alt text](https://raw.githubusercontent.com/ppvalluri09/Face-Generation-with-Variational-Autoencoders/master/output/Version2Epochs25.png)

### Output after 50 Epochs
![alt text](https://raw.githubusercontent.com/ppvalluri09/Face-Generation-with-Variational-Autoencoders/master/output/Version3Epochs50.png)

### Output after 75 Epochs
![alt text](https://raw.githubusercontent.com/ppvalluri09/Face-Generation-with-Variational-Autoencoders/master/output/Version4Epochs75.png)

### Output after 100 Epochs
![alt text](https://raw.githubusercontent.com/ppvalluri09/Face-Generation-with-Variational-Autoencoders/master/output/Version5Epochs100.png)
