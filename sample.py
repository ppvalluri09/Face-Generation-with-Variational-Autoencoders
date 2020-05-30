import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from models import *

def sample_images(batch_size, model):
    # creating a random latent vector
    # 800 since our latent space is of 800 dimensions
    z = torch.randn(batch_size, 800)
    model.eval()
    with torch.no_grad():
        output = model.decode(z.float()).detach()
        if batch_size % 8 == 0:
            grid = make_grid(output, nrow=8)
        else:
            grid = output
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    states = [(1, 1), (25, 2), (50, 3), (75, 4), (100, 5)]
    for epoch, version in states:
        model = VAE()
        # the model was trained on a GPU so we need to map location to cpu
        model.load_state_dict(torch.load(f"./models/face_generatorE{epoch}v{version}.pt",
                                         map_location=torch.device("cpu")))
        sample_images(64, model)
