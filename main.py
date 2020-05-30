from models import *
from preprocess import *
# from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.optim as optim
from matplotlib import pyplot as plt

BATCH_SIZE = 64

image_transform = transforms.Compose([transforms.ToTensor()])
train_data = LoadCelebFaces(transform=image_transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

print(f'''Training Data Size: {len(train_data)}\n''')

cuda = torch.cuda.is_available()

model = VAE()
if cuda:
    model = model.cuda()
print_model_architecture = lambda model: f'\nModel Architecture' + '='*32 + '\n' + f'{model}\n' + '='*32 + '\n'
print(print_model_architecture(model))

def loss_function(reconstructed_x, x, mean, logvar):
    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return BCE + KLD

def plot_img(image, epoch, cmap='gray'):
    plt.imshow(image.permute(1, 2, 0), cmap=cmap)
    plt.axis('off')
    plt.title(f'Generated Image after {epoch} epochs')
    plt.show()

def train(model, train, EPOCHS):
    print('Model Training...')
    model.train()
    training_loss = []

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    version = 1

    for epoch in range(EPOCHS):
        train_loss = 0.0
        for i, data in enumerate(train):

            optimizer.zero_grad()

            if cuda:
                data = data.cuda()

            out, mean, std = model(data)
            loss = loss_function(out, data.float(), mean, std)
            loss.backward()

            train_loss += loss.item()

            optimizer.step()

            if i % 500 == 499:
                print(f'Epoch {epoch+1}, Batch [{i+1}/{len(train)}], Training Loss {loss.item()/i+1}')

        average_train_loss = train_loss / len(train)
        print(f'Training Set Loss {epoch+1}: {round(average_train_loss, 4)}')
        training_loss.append(average_train_loss)

        if epoch % 25 == 0:
            torch.save(model.state_dict(), f"./models/face_generatorE{epoch+1}v{version}.pt")
            version += 1
            
            model.eval()
            with torch.no_grad():
                z = torch.randn(BATCH_SIZE, 800)
                if cuda:
                    z = z.cuda()
                output_img = model.decode(z).cpu()
                print(output_img.shape)

                grid_img = make_grid(output_img, nrow=8)
                plot_img(grid_img, epoch, cmap='gray')
            model.train()
    return model, training_loss

model, training_loss = train(model, train_loader, 100)
torch.save(model.state_dict(), "./models/face_generatorE100v5.pt")
