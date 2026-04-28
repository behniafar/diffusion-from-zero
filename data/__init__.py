import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

ROOT = __file__[:-11]
print(ROOT)

def get_cifar10(
        train = True, 
        normalize = False
        ):
    transform = [torchvision.transforms.ToTensor()]
    if normalize:
        transform.append(torchvision.transforms.Normalize([.5], [.5]))
    transform = torchvision.transforms.Compose(transform)
    return torchvision.datasets.CIFAR10(ROOT, train=train, transform=transform)

if __name__ == '__main__':
    get_cifar10()
    print("It's working!")
