import torchvision
import torch

def get_loader_normal(batch_size_train=128, batch_size_test=10000):
  train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
  transform=torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  ])),
  batch_size=batch_size_train, shuffle=True)
  test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
  transform=torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  ])),
  batch_size=batch_size_test, shuffle=False)
  return train_loader, test_loader
