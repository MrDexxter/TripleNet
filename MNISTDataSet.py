import random
import torch
from torchvision import datasets
from torchvision import transforms

class MNISTDataSet(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(MNISTDataSet, self).__init__(*args, **kwargs)
        if kwargs["train"] is True:
            self.xdata, self.y = self.train_data, self.train_labels
        else:
            self.xdata, self.y = self.test_data, self.test_labels

    def __getitem__(self, idx):
        x1, t1 = self.xdata[idx], self.y[idx]
        is_diff = random.randint(0, 1)
        while True:
            idx2 = random.randint(0, len(self.y) - 1)
            x2, t2 = self.xdata[idx2], self.y[idx2]
            if is_diff and t1 != t2:
                break
            if not is_diff and t1 == t2:
                break
        x1 = x1.reshape(1, 28, 28) / 255
        x2 = x2.reshape(1, 28, 28) / 255
        return x1, x2, int(is_diff)

    def get_loaders(batch_size):
        train_loader = torch.utils.data.DataLoader(
            MNISTDataSet("./data", train=True, download=True,
                         transform=transforms.Compose([
                             # transforms.RandomHorizontalFlip(),
                             transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            MNISTDataSet("./data", train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor()])),
            batch_size=batch_size, shuffle=False)
        return train_loader, test_loader