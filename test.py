import torch
import torchvision

from model import *
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    #img = img / 2 + 0.5
    nping = img.numpy()
    plt.imshow(np.transpose(nping, (1, 2, 0)))
    plt.show()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=4)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = VGG()
net.load_state_dict(torch.load('./log/vgg10_net.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# dataiter = iter(test_loader)
# images, labels = dataiter.__next__()
#
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print(' '.join('%5s' % classes[predicted[i]] for i in range(4)))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# imshow(torchvision.utils.make_grid(images))

correct = 0
total =0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct +=(predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

