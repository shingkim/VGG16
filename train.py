import torch
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from model import *
import torch.nn as nn
from time import sleep
from tqdm import tqdm


transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2)

test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     #img = img / 2 + 0.5
#     nping = img.numpy()
#     plt.imshow(np.transpose(nping, (1, 2, 0)))
#     plt.show()
#
# dataiter = iter(train_loader)
# images, labels = dataiter.__next__()
# print(images.size())
# labels = labels.numpy()
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(images))


#net = VGG()

net = torchvision.models.vgg16(pretrained=True)
#print(net)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5,15,30], gamma=0.1)
num_epochs = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    for i,data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #scheduler.step()

        running_loss += loss.item()
        if i%1000 == 999:
            print('[%d, %5d] loss: %3f'%
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    sleep(0.5)
print('Finished Training')
PATH = './log/vgg_pre_net.pth'
torch.save(net.state_dict(), PATH)