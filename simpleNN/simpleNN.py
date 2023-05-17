from __future__ import print_function
import argparse
import os, sys
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset,DataLoader

curr_path = os.getcwd()
sys.path.append(curr_path)
from custom_data_loader.dataloader import load_ocean

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 4)

    def forward(self, x):
        x = self.fc1(x)
        #x = nn.BatchNorm1d(64) 
        x = F.relu(x)
        x = self.fc2(x)
        #x = nn.BatchNorm1d(32) 
        x = F.relu(x)
        output = self.output(x)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.huber_losss(output, target, delta=0.8)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.huber_losss(output, target, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch simpleNN Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--num_cpus', type=int, default=1, metavar='N',
                        help='number of CPU vCores to train with (default: use all available)')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print()
    print("Number of CPU vCores specified to be used {}".format(args.num_cpus))
    print("Total # of CPU threads on OS {}".format(os.cpu_count()))
    print("Total # of usable CPU threads on OS {}".format(len(os.sched_getaffinity(0))))

    print("Total # of Intra-op CPU threads - PyTorch {}".format(torch.get_num_threads()))
    print("Total # of Inter-op threads - PyTorch {}".format(torch.get_num_interop_threads()))
    print()
    print("Setting # of Intra-op and Inter-op CPU threads in PyTorch to {}".format(args.num_cpus))
    torch.set_num_threads(args.num_cpus)
    torch.set_num_interop_threads(args.num_cpus)
    print()
    print("Total # of Intra-op CPU threads - PyTorch {}".format(torch.get_num_threads()))
    print("Total # of Inter-op threads - PyTorch {}".format(torch.get_num_interop_threads()))
    print()
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset, test_dataset = load_ocean("./dataset/train.txt", "./dataset/test.txt")
    Train_DS = TensorDataset(train_dataset[0],train_dataset[1])
    train_loader = DataLoader(Train_DS, shuffle=True, batch_size = 256)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train(args, model, device, train_loader, optimizer, epoch)        
        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print("Epoch training time {}".format(elapse_time))        
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':    
    overall_start = time.time()
    main()
    total_time = time.time() - overall_start
    total_time = datetime.timedelta(seconds=total_time)
    print("Total time {}".format(total_time))      
