import torch

import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from os.path import join

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        
        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        return self.fc1(out)
        
class Train(CNNModel):

    def __init__(self):
        super().__init__()
        self.model = CNNModel()
        self.criterion = nn.CrossEntropyLoss()

    def optimizer(self, learning_rate):
        return torch.optim.SGD(self.model.parameters(), lr = learning_rate)

    def epochs(self, iterations, batch_size):
        return int(n_iters/(len(train_dataset)/batch_size))

    def trainModel(self, train_data, learning_rate, n_iters, batch_size, GPU, directory, name):
        print("Training Model")
        epochs = self.epochs(n_iters, batch_size)
        optimizer = self.optimizer(learning_rate)
        train_data = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

        if GPU != False:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            device = self.model.to(device)

        for epoch in range(epochs):
            for images, labels in train_data:
                
                if GPU != False:
                    images = images.requires_grad_().to(device)
                    labels = labels.to(device)
                else:
                    images = images.requires_grad_()

                optimizer.zero_grad() 
                images = self.model(images)
                loss = self.criterion(images, labels)
                loss.backward()
                optimizer.step()
            print("Epoch: {}, Loss: {}".format(epoch, loss))
        
        torch.save(self.model.state_dict(), join(directory, name+".pkl"))

class Test(CNNModel):

    def __init__(self, directory, name):
        super().__init__()
        self.model = CNNModel()
        self.model.load_state_dict(torch.load(join(directory, name)))

    def testModel(self, test_data, batch_size, GPU):
        print("Testing Model")
        total = 0
        correct = 0
        test_data = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

        if GPU != False:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            device = self.model.to(device)

        for images, labels in test_data:

            if GPU != False:
                images = images.requires_grad_().to(device)
            else:
                images = images.requires_grad_()

            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()
    
        accuracy = 100 * correct / total #Defintely not the right way to measure accuracy. 

        print("Accuracy: {}".format(accuracy))

if __name__ == "__main__":

    n_iters = 3000
    batch_size = 100
    learning_rate = 0.1
    train_CNN = Train()
    train_dataset = dsets.MNIST(root = "./data", train = True, transform = transforms.ToTensor(), download = True)
    epochs = int(n_iters/(len(train_dataset)/batch_size))
    
    train_CNN.trainModel(train_dataset, learning_rate, n_iters, batch_size, False, "models", "CNN_1")
    test_CNN = Test(directory = "models", name = "CNN_1.pkl")
    test_dataset = dsets.MNIST(root = './data', train = False, transform = transforms.ToTensor())
    test_CNN.testModel(test_dataset, batch_size, False)