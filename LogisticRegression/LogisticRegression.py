import torch

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from os.path import join

class LogisticRegressionModel(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)  
    
    def forward(self, x):
        return self.linear(x)

class Train(LogisticRegressionModel):

    def __init__(self, input_size, num_classes):
        super().__init__(input_size, num_classes)
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = LogisticRegressionModel(self.input_size, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def optimizer(self, learning_rate):
        return torch.optim.SGD(self.model.parameters(), lr = learning_rate)

    def epochs(self, iterations, batch_size):
        return int(n_iters/(len(train_dataset)/batch_size))

    def trainModel(self, train_data, learning_rate, n_iters, batch_size, GPU, directory, name):
        print("Training Model")
        iteration = 0
        epochs = self.epochs(n_iters, batch_size)
        optimizer = self.optimizer(learning_rate)
        train_data = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

        if GPU != False:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            device = self.model.to(device)

        for epoch in range(epochs):
            for images, labels in train_data:
                
                if GPU != False:
                    images = images.view(-1, self.input_size).requires_grad_().to(device)
                    labels = labels.to(device)
                else:
                    images = images.view(-1, self.input_size).requires_grad_()

                optimizer.zero_grad() 
                images = self.model(images)
                loss = self.criterion(images, labels)
                loss.backward()
                optimizer.step()
                iteration += 1
            print("Epoch: {}, Loss: {}".format(epoch, loss))
        
        torch.save(self.model.state_dict(), join(directory, name+".pkl"))

class Test(LogisticRegressionModel):

    def __init__(self, input_size, num_classes, directory, name):
        super().__init__(input_size, num_classes)
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = LogisticRegressionModel(self.input_size, self.num_classes)
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
                images = images.view(-1, self.input_size).requires_grad_().to(device)
            else:
                images = images.view(-1, self.input_size).requires_grad_()

            images = self.model(images)
            _, predicted = torch.max(images.data, 1)
            total += labels.size(0)

            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()
    
        accuracy = 100 * correct / total #Defintely not the right way to measure accuracy. 

        print("Accuracy: {}".format(accuracy))

if __name__ == "__main__":

    n_iters = 5000
    batch_size = 100
    learning_rate = 0.001
    train_logistic_regress = Train(input_size = 28*28, num_classes = 10)
    train_dataset = dsets.MNIST(root = "./data", train = True, transform = transforms.ToTensor(), download = True)
    epochs = int(n_iters/(len(train_dataset)/batch_size))
    train_logistic_regress.trainModel(train_dataset, learning_rate, n_iters, batch_size, False, "models", "LR_1")
    test_logistic_regress = Test(input_size = 28*28, num_classes = 10, directory = "./models", name = "LR_1.pkl")
    test_dataset = dsets.MNIST(root = './data', train = False, transform = transforms.ToTensor())
    test_logistic_regress.testModel(test_dataset, batch_size, False)