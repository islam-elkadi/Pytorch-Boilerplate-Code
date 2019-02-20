import torch

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from os.path import join

class FeedforwardNeuralNetModel(nn.Module):

    #2 Hidden Layers

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()
        
        # Linear function 3 (readout): 100 --> 10
        self.fc3 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)
        
        # Linear function 3 (readout)
        return self.fc3(out)

class Train(FeedforwardNeuralNetModel):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = FeedforwardNeuralNetModel(self.input_dim, self.hidden_dim, self.output_dim)
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
                    images = images.view(-1, self.input_dim).requires_grad_().to(device)
                    labels = labels.to(device)
                else:
                    images = images.view(-1, self.input_dim).requires_grad_()

                optimizer.zero_grad() 
                images = self.model(images)
                loss = self.criterion(images, labels)
                loss.backward()
                optimizer.step()
            print("Epoch: {}, Loss: {}".format(epoch, loss))
        
        torch.save(self.model.state_dict(), join(directory, name+".pkl"))

class Test(FeedforwardNeuralNetModel):

    def __init__(self, input_dim, hidden_dim, output_dim, directory, name):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = FeedforwardNeuralNetModel(self.input_dim, self.hidden_dim, self.output_dim)
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
                images = images.view(-1, self.input_dim).requires_grad_().to(device)
            else:
                images = images.view(-1, self.input_dim).requires_grad_()

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
    train_FF_NN = Train(input_dim = 28*28, hidden_dim = 100, output_dim = 10)
    train_dataset = dsets.MNIST(root = "./data", train = True, transform = transforms.ToTensor(), download = True)
    epochs = int(n_iters/(len(train_dataset)/batch_size))
    
    train_FF_NN.trainModel(train_dataset, learning_rate, n_iters, batch_size, False, "models", "FF_NN_1")
    test_FF_NN = Test(input_dim = 28*28, hidden_dim = 100, output_dim = 10, directory = "models", name = "FF_NN_1.pkl")
    test_dataset = dsets.MNIST(root = './data', train = False, transform = transforms.ToTensor())
    test_FF_NN.testModel(test_dataset, batch_size, False)