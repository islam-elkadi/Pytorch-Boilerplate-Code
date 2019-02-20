import torch

import numpy as np
import torch.nn as nn

from os.path import join

class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)  
    
    def forward(self, x):
        return self.linear(x)


class Train(LinearRegressionModel):

	def __init__(self, input_dim, output_dim):
		super().__init__(input_dim, output_dim)
		self.model = LinearRegressionModel(input_dim, output_dim)
		self.criterion = nn.MSELoss()

	def optimizer(self, learning_rate):
		return torch.optim.SGD(self.model.parameters(), lr = learning_rate)

	def trainModel(self, train_data, train_labels, epochs, learning_rate, directory, name, GPU = False):
		optimize = self.optimizer(learning_rate)

		if GPU != False:
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			device = self.model.to(device)

		for epoch in range(epochs):
			if GPU != False:
				inputs = torch.from_numpy(x_train).to(device)
				labels = torch.from_numpy(y_train).to(device)
			else:
				inputs = torch.from_numpy(x_train).requires_grad_()
				labels = torch.from_numpy(y_train)

			optimize.zero_grad() 
			outputs = self.model(inputs)
			loss = self.criterion(outputs, labels)
			loss.backward()
			optimize.step()
			print("epoch {}, loss {}".format(epoch, loss.item()))

		torch.save(self.model.state_dict(), join(directory, name+".pkl"))


if __name__ == "__main__":

	train_model = Train(1, 1)

	x_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	x_train = np.array(x_values, dtype = np.float32)
	x_train = x_train.reshape(-1, 1)

	y_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
	y_train = np.array(y_values, dtype = np.float32)
	y_train = y_train.reshape(-1, 1)

	train_model.trainModel(x_train, y_train, 100, 0.01, "models", "LR_1", True)