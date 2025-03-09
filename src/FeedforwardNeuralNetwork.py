import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=1, hidden_layers=2, neurons_per_layer=10, activation_fn=nn.ReLU):
        super().__init__()
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_size, neurons_per_layer))
            layers.append(activation_fn())
            input_size = neurons_per_layer
        layers.append(nn.Linear(neurons_per_layer, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralNetworkTrainer:
    def __init__(self, x_data, y_data, epochs=1000, lr=0.01):
        self.x_train = torch.tensor(x_data, dtype=torch.float32)
        self.y_train = torch.tensor(y_data, dtype=torch.float32)
        self.epochs = epochs
        self.lr = lr
        self.activation_functions = {
            'ReLU': nn.ReLU,
            'Sigmoid': nn.Sigmoid,
            'Tanh': nn.Tanh,
            'LeakyReLU': nn.LeakyReLU
        }

    def train_model(self, model):
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss_function = nn.MSELoss()
        losses = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            predictions = model(self.x_train)
            loss = loss_function(predictions, self.y_train)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                losses.append(loss.item())

        return losses, model

    def plot_training_loss(self, title, architectures):
        plt.figure(figsize=(10, 5))
        for hidden_layers, neurons in architectures:
            model = NeuralNetwork(hidden_layers=hidden_layers, neurons_per_layer=neurons)
            losses, _ = self.train_model(model)
            plt.plot(range(0, self.epochs, 100), losses, label=f'L{hidden_layers} N{neurons}')

        plt.xlabel('Эпохи')
        plt.ylabel('Loss')
        plt.title(f'Вплив архітектури на помилку для {title}')
        plt.legend()
        plt.show()

    def plot_activation_functions(self, title):
        plt.figure(figsize=(10, 5))
        for name, activation in self.activation_functions.items():
            model = NeuralNetwork(hidden_layers=2, neurons_per_layer=10, activation_fn=activation)
            losses, _ = self.train_model(model)
            plt.plot(range(0, self.epochs, 100), losses, label=name)

        plt.xlabel('Эпохи')
        plt.ylabel('Loss')
        plt.title(f'Вплив функціїї активації на помилку для {title}')
        plt.legend()
        plt.show()

    def plot_approximation(self, title, architectures):
        plt.figure(figsize=(10, 5))
        for hidden_layers, neurons in architectures:
            model = NeuralNetwork(hidden_layers=hidden_layers, neurons_per_layer=neurons)
            _, trained_model = self.train_model(model)
            with torch.no_grad():
                y_pred = trained_model(self.x_train).numpy()
            plt.plot(self.x_train, y_pred, label=f'L{hidden_layers} N{neurons}')

        plt.scatter(self.x_train, self.y_train, label='Ground Truth', color='black')
        plt.title(f'Вплив архітектури на апроксимацію {title}')
        plt.legend()
        plt.show()

    def plot_activation_approximation(self, title):
        plt.figure(figsize=(10, 5))
        for name, activation in self.activation_functions.items():
            model = NeuralNetwork(hidden_layers=2, neurons_per_layer=10, activation_fn=activation)
            _, trained_model = self.train_model(model)
            with torch.no_grad():
                y_pred = trained_model(self.x_train).numpy()
            plt.plot(self.x_train, y_pred, label=name)

        plt.scatter(self.x_train, self.y_train, label='Ground Truth', color='black')
        plt.title(f'Вплив функціїї активації на апроксимацію {title}')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    x = np.linspace(-2, 2, 100).reshape(-1, 1)
    y1 = x ** 2
    y2 = x ** 3 + 2 * x

    trainer1 = NeuralNetworkTrainer(x, y1)
    trainer2 = NeuralNetworkTrainer(x, y2)

    architectures = [(1, 5), (1, 10), (1, 20), (2, 5), (2, 10), (2, 20), (3, 5), (3, 10), (3, 20)]

    trainer1.plot_training_loss('f(x) = x^2', architectures)
    trainer2.plot_training_loss('f(x) = x^3 + 2x', architectures)

    trainer1.plot_activation_functions('f(x) = x^2')
    trainer2.plot_activation_functions('f(x) = x^3 + 2x')

    trainer1.plot_approximation('f(x) = x^2', architectures)
    trainer2.plot_approximation('f(x) = x^3 + 2x', architectures)

    trainer1.plot_activation_approximation('f(x) = x^2')
    trainer2.plot_activation_approximation('f(x) = x^3 + 2x')
