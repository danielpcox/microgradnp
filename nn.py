import numpy as np
from engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Value(np.random.randn(in_features, out_features) * 0.01, _op=f'linW:{in_features}->{out_features}')
        self.b = Value(np.zeros((1, out_features)), _op=f'linb:{in_features}->{out_features}')

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()





#------------------------------------------------------------

if __name__ == '__main__':

    # Define dataset and model
    n_samples = 100
    input_features = 5
    output_features = 3
    hidden_dim = 10
    n_epochs = 200
    learning_rate = 0.01

    X = np.random.randn(n_samples, input_features)
    Y = np.random.randn(n_samples, output_features)

    model = Sequential(Linear(input_features, hidden_dim),
                       ReLU(),
                       Linear(hidden_dim, output_features))

    def mse_loss(y_pred, y_true):
        diff = y_pred - y_true
        squared_diff = diff**2
        loss = squared_diff.mean()
        return loss

    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for x, y in zip(X, Y):
            x_value = Value(x.reshape(1, -1))
            y_value = Value(y.reshape(1, -1))

            model.zero_grad()

            y_pred = model.forward(x_value)
            loss = mse_loss(y_pred, y_value)
            epoch_loss += loss.data

            loss.backward()

            for param in model.parameters():
                # print('Param:', param._op, param.grad)
                param.data -= learning_rate * param.grad

        epoch_loss /= n_samples
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss}")
