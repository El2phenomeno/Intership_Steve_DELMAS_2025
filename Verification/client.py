import flwr as fl
import numpy as np

class SimpleClient(fl.client.NumPyClient):
    def __init__(self):
        self.weights = np.array([1.0, 2.0])

    def get_parameters(self, config):
        return [self.weights]

    def fit(self, parameters, config):
        # Simule un petit "entraînement"
        print(f"Client received weights: {parameters}")
        self.weights += 1.0  # simulate update
        print(f"Client updated weights to: {self.weights}")
        return [self.weights], 1, {}

    def evaluate(self, parameters, config):
        print("Evaluating with:", parameters)
        loss = float(np.sum(parameters[0]) * 0.01)  # simuler une perte
        return loss, 1, {}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=SimpleClient())
