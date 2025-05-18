import flwr as fl

# Stratégie compatible Flower 1.10.0
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=1,
    min_available_clients=1
)

# Démarrage du serveur
fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
