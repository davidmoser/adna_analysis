import numpy as np
from re import L
from matplotlib import pyplot as plt
import numpy as np

from scripts.train_genonet import train_genonet

# Hyperparameters
batch_size = 128
hidden_dim, hidden_layers = 150, 20
epochs = 30

learning_rates = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1]

last_losses = {"constant": [], "cosine": []}
for schedule in ["constant", "cosine"]:
    for learning_rate in learning_rates:
        _, train_losses, _ = train_genonet(epochs, learning_rate, batch_size, hidden_dim, hidden_layers, schedule=schedule, small=True)
        last_losses[schedule].append(train_losses[-1])

title = f'Loss after {epochs} epochs: Learning rates: 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, Dimension: {hidden_dim}, Layers: {hidden_layers}, Batch size: {batch_size}'

plt.figure(figsize=(10, 5))
plt.plot(learning_rates, np.log10(last_losses["constant"]), label="constant")
plt.plot(learning_rates, np.log10(last_losses["cosine"]), label="cosine")
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Log Loss')
plt.title(title)
plt.legend()
plt.show()
