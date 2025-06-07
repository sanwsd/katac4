import matplotlib.pyplot as plt
import json

with open('elo.json', 'r', encoding='utf-8') as file:
    elos = {int(k): v for k, v in json.load(file).items()}

players = sorted(elos.keys())
ratings = [elos[player] for player in players]
plt.plot(players, ratings, marker='o')
plt.xlabel('Epochs')
plt.ylabel('ELO Rating')
plt.title('ELO Rating Over Time')
plt.grid(True)
plt.show()
