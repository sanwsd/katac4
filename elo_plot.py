import matplotlib.pyplot as plt
import json

run = 'b3c128nbt_2025-05-24_20-47-22'

with open('elo.json', 'r', encoding='utf-8') as file:
    elos = {int(k): v for k, v in json.load(file).items()}

players = sorted(elos.keys())
ratings = [elos[player] for player in players]

plt.plot(players, ratings, marker='o', label=run)

max_rating = max(ratings)
max_player = players[ratings.index(max_rating)]
plt.plot(max_player, max_rating, marker='o', color='red', markersize=6, label=f'Best checkpoint ({max_player}, {max_rating:.0f})')

plt.xlabel('Epochs')
plt.ylabel('ELO Rating')
plt.title('ELO Rating Over Time')
plt.grid(True)
plt.legend()
plt.show()
