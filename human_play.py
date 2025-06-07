import numpy as np
import torch

from game import ConnectFour
from mcts import MCTS
from model import InferenceGraph, Net

c_puct = 1.1
c_fpu = 0.2
n_playout = 160
model_path = './katac4_b3c128nbt_30000.pth'

device = torch.device('cuda')
net = Net().to(device)
net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
net.eval()

game = ConnectFour(9, 9, (0, 0))
graph = InferenceGraph(net, device, game.height, game.width)

policy_value_fn = graph.policy_value_fn
mcts = MCTS(policy_value_fn, exploration=False, c_puct=c_puct, c_fpu=c_fpu, n_playout=n_playout)

while not game.is_terminal():
    print()
    print(game)
    moves = game.sensible_moves()
    print('Available columns:', moves)
    while True:
        try:
            col = int(input('Your move> '))
        except ValueError:
            continue
        if col not in moves:
            continue
        break
    game.step(col)
    mcts.apply_move(col)
    print()
    print(game)
    if game.is_terminal():
        break
    acts, probs = mcts.get_move_probs(game)
    winrate = (1 + mcts.root.Q) / 2
    print(f'N={mcts.root.N} Q={mcts.root.Q:.3f} {winrate:.1%}')
    ai_move = acts[np.argmax(probs)]
    print('AI move:', ai_move)
    print('MCTS policy:', *(f'{p:.0%}' for p in probs))
    _, nn_policy, _ = policy_value_fn(game, 1.0)
    print('NN policy:', *(f'{p:.0%}' for p in nn_policy))
    game.step(ai_move)
    mcts.apply_move(ai_move)

print('--- GAME OVER ---')
print(game)
if game.winner == 1:
    print('You win!')
elif game.winner == -1:
    print('AI wins!')
else:
    print('Draw!')
