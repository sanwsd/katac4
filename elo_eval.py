import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing as mp

from game import ConnectFour
from mcts import MCTS
from model import Net, InferenceGraph


def random_policy_value_fn(game, *args):
    moves = game.sensible_moves()
    logits = np.random.randn(len(moves)) * 0.4
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    return moves, probs, 0.0

residual_blocks = 6
residual_channels = 96

def get_policy_value_fn(model_path, game, device):
    if not model_path:
        return random_policy_value_fn
    net = Net()
    net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    net.eval()
    graph = InferenceGraph(net, device, game.height, game.width)
    return graph.policy_value_fn

c_puct = 1.1
c_fpu = 0.2
n_playout = 200
ai_list = list(range(0, 30001, 500))
weight_format = './weights/b3c128nbt_2025-05-24_20-47-22/katac4_b3c128nbt_{it}.pth'

def apply_temperature(probs, temp):
    logits = np.log(np.clip(probs, 1e-10, None)) / temp
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def one_game(game, mcts1, mcts2):
    step = 0
    board_size = game.height * game.width
    while not game.is_terminal():
        temp = max(1.03, 1.35 * pow(0.66, step / board_size))
        mcts = mcts1 if game.player == 1 else mcts2
        acts, probs = mcts.get_move_probs(game, root_prior_temp=temp)
        act_temp = 0.5 * pow(0.8, step / game.width)
        move = np.random.choice(acts, p=apply_temperature(probs, act_temp))
        game.step(move)
        mcts1.apply_move(move)
        mcts2.apply_move(move)
        step += 1
    return game.winner

kwargs = {
    'exploration': False,
    'c_puct': c_puct,
    'c_fpu': c_fpu,
    'n_playout': n_playout
}

def one_test(model1_path, model2_path, device):
    game1 = ConnectFour()
    f1 = get_policy_value_fn(model1_path, game1, device)
    f2 = get_policy_value_fn(model2_path, game1, device)
    game2 = deepcopy(game1)
    mcts1 = MCTS(f1, **kwargs)
    mcts2 = MCTS(f2, **kwargs)
    result1 = one_game(game1, mcts1, mcts2)
    mcts1.reset()
    mcts2.reset()
    result2 = one_game(game2, mcts2, mcts1)
    return result1, -result2

def get_model_path(it):
    if it == 0:
        return None
    return weight_format.format(it=it)

def elo_worker(device, result_queue):
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    while True:
        p1, p2 = random.sample(ai_list, 2)
        path1 = get_model_path(p1)
        path2 = get_model_path(p2)
        result = one_test(path1, path2, device)
        result_queue.put((p1, p2, (sum(result) + 2) / 4))

K = 32
def expected_score(elo1, elo2):
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))

def elo_update(elo1, elo2, outcome):
    elo1 += K * (outcome - expected_score(elo1, elo2))
    elo2 += K * ((1 - outcome) - expected_score(elo2, elo1))
    return elo1, elo2

def main():
    global K
    num_gpus = torch.cuda.device_count()
    num_workers = 4 * num_gpus
    save_filename = 'elo.json'
    save_freq = 16
    print(f'{num_workers} workers on {num_gpus} GPUs')
    elos = {ai: 0 for ai in ai_list}
    if os.path.exists(save_filename):
        print('Initializing with existing ELO ratings')
        with open(save_filename, 'r', encoding='utf-8') as file:
            elos = {int(k): v for k, v in json.load(file).items()}
    result_queue = mp.Queue()
    workers = [mp.Process(target=elo_worker, args=(f'cuda:{i % num_gpus}', result_queue))
               for i in range(num_workers)]
    for worker in workers:
        worker.start()
    game_count = 0
    while True:
        p1, p2, result = result_queue.get()
        elos[p1], elos[p2] = elo_update(elos[p1], elos[p2], result)
        game_count += 1
        if game_count % save_freq == 0:
            K *= 0.9993
            benchmark = elos[0]
            for player in elos:
                elos[player] -= benchmark
            print(f'Games played: {game_count}, K: {K:.3f}')
            with open(save_filename, 'w', encoding='utf-8') as file:
                json.dump(elos, file)

if __name__ == '__main__':
    main()
