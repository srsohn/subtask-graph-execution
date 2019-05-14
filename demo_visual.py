import numpy as np
import os
import sys
import pygame
from sge.mazeenv import MazeEnv
from sge.utils import KEY

###################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default='mining',
                        help='MazeEnv/config/%s.lua')
    parser.add_argument('--graph_param', default='train_1',
                        help='difficulty of subtask graph')
    parser.add_argument('--game_len', default=70,
                        type=int, help='episode length')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='discount factor')
    args = parser.parse_args()

    render_config = {
        'vis': True,
        'save': True,
        'key_cheatsheet': False
    }
    env = MazeEnv(args.game_name, args.graph_param,
                  args.game_len, args.gamma, render_config)

    env.reset(args.seed)
    while not env.game_over and not env.time_over:
        action = None
        events = pygame.event.get()  # handle keyboard/mouse input
        for event in events:
            if event.type == pygame.KEYDOWN:
                action = {
                    pygame.K_UP: KEY.UP,
                    pygame.K_DOWN: KEY.DOWN,
                    pygame.K_LEFT: KEY.LEFT,
                    pygame.K_RIGHT: KEY.RIGHT,
                    pygame.K_p: KEY.PICKUP,
                    pygame.K_t: KEY.TRANSFORM,
                    pygame.K_1: KEY.USE_1,
                    pygame.K_2: KEY.USE_2,
                    pygame.K_3: KEY.USE_3,
                    pygame.K_4: KEY.USE_4,
                    pygame.K_5: KEY.USE_5,
                    pygame.K_q: KEY.QUIT
                }.get(event.key, None)
            if action is not None:
                break
        if action is None:
            continue
        if action == KEY.QUIT:
            break
        state, rew, done, info = env.step(action)
        pygame.time.wait(100)
