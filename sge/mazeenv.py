import os
import sys
from .graph import SubtaskGraph
from sge.mazemap import Mazemap
import numpy as np
from .utils import get_id_from_ind_multihot
from sge.utils import WHITE, BLACK, DARK, LIGHT, GREEN, DARK_RED


class MazeEnv(object):  # single batch
    def __init__(self, game_name, graph_param, game_len, gamma, render_config={}):
        if game_name == 'playground':
            from sge.playground import Playground
            game_config = Playground()
            graph_folder = os.path.join('.', 'data', 'subtask_graph_play')
            filename = 'play_{param}'.format(param=graph_param)

        elif game_name == 'mining':
            from sge.mining import Mining
            game_config = Mining()
            graph_folder = os.path.join('.', 'data', 'subtask_graph_mining')
            filename = 'mining_{param}'.format(param=graph_param)

        self.config = game_config
        self.max_task = self.config.nb_subtask_type
        self.subtask_list = self.config.subtask_list
        if len(render_config)>0:
            self.rendering = render_config['vis']
        else:
            self.rendering = False

        # graph & map
        self.graph = SubtaskGraph(
            graph_folder, filename, self.max_task)  # just load all graph
        self.map = Mazemap(game_name, game_config, render_config)
        self.gamma = gamma

        # init
        self.game_length = int(np.random.uniform(
            0.8, 1.2) * game_len)
        self.step_reward = 0.0

    def step(self, action):
        if self.graph.graph_index is None:
            raise RuntimeError('Error: Environment has never been reset()')
        sub_id = -1
        if self.game_over or self.time_over:
            raise ValueError(
                'Environment has already been terminated. need to be reset!')
        oid = self.map.act(action)
        if (action, oid) in self.config.subtask_param_to_id:  # if (action, item) is one of the subtasks
            sid = self.config.subtask_param_to_id[(action, oid)]

            if sid in self.subtask_id_list:  # if sub_id is in the subtask graph
                sub_id = sid
            else:
                #print('Warning! Executed a non-existing subtask')
                pass
        #
        self.reward = self._act_subtask(sub_id)
        self.ret += self.reward*self.gamma
        self.step_count += 1
        self.time_over = self.step_count >= self.game_length
        self.game_over = (self.eligibility*self.mask).sum().item() == 0
        if self.rendering:
            self.render()

        return self._get_state(), self.reward, (self.game_over or self.time_over), self._get_info()

    def reset(self, seed=None, graph_index=None):  # after every episode
        if seed is not None:
            np.random.seed(seed)
        if graph_index is None:
            graph_index = np.random.permutation(self.graph.num_graph)[0]
        else:
            graph_index = graph_index % self.graph.num_graph

        # 1. reset graph
        if graph_index >= 0:
            self.graph.set_graph_index(graph_index)
            self.nb_subtask = len(self.graph.subtask_id_list)
            self.rew_mag = self.graph.rew_mag
            self.subtask_id_list = self.graph.subtask_id_list

        # 2. reset subtask status
        self.executed_sub_ind = -1
        self.game_over = False
        self.time_over = False
        self.mask, self.mask_id = np.ones(
            self.nb_subtask, dtype=np.uint8), np.zeros(self.max_task, dtype=np.uint8)
        for ind, sub_id in self.graph.ind_to_id.items():
            self.mask_id[sub_id] = 1
        self.completion, self.comp_id = np.zeros(
            self.nb_subtask, dtype=np.int8), np.zeros(self.max_task, dtype=np.uint8)
        self._compute_elig()
        self.step_count, self.ret, self.reward = 0, 0, 0

        # 3. reset map
        self.map.reset(self.subtask_id_list)
        if self.rendering:
            self.render()
        return self._get_state(), self._get_info()

    def state_spec(self):
        return [
            {'dtype': self.map.get_obs().dtype, 'name': 'observation', 'shape': self.map.get_obs().shape},
            {'dtype': self.mask_id.dtype, 'name': 'mask', 'shape': self.mask_id.shape},
            {'dtype': self.comp_id.dtype, 'name': 'completion', 'shape': self.comp_id.shape},
            {'dtype': self.elig_id.dtype, 'name': 'eligibility', 'shape': self.elig_id.shape},
            {'dtype': int, 'name': 'step', 'shape': ()}
        ]

    def get_actions(self):
        return self.config.legal_actions
    
    def render(self):
        GREEN = "#50a000"
        DARK_RED = "#a30000"
        LIGHT = "#c0c0c0"  # light gray
        color = []
        for ind, sub_id in enumerate(self.graph.subtask_id_list):
            if self.completion[ind] == 1:  # success
                color.append(GREEN)  # Green
            elif self.mask[ind] == 0:  # fail
                color.append(DARK_RED)
            elif self.eligibility[ind] == 0:  # inelig
                color.append(LIGHT)
            else:  # elig
                color.append('white')
        self.graph.draw_graph(self.config, self.rew_mag, color)
        text_lines, text_widths, status, bg_colors = self._get_status()
        self.map.render(self.step_count, text_lines,
                        text_widths, status, bg_colors)

    # internal
    def _get_state(self):
        step = self.game_length - self.step_count
        return {
            'observation': self.map.get_obs(),
            'mask': self.mask_id.astype(np.float),
            'completion': self.comp_id.astype(np.float),
            'eligibility': self.elig_id.astype(np.float),
            'step': step
        }
    
    def _get_info(self):
        return {
            'graph': self.graph
        }

    def _act_subtask(self, sub_id):
        self.executed_sub_ind = -1
        reward = self.step_reward
        if sub_id < 0:
            return reward
        sub_ind = self.graph.id_to_ind[sub_id]
        if self.eligibility[sub_ind] == 1 and self.mask[sub_ind] == 1:
            self.completion[sub_ind] = 1
            self.comp_id[sub_id] = 1
            reward += self.rew_mag[sub_ind]
            self.executed_sub_ind = sub_ind
        self.mask[sub_ind] = 0
        self.mask_id[sub_id] = 0

        self._compute_elig()

        return reward

    def _compute_elig(self):
        self.eligibility = self.graph.get_elig(self.completion)
        self.elig_id = get_id_from_ind_multihot(
            self.eligibility, self.graph.ind_to_id, self.max_task)

    # rendering
    def _get_status(self, reward=None):
        text_lines, bg_colors = [], []
        text_lines.append(['      Name', ' Obj', '+', 'Key'])
        bg_colors.append(WHITE)

        for ind, sub_id in enumerate(self.graph.subtask_id_list):
            (action, oid) = self.config.subtask_param_list[sub_id]
            action_key = self.config.operation_list[action]['key']
            if 'name' in self.config.subtask_list[sub_id]:
                sub_name = self.config.subtask_list[sub_id]['name']
            else:
                sub_name = self.action_meanings[action]+' ' + \
                    self.config.object_param_list[oid]['name']
            if self.completion[ind] == 1:  # success
                color = GREEN
            elif self.mask[ind] == 0:  # fail
                color = DARK_RED
            elif self.eligibility[ind] == 0:  # inelig
                color = LIGHT
            else:  # elig
                color = WHITE
            text_lines.append(
                [sub_name, oid, '+', ' '+action_key])
            bg_colors.append(color)

        # text widths
        text_widths = [0]*4
        for _, line in enumerate(text_lines):
            for nn, item in enumerate(line):
                if type(item) == str:
                    text_widths[nn] = max(text_widths[nn], len(item)+2)

        fmt = "Step={:02d}/{:02d} | Return={:+2.02f} | Reward={:+1.02f}"
        status = fmt.format(
            self.step_count, self.game_length, self.ret, self.reward)
        return text_lines, text_widths, status, bg_colors
