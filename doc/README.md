# Example
This is an example of running random agent in Mining environment for 1 episode:
```
from sge import MazeEnv
import random

env = MazeEnv(game_name='mining', graph_param='train_1', game_len=70, gamma=0.99)

env.reset(1)
action_set = env.get_actions()
step, done = 0, False
while not done:
    action = random.sample(list(action_set), 1)[0]

    state, rew, done, info = env.step(action)

    string = 'Step={:02d}, Action={}, Reward={:.2f}, Done={}'
    print(string.format(step, action, rew, done))
    step += 1
```
## API

### class `sge.MazeEnv`(*game_name*, *graph_param*, *game_len*, *gamma*, *render_config=None*)

Creates an environment object of either Mining or Playground depending on the *game_name*. It load the pre-generated subtask graph set file *graph_param*. It sets the length of episode as *game_len* steps and discount factor as *gamma*. The *render_config* is a dictionary of the followings:
* 'vis': visualization flag. Visualizing the environment if True.
* 'save': saving flag. The visual observation is saved in the 'sge/render' folder if True.
* 'key_cheatsheet': The cheatsheet of how to execute each subtask using keyboard is visualized if True.

### *state*, *reward*, *done*, *info* = `step`(*action*)

Step forward the environment, executing the action defined by *action*. Returns the *state*, *reward*, *done*, and *info*.
* *state* is a dictionary of the followings:

| key          | Description                                                | shape         |  type       |
| -------------| -----------------------------------------------------------| --------------| ---------   |
| `observation`| the encoded map of 10x10 grid world                        | #objectx10x10 | numpy.uint8 |
| `mask`       | subtask mask vector. 1 if never executed                   | #subtasks     | numpy.float |
| `eligibility`| subtask eligibility vector. 1 if eligible                  | #subtasks     | numpy.float |
| `completion` | subtask completion vector. 1 if completed                  | #subtasks     | numpy.float |
| `step`       | number of remaining step in the episode                    | scalar        | int         |
    
* *info* is a dictionary of the following:
    * 'graph': the subtask graph used to generate current environment.

### `state_spec`()

Returns a list specifying the available observations.

### `get_actions`()

Returns a set of actions that agent can take in the current domain.

### *state*, *info* = `reset`(*seed=None*, *graph_index=None*)

Resets the environment to its initial state. This method needs to be called to start a new episode after the last episode ended. Returns the initial *state* and *info*.
* *seed* (optional) is the random seed of environment (Playground domain is stochastic).
* *graph_index* (optional) is the index of the subtask graph to be used for generating environment. 
