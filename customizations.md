# Command line statements (for Mac):

## Initialization:

conda activate MARL_env

cd /Users/danieltabas/Documents/GitHub/epymarl

## Running: 

### Simple 1 player:
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:Simple-v0"

### Simple 2 player:
python3 src/main.py --config=coma_ns --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:Simple-2p-v0"

### Simple 2 player with primal dual algorithm and customized critic (MADDPG):
python3 src/main.py --config=maddpg_pd_ns --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:Simple-2p-v0"

# Other modifications:

### Register new gym environment:
multiagent-particle-envs/mpe/_\_init__.py

### To modify t_max:
For qmix, modify epymarl/src/config/envs/gymma.yaml

For coma, modify epymarl/src/config/algs/coma.yaml

### To toggle rendering:
If training is episodic, modify epymarl/src/runners/episode_runner.py line 80 (episodic vs parallel training is specified in one of the config files, either in config/envs or config/algs)

### Using tensorboard:

use_tensorboard: True (in config file)

In command line: tensorboard --logdir results/tb_logs/