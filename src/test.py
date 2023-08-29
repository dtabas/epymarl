import gym
import gridworld

# First approach:
# from gridworld.scenarios.test_example import make_env
# make_env()

# Second approach:
# from gridworld.scenarios import test_example
# gym.make("MultiBuilding-v0")

# Third approach:
gym.make("gridworld.scenarios.test_example:MultiBuilding-v0")
# ^ THIS is the key that you have to use when running epymarl from the command line.

print('done')