from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        
        self.lam_init = 1e3
        self.lam = np.ones(self.env.n_constraints)* self.lam_init
        self.lam_max = 1e4

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = np.array([0.]*self.env.n_agents)
        original_rewards = []
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward_cost, terminated, env_info = self.env.step(actions[0])
            
            reward = np.array([r - np.dot(self.lam,c) for r,c in reward_cost]).flatten()
            cost = [c for r,c in reward_cost]
            cost = cost[0] # costs are the same for each agent
            episode_return += reward
            original_reward = np.array([r for r,c in reward_cost]).flatten()
            original_rewards.append(original_reward)
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "cost": [(cost,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)
            
            
            rendering = False
            if rendering:
                if self.train_stats.get("n_episodes", 0) % 50 == 0:
                    self.env.render()
            
            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            "lam": self.lam
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        extra_stats = {}
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            
        if not test_mode:
            # Dual update:
            constraint_type = self.args.env_args['constraint_type']
            episode_cost = self.batch["cost"][0]
            T = episode_cost.size()[0]
            weight = [(1-self.args.gamma)/(1-self.args.gamma**T) * self.args.gamma**t for t in range(T)]
            
            if constraint_type == "expectation":
                LHS_tolerance = 0
                RHS_tolerance = 0
                step_size = .1
                modified_cost = episode_cost
            elif constraint_type == "CVaR":
                LHS_tolerance = 0.2
                RHS_tolerance = 0.005
                step_size = .0001
                modified_cost = np.maximum(episode_cost-LHS_tolerance,0)
            elif constraint_type == "chance":
                LHS_tolerance = 0.1
                RHS_tolerance = 0.1
                step_size = .0002
                modified_cost = (episode_cost > LHS_tolerance)*1.0
            
            discounted_cost = np.dot(weight,episode_cost)
            lam_grad = np.dot(weight,modified_cost - RHS_tolerance)
            self.lam += step_size*lam_grad
            self.lam = np.maximum(self.lam,0)
            self.lam = np.minimum(self.lam,self.lam_max)
            extra_stats["cost"] = np.sum(discounted_cost)
            extra_stats["lam"] = np.sum(self.lam)
            
            original_rewards = np.array(original_rewards)
            original_returns = np.dot(weight[:-1],original_rewards)
            extra_stats["agent1_return"] = original_returns[0]
            extra_stats["agent2_return"] = original_returns[1]
            extra_stats["agent3_return"] = original_returns[2]
            
        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            self._log2(extra_stats)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def _log2(self,stats):
        for k,v in stats.items():
            self.logger.log_stat(k,v,self.t_env)
        stats.clear()