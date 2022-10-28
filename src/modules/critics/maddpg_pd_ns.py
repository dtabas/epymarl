import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.critics.mlp import MLP


class MADDPGPDCriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGPDCriticNS, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions * self.n_agents
        self.n_constraints = scheme["cost"]["vshape"][0]
        self.input_multipliers = False
        if self.input_multipliers: self.input_shape +=  self.n_constraints# Include multipliers in the input to the critic neural network
        if self.args.obs_last_action:
            self.input_shape += self.n_actions
        self.output_type = "q"
        self.n_outputs = 1 + self.n_constraints
        self.critics = [MLP(self.input_shape, self.args.hidden_dim, self.n_outputs) for _ in range(self.n_agents)]

    def forward(self, inputs, actions):
        inputs = th.cat((inputs, actions), dim=-1) # dimensions: batch, max_t, n_agents, n_inputs
        qs = []
        for i in range(self.n_agents):
            q = self.critics[i](inputs[:, :, i]).unsqueeze(2)
            qs.append(q)
        return th.cat(qs, dim=2) # dimensions: batch, max_t, n_agents, n_outputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        return input_shape

    def parameters(self):
        params = list(self.critics[0].parameters())
        for i in range(1, self.n_agents):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, c in enumerate(self.critics):
            c.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()
