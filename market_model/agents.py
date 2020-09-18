import copy
import torch
import numpy as np
from torch import nn

np.random.seed(42)


class Agent(object):
    def __init__(self, act_space, n_state, device):
        self.__state = None
        n_neurons = 16
        self.__action_space = act_space
        self._device = device

        # Define the neural network
        self.__base_network = nn.Sequential(nn.Linear(n_state, n_neurons),
                                            nn.PReLU(),
                                            nn.Linear(n_neurons, len(act_space)),
                                            nn.Softmax(dim=-1))
        pass

    def get_actions(self, state) -> dict:
        raise NotImplementedError

    def get_base_net(self):
        return self.__base_network

    def get_action_space(self):
        return self.__action_space


class Shell(Agent):
    def __init__(self, action_space, n_state, act_partners, device):
        super().__init__(action_space, n_state, device)
        self.__networks = dict()

        # initialize all necessary networks by copying the base network and send it to device
        for agt in act_partners:
            self.__networks.update({agt: copy.copy(super().get_base_net()).to(device)})

    def get_actions(self, state) -> dict:
        nets = self.__networks
        actions = dict()

        # derive an action for each network (i.e., policy)
        for key in nets.keys():
            # detach() should not be a problem hear, as for optimization predict() is called again,
            # where no detach() is used
            action_probs = self.predict(state, key).cpu().detach().numpy()
            actions.update({key: np.random.choice(super().get_action_space(), p=action_probs)})
        return actions

    def predict(self, state, partner_agt):
        # get the action probabilities as a tensor
        action_probs = self.__networks[partner_agt](torch.FloatTensor(state).to(self._device))
        return action_probs

    def get_networks(self):
        return self.__networks


class FSC(Agent):
    def __init__(self, action_space, n_state, act_partners, device):
        super().__init__(action_space, n_state, device)
        self.__networks = dict()
        self.__action_space = action_space

        # initialize all necessary networks by copying the base network and send it to device
        for agt in act_partners:
            self.__networks.update({agt: copy.copy(super().get_base_net()).to(device)})

    def get_actions(self, state) -> dict:
        nets = self.__networks
        actions = dict()

        # derive an action for each network (i.e., policy)
        for key in nets.keys():
            action_probs = self.predict(state, key).cpu().detach().numpy()
            actions.update({key: np.random.choice(super().get_action_space(), p=action_probs)})
        return actions

    def predict(self, state, partner_agt):
        # get the action probabilities as a tensor
        action_probs = self.__networks[partner_agt](torch.FloatTensor(state).to(self._device))
        return action_probs

    def get_networks(self):
        return self.__networks


class Gov(Agent):
    def __init__(self, action_space, n_state, activ_con, device):
        # super().__init__(action_space, n_state)
        pass

    def get_actions(self, state) -> dict:
        # government is performing maintain as action. That equals an passiv agent.
        return {'Shell': 0, 'FSC': 0}


def init_agents(act_space, n_states, change_all, act_agt, lr, device):
    optimizers = dict()
    all_agents = dict()
    for agt in act_agt:
        agt_optim = dict()
        all_agents.update({agt: eval(agt)(act_space, n_states[agt], change_all[agt], device)})
        # gov is passiv agent, thus need no optimizer
        if agt != 'Gov':
            networks = all_agents[agt].get_networks()
            for par_agt in networks.keys():
                agt_optim.update({par_agt: torch.optim.Adam(networks[par_agt].parameters(), lr=lr)})
            optimizers.update({agt: agt_optim})

    return optimizers, all_agents
