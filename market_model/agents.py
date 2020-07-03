import copy
import torch
import numpy as np
from torch import nn

np.random.seed(42)


class Agent(object):
    def __init__(self, act_space, n_state):
        self.__state = None
        self.__n_neurons = 16
        # TODO: nicht benötigte vars löschen und direkt übergeben
        self.__n_state_space = n_state
        self.__action_space = act_space

        # Define network
        self.__base_network = nn.Sequential(nn.Linear(self.__n_state_space, self.__n_neurons),
                                            nn.ReLU(),
                                            nn.Linear(self.__n_neurons, len(act_space)),
                                            nn.Softmax(dim=-1))

    def get_actions(self, state) -> dict:
        raise NotImplementedError

    def get_base_net(self):
        return self.__base_network

    def get_action_space(self):
        return self.__action_space


class Shell(Agent):
    def __init__(self, action_space, n_state, act_partners):
        super().__init__(action_space, n_state)
        self.__networks = dict()

        # initialize all necessary networks by copying the base network
        for agt in act_partners:
            self.__networks.update({agt: copy.copy(super().get_base_net())})

    def get_actions(self, state) -> dict:
        nets = self.__networks
        actions = dict()
        for key in nets.keys():
            action_probs = self.predict(state, key)
            actions.update({key: np.random.choice(super().get_action_space(), p=action_probs)})
        return actions

    def predict(self, state, partner_agt):
        action_probs = self.__networks[partner_agt](torch.FloatTensor(state)).detach().numpy()
        return action_probs

    def get_networks(self):
        return self.__networks


class FSC(Agent):
    def __init__(self, action_space, n_state, act_partners):
        super().__init__(action_space, n_state)
        self.__networks = dict()
        self.__action_space = action_space

        # initialize all necessary networks by copying the base network
        for agt in act_partners:
            self.__networks.update({agt: copy.copy(super().get_base_net())})

    def get_actions(self, state) -> dict:
        nets = self.__networks
        actions = dict()
        for key in nets.keys():
            action_probs = self.predict(state, key)
            actions.update({key: np.random.choice(super().get_action_space(), p=action_probs)})
        return actions

    def predict(self, state, partner_agt):
        action_probs = self.__networks[partner_agt](torch.FloatTensor(state)).detach().numpy()
        return action_probs

    def get_networks(self):
        return self.__networks


class Gov(Agent):
    def __init__(self, action_space, n_state, activ_con):
        # super().__init__(action_space, n_state)
        pass

    def get_actions(self, state) -> dict:
        # government is performing maintain as action. That equals an passiv agent.
        return {'Shell': 0, 'FSC': 0}
