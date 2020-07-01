import copy
from torch import nn


class Agent(object):
    def __init__(self, n_action, n_state):
        self.__state = None
        self.__n_neurons = 16
        # TODO: nicht benötigte vars löschen und direkt übergeben
        self.__n_action_space = n_action
        self.__n_state_space = n_state

        # Define network
        self.__base_network = nn.Sequential(nn.Linear(self.__n_state_space, self.__n_neurons),
                                            nn.ReLU(),
                                            nn.Linear(self.__n_neurons, self.__n_action_space),
                                            nn.Softmax(dim=-1))

    def get_action(self):
        pass

    def get_base_net(self):
        return self.__base_network


class Shell(Agent):
    def __init__(self, n_action, n_state, activ_con):
        super().__init__(n_action, n_state)
        self.__networks = dict()
        for agt in activ_con:
            self.__networks.update({agt: copy.copy(super().get_base_net())})

    def get_action(self, state):
        #TODO: hier weitermachen mit übergeben der states an das netz
        return 1

    def get_networks(self):
        return self.__networks


class FSC(Agent):
    def __init__(self, n_action, n_state, activ_con):
        super().__init__(n_action, n_state)
        self.__networks = dict()
        for agt in activ_con:
            self.__networks.update({agt: copy.copy(super().get_base_net())})

    def get_action(self, state):
        return 0

    def get_networks(self):
        return self.__networks


class Gov(Agent):
    def __init__(self, n_action, n_state, activ_con):
        # super().__init__(n_action, n_state)
        pass

    def get_action(self):
        return None
