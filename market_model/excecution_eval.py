import torch
import copy
import time
import numpy as np
from agents import init_agents
from fsc_network_env import FSCNetworkEnvAlternative
from functions import discount_rewards, create_dir_ex_val, create_dict
from torch.utils.tensorboard import SummaryWriter

# to run tensorboard use following cmd command:
# tensorboard --logdir=C:\Users\fredd\PycharmProjects\Masterthesis\saved_data\FOLDER

# runtime parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = False                 # True if in debug mode
save_calc = False             # True if support and resource calculations should be saved
store_graph = False          # True if computational graph of network should be saved
reward_fun = 'fsc_V2.3_eps0.15_0.17_shell_V2.2_eps5e-6'

# model parameters
AGENTS = ['FSC', 'Shell', 'Gov']
ACT_AGT = ['FSC', 'Shell']               # set the active agents
ACTION_SPACE = [0, 1, 2]                 # action definition: 0 = decrease, 1 = maintain, 2 = increase
N_STATE_SPACE = {'FSC': 4,
                 'Shell': 4,
                 'Gov': 3}

INIT_RESOURCE = {'Shell': [0.03,  0.94, 0.03],    # initial resource assignment  FSC  Shell  Gov
                 'Gov':   [0.5,   0.5]}
BASE_IMPACTS = {'Shell': [0.38, 0.11],              # impact according to action 2 and 3 on agent
                'Gov':   [0.03, 0.33]}

REQUIRED_NEURAL_NETS = {'FSC':   ['All'],
                        'Shell': ['FSC', 'Gov'],
                        'Gov':   []}

DELTA_RESOURCE = 0.03           # factor by which resource assignment is changed due to action
BETA = 0.125                    # factor influences how fast support is changed due to interaction
DELTA_RESEARCH = 0.10           # factor by which research is changed due to action of FSC

lambdas = [0.5, 1]
sub_maxes = [0.5, 1]
supports = [0.25, 0.50, 0.75]

start_time_all = time.time()
for lambda_ in lambdas:
    for kappa in sub_maxes:
        for sup in supports:
            INIT_SUPPORT = [np.inf, sup, sup]       # initial support by the agents, must be in order (FSC, Shell, Gov)
            LAMBDA = lambda_                              # share of innovation subsidy from adaption subsidy sub_lvl
            SUB_MAX = kappa                             # kappa: base 0.1

            # RL parameters
            LENGTH_EPISODE = 78                # limits are based on aggregation agg_weeks=1 -> 417, agg_weeks=4 -> 105
            NUM_EPISODES = 1010
            LEARNING_RATE = 0.001
            BATCH_SIZE = 10
            GAMMA = 0.99
            SAVE_INTERVAL = 10                        # numbers of updates until data/model is saved

            CONFIG = {'agents': AGENTS,
                      'active_agents': ACT_AGT,
                      'init_support': INIT_SUPPORT,
                      'init_resource': INIT_RESOURCE,
                      'delta_resource': DELTA_RESOURCE,
                      'delta_research': DELTA_RESEARCH,
                      'base_impacts': BASE_IMPACTS,
                      'beta': BETA,
                      'lambda': LAMBDA,
                      'max subsidy': SUB_MAX,
                      'reward_function': reward_fun,
                      'length_ep': LENGTH_EPISODE,
                      'n_ep': NUM_EPISODES,
                      'lr': LEARNING_RATE,
                      'batch_size': BATCH_SIZE,
                      'gamma': GAMMA,
                      'save_interval': SAVE_INTERVAL}


            # ######################################################################################################################
            # ######################################################################################################################
            # ######################################################################################################################

            # create saving directory and save config
            SAVE_DIR = create_dir_ex_val(DEBUG, LEARNING_RATE, LAMBDA, SUB_MAX, sup)
            writer = SummaryWriter(SAVE_DIR)
            torch.save(CONFIG, (SAVE_DIR / 'config'), _use_new_zipfile_serialization=False)
            with open((SAVE_DIR / 'config.txt'), 'w') as file:
                for key in CONFIG.keys():
                    file.write(str(key) + ': ' + str(CONFIG[key]) + '\n')

            # create environment
            env = FSCNetworkEnvAlternative(init_sup=INIT_SUPPORT, init_res=INIT_RESOURCE, ep_len=LENGTH_EPISODE, lambda_=LAMBDA,
                                           sub_max=SUB_MAX, delta_res=DELTA_RESOURCE, beta=BETA, delta_search=DELTA_RESEARCH,
                                           n_state_space=N_STATE_SPACE, base_impacts=BASE_IMPACTS, agg_weeks=4, save_calc=False)

            print('--------------------------------    ' + str(device) + '    --------------------------------')
            if device == 'cuda':
                print(torch.cuda.get_device_name(0))

            # initialize agents and network optimizers and store them in dicts
            optimizers, all_agents = init_agents(ACTION_SPACE, N_STATE_SPACE, REQUIRED_NEURAL_NETS, ACT_AGT, LEARNING_RATE, device)

            # save initial agents
            torch.save(all_agents, (SAVE_DIR / 'agents_init'), _use_new_zipfile_serialization=False)

            # initialise loop variables
            total_rewards = {'FSC': [], 'Shell': []}
            batch_returns = {'FSC': [], 'Shell': []}
            batch_actions = create_dict(REQUIRED_NEURAL_NETS, ACT_AGT)
            batch_states = {'FSC':   np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['FSC']]),
                            'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['Shell']]),
                            'Gov':   np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['Gov']])}
            batch_rewards = {'FSC': np.empty([BATCH_SIZE, LENGTH_EPISODE, 1]),
                             'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, 1])}
            support_calc = {'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1]),
                            'Gov':   np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1])}
            reward_shell_calc = np.empty([BATCH_SIZE, LENGTH_EPISODE, 4])
            batch_count = 0
            optim_count = 0
            step_count = 0
            ep = 0
            times = []
            # loop over all episodes (= rollouts)
            while ep < NUM_EPISODES:
                ep += 1
                start_time = time.time()

                state_0 = env.reset()
                states = {'FSC':   np.empty([LENGTH_EPISODE, N_STATE_SPACE['FSC']]),
                          'Shell': np.empty([LENGTH_EPISODE, N_STATE_SPACE['Shell']]),
                          'Gov':   np.empty([LENGTH_EPISODE, N_STATE_SPACE['Gov']])}
                rewards = {'FSC': np.empty([LENGTH_EPISODE, 1]), 'Shell': np.empty([LENGTH_EPISODE, 1])}
                actions = create_dict(REQUIRED_NEURAL_NETS, ACT_AGT)
                step_actions = {}
                done = False

                # loop over episode steps
                while not done:

                    # get action from each agent and store it
                    for key in ACT_AGT:
                        step_actions.update({key: all_agents[key].get_actions(state_0[key])})
                        for par_agt in actions[key].keys():
                            actions[key][par_agt].append(copy.copy(step_actions[key][par_agt]))

                    # perform action on environment
                    if save_calc:
                        state_1, r1, done, sup_calc, r_shell = env.step_calc(step_actions)
                        for key in support_calc.keys():
                            support_calc[key][batch_count][step_count] = sup_calc[key]
                        reward_shell_calc[batch_count][step_count] = r_shell
                    else:
                        state_1, r1, done, _, _ = env.step(step_actions)

                    # store rewards
                    for agt in ACT_AGT:
                        rewards[agt][step_count] = r1[agt]

                    # update old state and save current state
                    for key in AGENTS:
                        states[key][step_count] = state_0[key]
                    state_0 = state_1
                    step_count += 1

                    # if done (= new rollout complete), data is put into the batch
                    if done:
                        step_count = 0
                        # put data in batch
                        for agt in AGENTS:
                            batch_states[agt][batch_count] = states[agt]
                            if agt != 'Gov':
                                batch_rewards[agt][batch_count] = rewards[agt]
                                batch_returns[agt].extend(discount_rewards(rewards[agt], GAMMA))
                                for par_agt in actions[agt].keys():
                                    batch_actions[agt][par_agt].extend(actions[agt][par_agt])
                                total_rewards[agt].append(np.array(rewards[agt]).sum())

                            # write rewards to tensorboard
                            writer.add_scalar('reward_FSC', np.array(rewards['FSC']).sum(), ep)
                            writer.add_scalar('reward_Shell', np.array(rewards['Shell']).sum(), ep)

                        if save_calc:
                            for i in range(reward_shell_calc[batch_count].shape[1]):
                                reward_shell_calc[batch_count][:, i] = discount_rewards(reward_shell_calc[batch_count][:, i], GAMMA)

                        batch_count += 1

                        # if batch full (= enough rollouts for optimization), perform optimization on networks
                        if batch_count == BATCH_SIZE:

                            # loop over all agents and over the changeable allocation
                            for agt in ACT_AGT:
                                for par_agt in optimizers[agt].keys():

                                    # set gradients of all optimizers to zero
                                    optimizers[agt][par_agt].zero_grad()

                                    # create tensors for calculation
                                    return_tensor = torch.FloatTensor(batch_returns[agt]).to(device)

                                    # create Long Tensor to use it as indices
                                    action_tensor = torch.LongTensor(np.array(batch_actions[agt][par_agt])).to(device)

                                    # calculate the loss
                                    logprob = torch.log(
                                        all_agents[agt].predict(batch_states[agt].reshape(
                                            [BATCH_SIZE*LENGTH_EPISODE, N_STATE_SPACE[agt]]), par_agt)).to(device)
                                    # use torch.gather to get the logprob to the corresponding action
                                    selected_logprobs = return_tensor * torch.gather(
                                        logprob, 1, action_tensor.unsqueeze(1)).squeeze().to(device)
                                    loss = -selected_logprobs.mean()

                                    # Calculate gradients
                                    loss.backward()

                                    # Apply gradients
                                    optimizers[agt][par_agt].step()

                                    # write loss to tensorboard after each optimization step
                                    writer.add_scalar('mean_loss_' + agt + '_' + par_agt, loss, optim_count)

                                    # save weights and its gradients for tensorboard
                                    for name, weight in all_agents[agt].get_networks()[par_agt].named_parameters():
                                        writer.add_histogram(agt + '_' + par_agt + '_' + name, weight, optim_count)
                                        writer.add_histogram(agt + '_' + par_agt + '_' + name + '_grad', weight.grad, optim_count)

                            # save agents, all states and rewards after SAVE_INTERVAL number of optimization steps
                            if optim_count % SAVE_INTERVAL == 0 or optim_count == 0:
                                torch.save(batch_rewards, (SAVE_DIR / 'rewards_optim{}_ep{}'.format(optim_count, ep)),
                                           _use_new_zipfile_serialization=False)
                                torch.save(total_rewards, (SAVE_DIR / 'tot_r'), _use_new_zipfile_serialization=False)
                                torch.save(batch_states, (SAVE_DIR / 'batch_states_optim{}'.format(optim_count)),
                                           _use_new_zipfile_serialization=False)
                                if save_calc:
                                    torch.save(support_calc, (SAVE_DIR / 'support_calc_optim{}'.format(optim_count)),
                                               _use_new_zipfile_serialization=False)
                                    torch.save(reward_shell_calc, (SAVE_DIR / 'reward_shell_calc_optim{}'.format(optim_count)),
                                               _use_new_zipfile_serialization=False)
                            optim_count += 1
                            # save agents after according optimization count
                            if optim_count % SAVE_INTERVAL == 0 or optim_count == 1:
                                torch.save(all_agents, (SAVE_DIR / 'agents_optim{}_ep{}'.format(optim_count, ep)),
                                           _use_new_zipfile_serialization=False)
                            # empty batch
                            batch_actions = create_dict(REQUIRED_NEURAL_NETS, ACT_AGT)
                            batch_returns = {'FSC': [], 'Shell': []}
                            batch_states = {'FSC': np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['FSC']]),
                                            'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['Shell']]),
                                            'Gov': np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['Gov']])}
                            batch_rewards = {'FSC': np.empty([BATCH_SIZE, LENGTH_EPISODE, 1]),
                                             'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, 1])}
                            batch_count = 0
                            if save_calc:
                                support_calc = {'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1]),
                                                'Gov': np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1])}
                                reward_shell_calc = np.empty([BATCH_SIZE, LENGTH_EPISODE, 4])
                            print('-------------------------------     Update step completed.     -------------------------------')

                # take time and save it
                times.append(time.time() - start_time)
                torch.save(times, (SAVE_DIR / 'running_times'), _use_new_zipfile_serialization=False)

                # Print moving average
                if ep % 10 == 0:
                    print('Episode {} complete. Avg time of last 10: {:.3f} sec. Average reward of last 100:'
                          ' FSC: {:.3f}, Shell: {:.3f}'.format(ep, np.mean(times[-10:]), np.mean(total_rewards['FSC'][-100:]),
                                                               np.mean(total_rewards['Shell'][-100:])))

                # save a graph to tensorboard (just for visualisation)
                if store_graph:
                    store_graph = False
                    writer.add_graph(all_agents['FSC'].get_networks()['Shell'], torch.FloatTensor(states['FSC'][0]))

                # flush the writer, so that everything is written to tensorboard
                writer.flush()

            writer.close()
            print('Total time: {:.3f}'.format(np.sum(times)))
print('Time of complete loop: {:.3f}'.format(time.time() - start_time_all))
