import matplotlib.pyplot as plt
import numpy as np

from environment.BankSimEnv import BankSimEnv
from CA2C.ca2c_agent import CA2C_Agent, Centralized_Critic
from utils.tools import MA_obs_to_bank_obs
from config import GAME_PARAMS


def ca2c_sensitivity_impact(impact_ls = (0.001 * x for x in range(0, 200, 5))):
    # loop over shocks
    eoe_equities = []
    for l in impact_ls:
        agent_dict = {}
        env = BankSimEnv(shock=0.05, Cifuentes_Impact_lambda=l)
        env.reset()
        local_critic, update_critic = Centralized_Critic(6, 2, seed=0, num_agents=5), Centralized_Critic(6, 2, seed=0, num_agents=5)

        bank_names = list(env.allAgentBanks.keys())
        print(f'Cifuentes_Impact_lambda = {l}! All {len(bank_names)} participants are: {bank_names}.')
        for idx, name in enumerate(bank_names):
            agent = CA2C_Agent(6, 2, critic_local=local_critic, critic_target=update_critic, num_agents=5, random_seed=0, name=name)
            agent_dict[name] = agent


        round_to_print = 50
        average_lifespans = []
        total_equities = []
        for episode in range(GAME_PARAMS.EPISODES):
            total = np.sum([bank.get_equity_value() for bank in env.allAgentBanks.values()])
            if episode == 0 or episode % round_to_print == 0:
                #print(f'=========================================Episode {episode}===============================================')
                a=1
            current_obs = env.reset()
            play, max_play = 0, 5
            num_default = []
            while play < GAME_PARAMS.MAX_PLAY:
                actions = {}
                for bank_name, bank in env.allAgentBanks.items():
                    if bank_name in env.DefaultBanks:
                        current_obs[bank_name]=np.asarray([0, 0, 0, 0, 0, 0])
                        continue
                    # conversion

                    my_obs = MA_obs_to_bank_obs(current_obs, bank)
                    current_obs[bank_name] = my_obs
                # choose action
                for bank_name, bank in env.allAgentBanks.items():
                    actions[bank_name] =agent_dict[bank_name].act(current_obs[bank_name])
                #                print(episode, play, bank_name, actions[bank_name])
                # convert actions
                actions_dict = {}
                for name, action in actions.items():
                    action_dict = {}
                    action_dict['CB'], action_dict['GB'] = action[0], action[1]
                    actions_dict[name] = action_dict
                new_obs, rewards, dones, infos = env.step(actions_dict)
                new_obs_dict = {}

                for bank_name, bank in env.allAgentBanks.items():
                    if bank_name in env.DefaultBanks:
                        new_obs_dict[bank_name] = np.asarray([0, 0, 0, 0, 0, 0])
                    else:
                        new_obs_dict[bank_name] = MA_obs_to_bank_obs(new_obs, bank)
                    current_obs[bank_name] = new_obs_dict[bank_name]
                for bank_name, bank in env.allAgentBanks.items():
                    if bank_name in env.DefaultBanks:
                        continue
                    agent_dict[bank_name].step(current_obs, actions, rewards, new_obs_dict, dones)
                current_obs = new_obs
                num_default.append(infos['NUM_DEFAULT'])
                play += 1
                if play == max_play:
                    # print(infos['AVERAGE_LIFESPAN'])
                    average_lifespans.append(infos['AVERAGE_LIFESPAN'])
                    total_equities.append(infos['TOTAL_EQUITY']/total)
        eoe_equity = np.asarray(total_equities).max()
        eoe_equities.append(eoe_equity)
    return eoe_equities


#fig, ax = plt.subplots()
#ax.plot(impact_ls, eoe_equities)
#ax.set(xlabel='initial asset shock', ylabel='end of episode equity',
#       title='Relation of shock and equity for Naive A2C, five agents')
# ax.set(xlabel='initial asset shock', ylabel='end of episode equity',
#        title=f'Heuristic Action buffer = {0.045}, five agents')
# ax.grid()