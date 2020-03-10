import numpy as np
import matplotlib.pyplot as plt
from environment.BankSimEnv import BankSimEnv
from CACC.cacc_agent import CACC_Agent
from utils.plot_utils import setup_matplotlib, plot_custom_errorbar_plot
from config import GAME_PARAMS
from utils.tools import MA_obs_to_bank_obs


print('cacc_impact')
# loop over shocks
eoe_equities = []
impact_ls=[0.001 * x for x in range(0, 200, 5)]
for l in impact_ls:
    agent_dict = {}
    env = BankSimEnv(shock=0.05, Cifuentes_Impact_lambda=l)
    env.reset()
    num_agents=5

    cacc_agent=CACC_Agent(6,2,num_agents)

    bank_names = list(env.allAgentBanks.keys())
    print(f'Game simulations starting! All {len(bank_names)} participants are: {bank_names}.')
    for idx, name in enumerate(bank_names):
        agent_dict[name] = cacc_agent.agents[name]


    round_to_print = 50
    average_lifespans = []
    total_equities = []
    for episode in range(1000):
        total = np.sum([bank.get_equity_value() for bank in env.allAgentBanks.values()])
        if episode == 0 or episode % round_to_print == 0:
            print(f'=========================================Episode {episode}===============================================')
        current_obs = env.reset()
        play, max_play = 0, 5
        num_default = []
        while play < GAME_PARAMS.MAX_PLAY:
            for bank_name, bank in env.allAgentBanks.items():
                if bank_name in env.DefaultBanks:
                    my_obs = np.asarray([0, 0, 0, 0, 0, 0])
                else:
                    my_obs = MA_obs_to_bank_obs(current_obs, bank)
                    if episode % round_to_print == 0:
                        print(f'Round {play}. Bank {bank_name}, CB: {int(bank.BS.Asset["CB"].Quantity)}, GB: {int(bank.BS.Asset["GB"].Quantity)}',
                            f'EQUITY: {int(bank.get_equity_value())}, ASSET: {int(bank.get_asset_value())}, LIABILITY: {int(bank.get_liability_value())}, LEV: {int(bank.get_leverage_ratio() * 10000)} bps')
                current_obs[bank_name] = my_obs
            obs = np.stack([current_obs[name] for name in range(num_agents)])
            actions=cacc_agent.act(obs)
            # choose action
            actions_dict={}
            for name, action in actions.items():
                actions_dict[name] = {'CB':action[0], 'GB':action[1]}
            new_obs, rewards, dones, infos = env.step(actions_dict)

            new_obs_dict = {}
            for bank_name, bank in env.allAgentBanks.items():
                if bank_name in env.DefaultBanks:
                    new_obs_dict[bank_name] = np.asarray([0, 0, 0, 0, 0, 0])
                    #print(f'Round:{play}, Bank:{bank_name}')
                else:
                    new_obs_dict[bank_name] = MA_obs_to_bank_obs(new_obs, bank)
            cacc_agent.step(current_obs, actions, rewards, new_obs_dict, dones)
            current_obs = new_obs
            num_default.append(infos['NUM_DEFAULT'])
            play += 1
            if play == max_play:
                # print(infos['AVERAGE_LIFESPAN'])
                total_equities.append(infos['TOTAL_EQUITY'] / total)
           
    eoe_equity = np.asarray(total_equities)[-1]
    eoe_equities.append(eoe_equity)
path = '/home/tonyairhe_gmail_com/cacc_sensitivity_impact.txt'
np.savetxt(path, eoe_equities)
