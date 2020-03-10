import matplotlib.pyplot as plt
import numpy as np

from environment.BankSimEnv import BankSimEnv
from Naive_MARL.NaiveA2C.ddpg_agent import Agent
from utils.plot_utils import setup_matplotlib, plot_custom_errorbar_plot
from utils.tools import MA_obs_to_bank_obs
from config import GAME_PARAMS


for shock in [0.1, 0.15, 0.2]:
    agent_dict = {}
    env = BankSimEnv(shock)
    env.reset()

    bank_names = list(env.allAgentBanks.keys())
    print(f'Game simulations starting! All {len(bank_names)} participants are: {bank_names}.')
    for idx, name in enumerate(bank_names):
        agent = Agent(state_size=6, action_size=2, random_seed=0, name=name)
        agent_dict[name] = agent

    round_to_print = 100
    average_lifespans = []
    total_equities = []
    for episode in range(GAME_PARAMS.EPISODES):
        if episode == 0 or episode % round_to_print == 0:
            print(f'=========================================Episode {episode}===============================================')
        current_obs = env.reset()
        play, max_play = 0, 5
        num_default = []
        while play < GAME_PARAMS.MAX_PLAY:
            actions = {}
            for bank_name, bank in env.allAgentBanks.items():
                if episode % round_to_print == 0:
                    print(
                        f'Round {play}. Bank {bank_name}, CB: {int(bank.BS.Asset["CB"].Quantity)}, GB: {int(bank.BS.Asset["GB"].Quantity)}',
                        f'EQUITY: {int(bank.get_equity_value())}, ASSET: {int(bank.get_asset_value())}, LIABILITY: {int(bank.get_liability_value())}, LEV: {int(bank.get_leverage_ratio() * 10000)} bps')
                if bank_name in env.DefaultBanks:
                    continue
               # conversion
                my_obs = MA_obs_to_bank_obs(current_obs, bank)
                current_obs[bank_name] = my_obs
                # choose action
                actions[bank_name] = agent_dict[bank_name].act(current_obs[bank_name].astype(float), add_noise=True, eps=0.005)
                # print(episode, play, bank_name, actions[bank_name])
            # convert actions
            actions_dict = {}
            for name, action in actions.items():
                action_dict = {}
                action_dict['CB'], action_dict['GB'] = action[0], action[1]
                actions_dict[name] = action_dict
            new_obs, rewards, dones, infos = env.step(actions_dict)
            for bank_name, bank in env.allAgentBanks.items():
                if bank_name in env.DefaultBanks:
                    continue
                my_new_obs = MA_obs_to_bank_obs(new_obs, bank)
            for bank_name, bank in env.allAgentBanks.items():
                if bank_name in env.DefaultBanks:
                    continue
                agent_dict[bank_name].step(current_obs[bank_name], actions[bank_name], rewards[bank_name], my_new_obs, dones[bank_name])
            current_obs = new_obs
            num_default.append(infos['NUM_DEFAULT'])
            play += 1
            if play == max_play:
                # print(infos['AVERAGE_LIFESPAN'])
                average_lifespans.append(infos['AVERAGE_LIFESPAN'])
                total_equities.append(infos['TOTAL_EQUITY'])


    setup_matplotlib()
    av_step = 100
    x_points = int(len(average_lifespans)/av_step)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plt.sca(axs[0])
    average_lifespans = np.array(average_lifespans).reshape(x_points, av_step)
    means_avg_lifespans = np.mean(average_lifespans, axis=1)
    stds_avg_lifespans = np.std(average_lifespans, axis=1)
    plot_custom_errorbar_plot(range(x_points), means_avg_lifespans, stds_avg_lifespans)
    plt.xlabel(f'Num episode / {av_step}')
    plt.ylabel('Avg life span of all banks')
    plt.sca(axs[1])
    total_equities = np.array(total_equities).reshape(x_points, av_step)
    means_total_equities = np.mean(total_equities, axis=1)
    stds_total_equities = np.std(total_equities, axis=1)
    plot_custom_errorbar_plot(range(x_points), means_total_equities, stds_total_equities)
    plt.xlabel(f'Num episode / {av_step}')
    plt.ylabel('End of episode system total equity')
    fig.suptitle(f'Learning behavior: simulation with {len(list(agent_dict.keys()))} banks')
    plt.subplots_adjust(top=0.85)
    plt.show()



