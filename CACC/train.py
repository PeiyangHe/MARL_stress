import numpy as np
import matplotlib.pyplot as plt
from environment.BankSimEnv import BankSimEnv
from CACC.new import CACC_Agent
from utils.plot_utils import setup_matplotlib, plot_custom_errorbar_plot
from config import GAME_PARAMS
from utils.tools import MA_obs_to_bank_obs



for shock in [0.1, 0.15, 0.2]:
    agent_dict = {}
    env = BankSimEnv(shock)
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
    for episode in range(GAME_PARAMS.EPISODES):

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
                    print(f'Round:{play}, Bank:{bank_name}')
                else:
                    new_obs_dict[bank_name] = MA_obs_to_bank_obs(new_obs, bank)
            cacc_agent.step(current_obs, actions, rewards, new_obs_dict, dones)
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
