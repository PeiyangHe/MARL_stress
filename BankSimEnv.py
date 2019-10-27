from ray.rllib.env import MultiAgentEnv
from AgentBank import Asset, Liability, BalanceSheet, AgentBank
from AssetMarket import AssetMarket
from ImpactFunctions import CifuentesImpact


def load_bs():
    with open('EBA_2018.csv', 'r') as data:
        bs_from_csv = data.read().strip().split('\n')[1:]

    BalanceSheets = {}

    for bs in bs_from_csv:
        assets, liabilities = {}, {}

        # extract different asset/liability types from the doc
        row = bs.split(' ')
        bank_name, equity, leverage, debt_sec, gov_bonds = row
        equity = float(equity)

        debt_sec = float(debt_sec)
        gov_bonds = eval(gov_bonds)
        corp_bonds = debt_sec - gov_bonds

        asset = equity / (float(leverage) / 100)
        cash = 0.05 * asset
        other_asset = asset - debt_sec - cash

        liability = asset - equity
        loan = other_liability = liability / 2

        assets['CASH'], assets['CB'], assets['GB'], assets['OTHER'] = \
            Asset('CASH', cash, CifuentesImpact), Asset('CB', corp_bonds, CifuentesImpact), \
            Asset('GB', gov_bonds, CifuentesImpact), Asset('OTHER', other_asset, CifuentesImpact)

        liabilities['LOAN'], liabilities['OTHER'] = Liability('LOAN', loan), Liability('OTHER', other_liability)

        BS = BalanceSheet(assets, liabilities)
        BalanceSheets[bank_name] = BS
    return BalanceSheets


def initialize_asset_market():
    assets = {}
    assets['CASH'], assets['CB'], assets['GB'], assets['OTHER'] = \
        Asset('CASH', 1e7, CifuentesImpact), Asset('CB', 1e7, CifuentesImpact), \
        Asset('GB', 1e7, CifuentesImpact), Asset('OTHER', 1e7, CifuentesImpact)
    return AssetMarket(assets)


class BankSimEnv(MultiAgentEnv):
    def __init__(self):
        self.allAgentBanks = {}
        self.AssetMarket = initialize_asset_market()

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        balance_sheets = load_bs()
        for bank_name, BS in balance_sheets.items():
            self.allAgentBanks[bank_name] = AgentBank(bank_name, self.AssetMarket, BS)
        self.AssetMarket.apply_initial_shock('GB', 0.2)
        obs = {}
        price_dict = self.AssetMarket.query_price()
        for bank_name, bank in self.allAgentBanks.items():
            obs[bank.BankName] = (price_dict, bank.BS.Asset, bank.BS.Liability, bank.get_leverage_ratio())
        return obs

    def step(self, action_dict):
        # action_dict: {AgentName: {TYPE: QTY}}
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        obs, rewards, dones, infos = {}, {}, {}, {}
        # get new prices that reflect the market impact of all orders
        new_prices = self.AssetMarket.process_orders(action_dict)
        for bank_name, bank in self.allAgentBanks.items():
            # reflect the asset sale on the bank' BS
            action = action_dict[bank_name]
            bank.BS.sell_action(action)
            bank.BS.Asset['CASH'].Quantity += self.AssetMarket.convert_to_cash(action)
            # force banks to pay back loans to keep leverage ratio above minimal
            minlev = bank.LeverageMin
            if bank.get_leverage_ratio() > minlev:
                continue
            else:
                asset_value = bank.get_asset_value()
                liability_value = bank.get_liability_value()
                equity_value = asset_value - liability_value
                cash_to_pay = (equity_value - (1+minlev) * liability_value) / (1 - minlev)
                if cash_to_pay > bank.BS.Asset['CASH']:
                    bank.DaysInsolvent += 1
                else:
                    bank.BS.Asset['CASH'].Quantity -= cash_to_pay
                    bank.BS.Liability['LOAN'].Quantity -= cash_to_pay
            # return obs
            obs[bank.BankName] = (new_prices, bank.BS.Asset, bank.BS.Liability, bank.get_leverage_ratio())
            # return reward
            if bank.DaysInsolvent >= 1:
                rewards[bank_name] = -1e3
            else:
                rewards[bank_name] = 1.
            # return dones
            if bank.DaysInsolvent >= 2:
                dones[bank_name] = True
            else:
                dones[bank_name] = False
            # return infos
            infos[bank_name] = None

            return obs, rewards, dones, infos



if __name__ == '__main__':
    env = BankSimEnv()
    init_obs = env.reset()
    for _, v in init_obs.items():
        print(v[3])






