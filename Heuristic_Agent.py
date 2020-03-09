import numpy as np

class Heuristic_Agent:
    def __init__(self, bar=0.035, target=0.045, percent=1, name=None):
        self.bar = bar
        self.target = target
        self.percent = percent
        self.name = name

    def act(self, state, env=None):
        agent_bank = env.allAgentBanks[self.name]
        lev = agent_bank.get_leverage_ratio()
        if lev < self.bar + (self.target - self.bar) * self.percent:
            asset = agent_bank.get_asset_value()
            portion_to_sell = 1 - (agent_bank.get_equity_value()/(asset*self.target))
        else:
            portion_to_sell = 0
        return np.asarray([portion_to_sell, portion_to_sell])

    def step(self, a,b,c,d,e):
        pass