from impact_sensitivity_tests.ha_sensitivity import ha_sensitivity_impact
from impact_sensitivity_tests.sp_sensitivity import sp_sensitivity_impact
from impact_sensitivity_tests.na2c_sensitivity import na2c_sensitivity_impact
from impact_sensitivity_tests.ca2c_sensitivity import ca2c_sensitivity_impact
from shock_sensitivity_tests.sp_sensitivity import sp_sensitivity_shock
from shock_sensitivity_tests.ha_sensitivity import ha_sensitivity_shock
from shock_sensitivity_tests.na2c_sensitivity import na2c_sensitivity_shock
from shock_sensitivity_tests.ca2c_sensitivity import ca2c_sensitivity_shock
import matplotlib.pyplot as plt


impact_ls = [0.001 * x for x in range(0, 200, 5)]
shocks = [0.001 * x for x in range(0, 200, 5)]
ha_sensitivity_impact=ha_sensitivity_impact(impact_ls)
sp_sensitivity_impact=sp_sensitivity_impact(impact_ls)
na2c_sensitivity_impact=na2c_sensitivity_impact(impact_ls)
ca2c_sensitivity_impact=ca2c_sensitivity_impact(impact_ls)
ha_sensitivity_shock=ha_sensitivity_shock(shocks)
sp_sensitivity_shock=sp_sensitivity_shock(shocks)
na2c_sensitivity_shock=na2c_sensitivity_shock(shocks)
ca2c_sensitivity_shock=ca2c_sensitivity_shock(shocks)
fig,(ax1,ax2)=plt.subplots(1, 2, sharey=True)
ax1.grid()
ax1.set(xlabel='Shock to initial assets (%)', ylabel='Remaining equity in the financial system (%)')
ax1.plot(shocks, ha_sensitivity_shock, shocks, sp_sensitivity_shock, shocks, na2c_sensitivity_shock, shocks, ca2c_sensitivity_shock)
ax1.legend(('heuristics', 'social planner', 'na2c', 'ca2c'))
ax2.set(xlabel='Fall in price (%)')
ax2.plot(impact_ls, ha_sensitivity_impact, impact_ls, sp_sensitivity_impact,impact_ls, na2c_sensitivity_impact, impact_ls, ca2c_sensitivity_impact)
ax2.legend(('heuristics', 'social planner', 'na2c', 'ca2c'))
ax2.grid()




plt.show()