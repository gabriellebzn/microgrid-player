# python 3
# this class combines all basic features of a generic player
import numpy as np
import pandas as pd


class Player:
    
    



	def __init__(self):
		# some player might not have parameters
		self.parameters = 0
		self.horizon = 48
        self.surface = 1000
        self.l_pv = np.zeros(48)
        self.a = np.zeros(48)
        self.C = 30000
        self.rho_c = rho_d = 0.95
        self.pmax = 10000
        self.prix = np.random.rand(48)
        self.T = 48
        self.delta_t = 0.5

    	df = pd.read_csv('pv_prod_scenarios.csv',sep=';')
        df = df[(df['region']=='grand_nord') & (df['day']=='01/01/2014')]
        prod = df[['pv_prod (W/m2)']]
        l_pv = np.array(prod)
        l_pv = (-1)*surface * l_pv
        l_pv = np.repeat(l_pv,2)

        l_bat = np.zeros(48)
        l_i = np.zeros(48)

	def set_scenario(self, scenario_data):
		self.data = scenario_data

	def set_prices(self, prices):
		self.prices = prices

	def compute_all_load(self):
		load = np.zeros(self.horizon)
		# for time in range(self.horizon):
		# 	load[time] = self.compute_load(time)
		return load

	def take_decision(self, time):
		lpv = l_pv[0:time]
		a_prov = 0
		min = 0
		min_li = 0
		for lbat in lpv:
			a_prov = a[time - 1] + (rho_c*max(lbat,0) - (1/rho_d)*min(lbat,0))*delta_t
			if (a_prov <= C) and (abs(lbat) <= pmax):
				if (lbat - l_pv[time])*prix[time] < min :
					min = (lbat - l_pv[time])*prix[time]
					min_li = lbat - l_pv[time]
		l_i[time] = min_li
		return 0

	def compute_load(self, time):
		load = self.take_decision(time)
		# do stuff ?
		return load

	def reset(self):
		# reset all observed data
		pass