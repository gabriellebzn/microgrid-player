# python 3
# this class combines all basic features of a generic player
import numpy as np
import pandas as pd

# python 3
# this class combines all basic features of a generic player

import pulp
import numpy as np
import pandas as pd


# import sys
# import os
# from linear_prog.pulp_utils import *

class Player:

    def __init__(self):
        # some player might not have parameters
        self.parameters = 15
        self.horizon = 48
        self.surface = 100
        self.l_pv = {}
        self.l_batp = {}
        self.l_batm = {}
        self.l_i = {}
        self.a = {}

        self.C = 30000
        self.rho_c = 0.95
        self.rho_d = 0.95
        self.pmax = 10000
        self.prix = 100 * np.random.rand(48)
        self.T = 48
        self.delta_t = 0.5

        df = pd.read_csv('pv_prod_scenarios.csv', sep=';')
        df = df[(df['region'] == 'grand_nord') & (df['day'] == '01/01/2014')]
        self.prod = df[['pv_prod (W/m2)']]
        self.l_pv = np.array(self.prod)
        self.l_pv = (-1) * self.surface * self.l_pv / 1000
        self.l_pv = np.repeat(self.l_pv, 2)

        for t in range(48):
            self.l_batp[t] = pulp.LpVariable('lbatp_' + str(t), 0, self.pmax)
            self.l_batm[t] = pulp.LpVariable('lbatm_' + str(t), -self.pmax,0)
            # self.l_bat = pulp.LpVariable.dicts('l_bat', indexs=np.arange(48), lowBound=-1*self.pmax, upBound=self.pmax)
            self.l_i[t] = pulp.LpVariable('li_' + str(t), None, None)
            # self.l_i = pulp.LpVariable.dicts('l_i', indexs=np.arange(48))
            if t==0:
                self.a[0]=0
            self.a[t] = pulp.LpVariable('a_' + str(t), lowBound=0.0, upBound=self.C)


    # self.a = pulp.LpVariable.dicts('a', indexs=np.arange(48), lowBound=0.0, upBound=self.C)

    def set_scenario(self, scenario_data):
        self.data = scenario_data

    def set_prices(self, prices):
        self.prices = prices

    def compute_all_load(self):
        load = np.zeros(self.horizon)

        load = self.compute_load(0)
        return load

    def take_decision(self, time):
        lp = pulp.LpProblem("takedecision.lp", pulp.LpMinimize)
        lp.setSolver()
        lpv = self.l_pv

        for t in range(time, 48):
            # if t == 48:
            #     lp += self.a[t] == 0, "cnt_batt_" + str(t)

                # OB ajout d'un affichage -> bogue est ici... votre dictionnaire
                # ne contient pas la clé 0 quand t= 1
                # print("t=", t, "et dict. = ",  self.a)
            lp += self.a[t] == self.a[t - 1] + (self.rho_c * self.l_batp[t] - (1 / self.rho_d) * self.l_batm[t]) * self.delta_t, "cnt_batt_" + str(t)
            lp += self.l_i[t] == self.l_batp[t]+self.l_batm[t] - lpv[t], "cnt_li_" + str(t)

        lp.setObjective(pulp.lpSum([self.l_i[t] * self.prix[t] for t in range(time, 48)]))

        status = lp.solve()  # Solver
        print(pulp.LpStatus[status])
        # model=Model(lp,self.l_i)
        for t in range(48):
            print(self.l_i[t].varValue)

        # return model
        return [self.l_i[t].varValue for t in range(time, 48)]

    def compute_load(self, time):
        load = self.take_decision(time)
        # do stuff ?
        return load

    def reset(self):
        # reset all observed data
        pass


if __name__ == "__main__":
    myplayer = Player()
    myload = myplayer.compute_load(1)

# class Player:
#
#     def __init__(self):
#         # some player might not have parameters
#         self.parameters = 15
#         self.horizon = 48
#         self.surface = 100
#         self.l_pv = np.zeros(48)
#         self.a = np.zeros(48)
#         self.C = 30000
#         self.rho_c = 0.95
#         self.rho_d = 0.95
#         self.pmax = 10000
#         self.prix = 100*np.random.rand(48)
#         self.T = 48
#         self.delta_t = 0.5
#
#         df = pd.read_csv('pv_prod_scenarios.csv',sep=';')
#         df = df[(df['region']=='grand_nord') & (df['day']=='01/01/2014')]
#         self.prod = df[['pv_prod (W/m2)']]
#         self.l_pv = np.array(self.prod)
#         self.l_pv = (-1)*self.surface * self.l_pv /1000
#         self.l_pv = np.repeat(self.l_pv,2)
#
#         self.l_bat = np.zeros(48)
#         self.l_i = np.zeros(48)
#
#     def set_scenario(self, scenario_data):
#         self.data = scenario_data
#
#     def set_prices(self, prices):
#         self.prices = prices
#
#     def compute_all_load(self):
#         load = np.zeros(self.horizon)
#         for time in range(self.horizon):
#             load[time] = self.compute_load(time)
#         return load
#
#     def take_decision(self, time):
#         lpv = self.l_pv
#         for t in range(time, 48):
#             a_prov = float(0)
#             mini = 0
#             min_li = 0
#             for i in range(len(lpv)):
#                 if t >= 1 and t<len(lpv)-1:
#                     a_prov = self.a[t - 1] + (self.rho_c*np.maximum(lpv[i],0) - (1/self.rho_d)*np.minimum(lpv[i],0))*self.delta_t
#                 else :
#                     a_prov = 0
#                 if (a_prov <= self.C) and (abs(lpv[i]) <= self.pmax):
#                     if (lpv[i] - self.l_pv[t])*self.prix[t] < mini :
#                         mini = (lpv[i] - self.l_pv[t])*self.prix[t]
#                         min_li = lpv[i] - self.l_pv[t]
#             self.l_i[t] = min_li
#         #print(self.l_i)
#         return 0
#
#     def compute_load(self, time):
#         load = self.take_decision(time)
#         # do stuff ?
#         return load
#
#     def reset(self):
#         # reset all observed data
#         pass
#
# if __name__ == "__main__":
#     myplayer = Player()
#     myload = myplayer.compute_load(1)
