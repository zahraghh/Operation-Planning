import numpy as np
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.util.misc import stack
from pymoo.model.problem import FunctionalProblem
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import datetime as dt
from collections import defaultdict
import os
import sys
from pathlib import Path
import battery
import boilers
import solar_PV
import CHP_system
import wind_turbine


def perform_EA(hour,path_test,CAP_CHP_elect,CAP_boiler,CAP_grid,electricity_demand,heating_demand,electricity_prices,electricity_EF,use_CHP,use_boilers,use_grid):
    class Operation_EA(Problem):
        def __init__(self,hour,path_test,CAP_CHP_elect,CAP_boiler,CAP_grid,electricity_demand,heating_demand,electricity_prices,electricity_EF,use_CHP,use_boilers,use_grid):
            self.path_test = path_test
            self.hour= hour
            self.CAP_CHP_elect = CAP_CHP_elect
            self.CAP_boiler = CAP_boiler
            self.CAP_grid = CAP_grid
            self.electricity_demand = electricity_demand
            self.heating_demand = heating_demand
            self.electricity_prices = electricity_prices
            self.electricity_EF = electricity_EF
            self.energy_component_number = {}
            energy_component_type = 0
            super().__init__(n_var=3,
                             n_obj=2,
                             n_constr=2,
                             xl=np.array([0,0,0]),
                             xu=np.array([CHP_system.CHP(self.CAP_CHP_elect,0,self.path_test)[5],self.CAP_boiler/boilers.NG_boiler(0,self.CAP_boiler,self.path_test)[4],self.electricity_demand+1]),
                             elementwise_evaluation=True)


            if  use_CHP=='yes':
                self.energy_component_number['CHP']=energy_component_type
                energy_component_type +=1
            if  use_boilers=='yes':
                self.energy_component_number['boilers']=energy_component_type
                energy_component_type +=1
            if  (use_grid=='yes' and self.electricity_demand!=0):
                self.energy_component_number['grid']=energy_component_type
                energy_component_type +=1
        def evaluate(self, x, out, *args, **kwargs):
            F_CHP = x[0]
            F_boilers = x[1]
            P_grid = x[2]
            if self.CAP_CHP_elect ==0:
                F_F_CHP = 0
            else:
                F_F_CHP = 1
            if self.CAP_boiler==0:
                F_F_boiler = 0
            else:
                F_F_boiler = 1
            if (self.electricity_demand or self.CAP_grid) ==0:
                F_E_grid = 0
            else:
                F_E_grid = 1

            f1 = CHP_system.CHP(self.CAP_CHP_elect,F_CHP,self.path_test)[3]*F_F_CHP +boilers.NG_boiler(F_boilers,self.CAP_boiler,self.path_test)[2]*F_F_CHP + P_grid*self.electricity_prices*F_E_grid
            f2 = CHP_system.CHP(self.CAP_CHP_elect,F_CHP*F_F_CHP,self.path_test)[4] + boilers.NG_boiler(F_boilers,self.CAP_boiler,self.path_test)[3]*F_F_boiler +P_grid*self.electricity_EF*F_E_grid #kg CO2
            g1 = (boilers.NG_boiler(F_boilers,self.CAP_boiler,self.path_test)[0]*F_F_boiler + CHP_system.CHP(self.CAP_CHP_elect,F_CHP,self.path_test)[1]*F_F_CHP -self.heating_demand)**2
            g2 = (CHP_system.CHP(self.CAP_CHP_elect,F_CHP,self.path_test)[0]*F_F_CHP + P_grid*F_E_grid - self.electricity_demand)**2

            out["F"] =[f1, f2]
            out["G"] = [g1, g2]

    problem= Operation_EA(hour,path_test,CAP_CHP_elect,CAP_boiler,CAP_grid,electricity_demand,heating_demand,electricity_prices,electricity_EF,use_CHP,use_boilers,use_grid)
    algorithm = NSGA2(pop_size=10,sampling=get_sampling("real_random"),eliminate_duplicates=True)
    res = minimize(problem,algorithm)
    print(res)
