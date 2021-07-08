from pyomo.opt import SolverFactory
import pyomo.environ as pyo
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
import json

def Operation(hour,path_test,CAP_CHP_elect,CAP_boiler,CAP_grid,electricity_demand,heating_demand,electricity_prices,electricity_EF,use_CHP,use_boilers,use_grid):
    path_test = path_test
    components_path = os.path.join(path_test,'Energy Components')
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    use_solar_PV = editable_data['Solar_PV']
    use_wind_turbine = editable_data['Wind_turbines']
    use_battery = editable_data['Battery']
    use_grid = editable_data['Grid']
    use_CHP = editable_data['CHP']
    use_boilers = editable_data['Boiler']
    hour= hour
    CAP_CHP_elect = CAP_CHP_elect
    CAP_boiler = CAP_boiler
    CAP_grid = CAP_grid
    electricity_demand = electricity_demand
    heating_demand = heating_demand
    electricity_prices = electricity_prices
    electricity_EF = electricity_EF
    energy_component_number = {}
    energy_component_type = 0
    model = pyo.ConcreteModel()
    k=0
    j=0
    i=0
    if  use_CHP=='yes' and CAP_CHP_elect!=0:
        energy_component_number['CHP']=energy_component_type
        model.F_CHP = pyo.Var(bounds=(0,CHP_system.CHP(CAP_CHP_elect,0,path_test)[5])) #Decision space for CHP fuel rate
        energy_component_type +=1
        F_F_CHP = 1
        CHP_model = model.F_CHP
        k=1
    else:
        F_F_CHP = 0
        CHP_model = 0
        k=0
    if  use_boilers=='yes' and CAP_boiler!=0 and heating_demand>0:
        energy_component_number['boilers']=energy_component_type
        energy_component_type +=1
        F_F_boiler = 1
        Boiler_model = heating_demand - CHP_system.CHP(CAP_CHP_elect,CHP_model,path_test)[1]
        j=1
    else:
        F_F_boiler =  0
        Boiler_model = 0
        CHP_model = heating_demand/CHP_system.CHP(CAP_CHP_elect,0,path_test)[7]
        j=0
    if  use_grid=='yes' and electricity_demand>0:
        energy_component_number['grid']=energy_component_type
        energy_component_type +=1
        F_E_grid = 1
        grid_model = electricity_demand - CHP_system.CHP(CAP_CHP_elect,CHP_model,path_test)[0]
        i=1
    else:
        F_E_grid = 0
        grid_model = 0
        i=0
    if (i*j*k==0):
        if k==0:
            if heating_demand>0:
                Boiler_model = heating_demand
            else:
                Boiler_model = 0
            if electricity_demand>0:
                grid_model = electricity_demand
            else:
                grid_model = 0
        if i==0:
            CHP_model = 0
            Boiler_model = heating_demand
        cost_objective = CHP_system.CHP(CAP_CHP_elect,CHP_model,path_test)[3]*F_F_CHP +boilers.NG_boiler(Boiler_model,CAP_boiler,path_test)[2]*F_F_CHP + grid_model*electricity_prices*F_E_grid
        emissions_objective = CHP_system.CHP(CAP_CHP_elect,CHP_model*F_F_CHP,path_test)[4] + boilers.NG_boiler(Boiler_model*F_F_boiler,CAP_boiler,path_test)[3] +grid_model*electricity_EF*F_E_grid
        population_size = int(editable_data['population_size'])
        #print('CHP model',i,j,k,CHP_model,cost_objective)
        return 'cost',[cost_objective]*population_size,'emisisons',[emissions_objective]*population_size,'CHP',[CHP_model]*population_size,'Boilers',[Boiler_model]*population_size,'Grid',[grid_model]*population_size
    model.Constraint_elect = pyo.Constraint(expr = grid_model>=0) # Electricity balance of demand and supply sides
    model.Constraint_heat = pyo.Constraint(expr = Boiler_model>=0) # Heating balance of demand and supply sides
    model.f1_cost = pyo.Var()
    model.f2_emissions = pyo.Var()
    model.C_f1_cost = pyo.Constraint(expr= model.f1_cost == CHP_system.CHP(CAP_CHP_elect,CHP_model,path_test)[3]*F_F_CHP +boilers.NG_boiler(Boiler_model,CAP_boiler,path_test)[2]*F_F_CHP + grid_model*electricity_prices*F_E_grid)
    model.C_f2_emissions = pyo.Constraint(expr= model.f2_emissions == CHP_system.CHP(CAP_CHP_elect,CHP_model*F_F_CHP,path_test)[4] + boilers.NG_boiler(Boiler_model*F_F_boiler,CAP_boiler,path_test)[3] +grid_model*electricity_EF*F_E_grid)
    model.O_f1_cost = pyo.Objective(expr= model.f1_cost)
    model.O_f2_emissions = pyo.Objective(expr= model.f2_emissions)
    model.O_f2_emissions.deactivate()

    opt = SolverFactory('glpk')
    results =opt.solve(model,load_solutions=False)
    model.solutions.load_from(results)
    if use_CHP=='yes':
        value_CHP_model=model.F_CHP.value
        if CAP_CHP_elect==0:
            value_CHP_model=0
    else:
        value_CHP_model = 0
    if use_boilers == 'yes':
        value_Boiler_model = heating_demand - CHP_system.CHP(CAP_CHP_elect,value_CHP_model,path_test)[1]
        if CAP_boiler==0:
            value_Boiler_model=0
    else:
        value_Boiler_model = 0
    if use_grid =='yes':
        value_grid_model = electricity_demand - CHP_system.CHP(CAP_CHP_elect,value_CHP_model,path_test)[0]
    else:
        value_grid_model = 0
    #print(hour,value_CHP_model, CAP_CHP_elect,CAP_boiler,boilers.NG_boiler(CAP_boiler/boilers.NG_boiler(0,CAP_boiler,path_test)[4],CAP_boiler,path_test)[0]*F_F_boiler + CHP_system.CHP(CAP_CHP_elect,CHP_system.CHP(CAP_CHP_elect,0,path_test)[5],path_test)[1]*F_F_CHP, heating_demand,electricity_demand, len(results))
    f2_max = pyo.value(model.f2_emissions)
    f1_min = pyo.value(model.f1_cost)
    ### min f2
    model.O_f2_emissions.activate()
    model.O_f1_cost.deactivate()
    solver = SolverFactory('glpk')
    results =opt.solve(model,load_solutions=False)
    model.solutions.load_from(results)
    if use_CHP=='yes':
        value_CHP_model=model.F_CHP.value
        if CAP_CHP_elect==0:
            value_CHP_model=0
    else:
        value_CHP_model = 0
    if use_boilers == 'yes':
        value_Boiler_model = heating_demand - CHP_system.CHP(CAP_CHP_elect,value_CHP_model,path_test)[1]
        if CAP_boiler==0:
            value_Boiler_model=0
    else:
        value_Boiler_model = 0
    if use_grid =='yes':
        value_grid_model = electricity_demand - CHP_system.CHP(CAP_CHP_elect,value_CHP_model,path_test)[0]
    else:
        value_grid_model = 0
    #print( '( CHP , Boiler, Grid ) = ( ' + str(value_CHP_model) + ' , ' + str(value_Boiler_model) +  ' , ' + str(value_grid_model) +' )')
    #print( 'f1_cost = ' + str(pyo.value(model.f1_cost)) )
    #print( 'f2_emissions = ' + str(pyo.value(model.f2_emissions)) )
    f2_min = pyo.value(model.f2_emissions)
    f1_max = pyo.value(model.f1_cost)

    #print('f2_min',f2_min)

    ### apply normal $\epsilon$-Constraint
    model.O_f1_cost.activate()
    model.O_f2_emissions.deactivate()
    model.e = pyo.Param(initialize=0, mutable=True)
    model.C_epsilon = pyo.Constraint(expr = model.f2_emissions == model.e)
    solver = SolverFactory('glpk')
    results =opt.solve(model,load_solutions=False)
    model.solutions.load_from(results)
    if use_CHP=='yes':
        value_CHP_model=model.F_CHP.value
        if CAP_CHP_elect==0:
            value_CHP_model=0
    else:
        value_CHP_model = 0
    if use_boilers == 'yes':
        value_Boiler_model = heating_demand - CHP_system.CHP(CAP_CHP_elect,value_CHP_model,path_test)[1]
        if CAP_boiler==0:
            value_Boiler_model=0
    else:
        value_Boiler_model = 0
    if use_grid =='yes':
        value_grid_model = electricity_demand - CHP_system.CHP(CAP_CHP_elect,value_CHP_model,path_test)[0]
    else:
        value_grid_model = 0

    #print('emissions range',str(f2_min)+', ' + str(f2_max))
    #print('cost range',str(f1_min)+', ' + str(f1_max))
    n = int(editable_data['population_size'])-2
    step = int((f2_max - f2_min) / n)
    if step==0:
        CHP_EC = [value_CHP_model]*(n+2)
        Boiler_EC = [value_Boiler_model]*(n+2)
        grid_EC = [value_grid_model]*(n+2)
        cost_objective_single = CHP_system.CHP(CAP_CHP_elect,value_CHP_model,path_test)[3]*F_F_CHP +boilers.NG_boiler(value_Boiler_model,CAP_boiler,path_test)[2]*F_F_CHP + value_grid_model*electricity_prices*F_E_grid
        emissions_objective_single = CHP_system.CHP(CAP_CHP_elect,value_CHP_model*F_F_CHP,path_test)[4] + boilers.NG_boiler(value_Boiler_model*F_F_boiler,CAP_boiler,path_test)[3] +value_grid_model*electricity_EF*F_E_grid
        cost_objective = [cost_objective_single]*(n+2)
        emissions_objective = [emissions_objective_single]*(n+2)
    else:
        steps = list(range(int(f2_min),int(f2_max),step)) + [f2_max]
        CHP_EC = []
        Boiler_EC = []
        grid_EC = []
        cost_objective = []
        emissions_objective = []
        for i in steps:
            model.e = i
            solver = SolverFactory('glpk')
            results =opt.solve(model,load_solutions=False)
            model.solutions.load_from(results)
            if use_CHP=='yes':
                value_CHP_model=model.F_CHP.value
                if CAP_CHP_elect==0:
                    value_CHP_model=0
            else:
                value_CHP_model = 0
            if use_boilers == 'yes':
                value_Boiler_model = heating_demand - CHP_system.CHP(CAP_CHP_elect,value_CHP_model,path_test)[1]
                if CAP_boiler==0:
                    value_Boiler_model=0
            else:
                value_Boiler_model = 0
            if use_grid =='yes':
                value_grid_model = electricity_demand - CHP_system.CHP(CAP_CHP_elect,value_CHP_model,path_test)[0]
            else:
                value_grid_model = 0
            CHP_EC.append(value_CHP_model)
            Boiler_EC.append(value_Boiler_model)
            grid_EC.append(value_grid_model)
            cost_objective.append(CHP_system.CHP(CAP_CHP_elect,value_CHP_model,path_test)[3]*F_F_CHP +boilers.NG_boiler(value_Boiler_model,CAP_boiler,path_test)[2]*F_F_CHP + value_grid_model*electricity_prices*F_E_grid)
            emissions_objective.append(CHP_system.CHP(CAP_CHP_elect,value_CHP_model*F_F_CHP,path_test)[4] + boilers.NG_boiler(value_Boiler_model*F_F_boiler,CAP_boiler,path_test)[3] +value_grid_model*electricity_EF*F_E_grid)
        #print('normal $\epsilon$-Constraint')
        #print(Boiler_EC)
        #print(grid_EC)
        #print('cost',cost_objective)
        #print('emisisons',emissions_objective)
    return 'cost',cost_objective,'emisisons',emissions_objective,'CHP',CHP_EC,'Boilers',Boiler_EC,'Grid',grid_EC
def results_extraction(hour, results,path_test,solar_PV_generation,wind_turbine_generation,E_bat):
    components_path = os.path.join(path_test,'Energy Components')
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    use_solar_PV = editable_data['Solar_PV']
    use_wind_turbine = editable_data['Wind_turbines']
    use_battery = editable_data['Battery']
    use_grid = editable_data['Grid']
    use_CHP = editable_data['CHP']
    use_boilers = editable_data['Boiler']
    num_components=0
    if use_boilers=='yes':
        num_components +=1
        boiler_component = pd.read_csv(os.path.join(components_path,'boilers.csv'))
    if use_CHP=='yes':
        num_components +=1
        CHP_component = pd.read_csv(os.path.join(components_path,'CHP.csv'))
    if use_solar_PV=='yes':
        num_components +=1
        PV_module = float(editable_data['PV_module']) #area of each commercial PV moduel is 1.7 M^2
        roof_top_area = float(editable_data['roof_top_area']) #60% percentage of the rooftop area of all buildings https://www.nrel.gov/docs/fy16osti/65298.pdf
    if use_wind_turbine=='yes':
        num_components +=1
        wind_component = pd.read_csv(os.path.join(components_path,'wind_turbine.csv'))
    if use_battery=='yes':
        num_components +=1
        battery_component = pd.read_csv(os.path.join(components_path,'battery.csv'))
    city = editable_data['city']
    file_name = city+'_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_MILP_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
    results_path = os.path.join(sys.path[0], file_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    i = 0
    df_operation = {}
    df_cost ={}
    df_object = {}
    df_object_all  = pd.DataFrame(columns = ['Cost ($)','Emission (kg CO2)'])
    df_operation_all = pd.DataFrame(columns = ['CHP Operation (kWh)','Boilers Operation (kWh)','Battery Operation (kWh)','Grid Operation (kWh)','Cost ($)','Emission (kg CO2)'])
    i=0
    solar_results = {}
    wind_results = {}
    CHP_results = {}
    boiler_results = {}
    battery_results = {}
    grid_results = {}
    data_object = {'Cost ($)':results[1],
    'Emission (kg CO2)':results[3]}
    data_operation={'Solar Generation (kWh)':solar_PV_generation,
    'Wind Generation (kWh)':wind_turbine_generation,
    'CHP Operation (kWh)':results[5],
    'Boilers Operation (kWh)':results[7],
    'Battery Operation (kWh)':E_bat,
    'Grid Operation (kWh)':results[9],
    'Cost ($)':results[1],
    'Emission (kg CO2)':results[3]}
    return pd.DataFrame(data_object),pd.DataFrame(data_operation)
