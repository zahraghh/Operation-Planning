import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv
import numpy as np
import plotly.express as px
import os
import sys
import json
import pandas as pd
editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
path_test =  os.path.join(sys.path[0])
df_object = {}
df_operation = {}
num_scenarios = int(editable_data['num_scenarios'])
num_clusters = int(editable_data['Cluster numbers'])+2
num_scenarios = 500
city =str(editable_data['city'])
def generate_UA_csv(num_cluster):
    for represent in range(num_cluster):
        df_object_all  = pd.DataFrame(columns = ['Cost ($)','Emission (kg CO2)'])
        df_operation_all = pd.DataFrame(columns = ['CHP Operation (kWh)','Boilers Operation (kWh)','Battery Operation (kWh)','Grid Operation (kWh)','Cost ($)','Emission (kg CO2)'])
        for day in range(num_scenarios):
            df_object = {}
            df_operation = {}
            file_name = city+'_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_EA_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
            results_path = os.path.join(sys.path[0], file_name)
            df_object[day]=pd.read_csv(os.path.join(results_path , str(represent)+'_'+str(day)+'_represent_objectives.csv'))
            df_operation[day]=pd.read_csv(os.path.join(results_path, str(represent)+'_'+str(day)+'_represent_sizing_all.csv'))
            df_object_all=df_object_all.append(df_object[day])
            df_operation_all=df_operation_all.append(df_operation[day])
            plt.scatter(df_object[day]['Cost ($)'],df_object[day]['Emission (kg CO2)'])
        df_object_all.to_csv(os.path.join(results_path,'UA_'+ str(represent)+'_represent_objectives_all.csv'), index=False)
        df_operation_all.to_csv(os.path.join(results_path,'UA_'+str(represent)+'_represent_operation_all.csv'), index=False)
generate_UA_csv(num_clusters)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.edgecolor'] = 'black'
cmap = plt.cm.RdYlGn
plt.rcParams["figure.figsize"] = (8,6)
city =str(editable_data['city'])
file_name = city+'_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_EA_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
results_path = os.path.join(path_test, file_name)

min_electricity = 0
max_electricity = 2900
min_heating = 0
max_heating = 2400

scatter_data = {}
label_data = {}
cost ={}
emissions = {}
num_clusters_UA = int(editable_data['Cluster numbers'])+2
with open(os.path.join(path_test,'UA_operation.json')) as f:
    scenario_generated = json.load(f)
electricty_UA = []
heating_UA = []
for represent in range(num_clusters_UA):
    for day in range(num_scenarios):
        for hour in range(0,24):
            data_now = scenario_generated[str(represent)][str(day)][str(hour)]
            heating_demand_now = data_now[1]
            electricity_demand_now = data_now[0]
            if heating_demand_now<min_heating:
                heating_demand_now = min_heating
            if electricity_demand_now<min_electricity:
                electricity_demand_now=min_electricity
            if heating_demand_now>max_heating:
                heating_demand_now = max_heating
            if electricity_demand_now>max_electricity:
                electricity_demand_now=max_electricity
            electricty_UA.append(electricity_demand_now)
            heating_UA.append(heating_demand_now)
data_demand = {'Electricity (kWh)': electricty_UA,
        'Heating (kWh)': heating_UA}
df_demand = pd.DataFrame(data_demand)
df_demand.to_csv(os.path.join(path_test , 'UA_demand.csv'))
df_demand = df_demand[min_electricity+0.01<df_demand['Electricity (kWh)']]
df_demand = df_demand[df_demand['Electricity (kWh)']<max_electricity-0.01]
df_demand = df_demand[min_heating+0.01<df_demand['Heating (kWh)']]
df_demand = df_demand[df_demand['Heating (kWh)']<max_heating-0.01]
plt.hist(df_demand['Electricity (kWh)'], bins= 20,facecolor='gold')
plt.xlabel('Electricity (kWh)')
plt.ylabel('Counts')
plt.savefig(os.path.join(results_path ,'electricty_UA'+'.png'),dpi=300,facecolor='w')
plt.close()
plt.hist(df_demand['Heating (kWh)'], bins= 20,facecolor='orangered')
plt.xlabel('Heating (kWh)')
plt.ylabel('Counts')
plt.savefig(os.path.join(results_path ,'heating_UA'+'.png'),dpi=300,facecolor='w')
plt.close()
plt.scatter(df_demand['Electricity (kWh)'],df_demand['Heating (kWh)'],facecolor='purple')
plt.xlabel('Electricity (kWh)')
plt.ylabel('Heating (kWh)')
plt.savefig(os.path.join(results_path ,'electricty_heating'+ '.png'),dpi=300,facecolor='w')
plt.close()

limits_represent = {}
limits_represent[0] = [0,0,1900,7500]
limits_represent[1] = [0,0,1300,10000]
limits_represent[2] = [0,0,1300,10000]
limits_represent[3] = [0,0,2000,10000]
limits_represent[4] = [0,0,1250,10000]
limits_represent[5] = [0,0,1100,10000]
limits_represent[6] = [0,0,1700,10000]
limits_represent[7] = [0,0,1200,10000]
limits_represent[8] = [0,0,1200,10000]
limits_represent[9] = [0,0,1300,10000]
limits_represent[10] = [0,0,2700,10000]

for represent in range(num_clusters):
    scatter_data[represent] = pd.read_csv(os.path.join(results_path,'UA_'+ str(represent)+'_represent_objectives_all.csv'))
    scatter_data[represent] =  scatter_data[represent][scatter_data[represent]['Cost ($)']<limits_represent[represent][2]]
    label_data[represent] = pd.read_csv(os.path.join(results_path,'UA_'+ str(represent)+'_represent_operation_all.csv'))
    label_data[represent] =  label_data[represent][label_data[represent]['Cost ($)']<limits_represent[represent][2]]
    label_data[represent] =  label_data[represent][label_data[represent]['Solar Generation (kWh)']>0]
    cost[represent] = [i/10**3 for i in scatter_data[represent]['Cost ($)']]
    emissions[represent] = [j/10**3 for j in scatter_data[represent]['Emission (kg CO2)']]
def ParetoFront_EFs(path_test,represent):
    fig,ax = plt.subplots()
    c = 'tab:red'
    m = "o"
    #plt.figure(figsize=(8, 5))
    plt.scatter(cost[represent],emissions[represent],color=c,marker=m, s=60, cmap=cmap)
    plt.title('Cost and emissions trade-off in representative day '+str(represent))
    plt.xlabel("Cost (thousand $)")
    plt.ylabel("Emissions (ton $CO_2$)")
    #plt.axis(list_limits)
    plt.savefig(os.path.join(results_path ,'ParetoFront_'+ str(represent)+'.png'),dpi=300,facecolor='w')
    plt.close()
def violin_plot(data_list,type_plot,labels,represent):
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        if type_plot=='cost':
            ax.set_ylabel("Cost (thousand $)")
        if type_plot=='emissions':
            ax.set_ylabel("Emissions (ton $CO_2$)")
    fig, ax2 = plt.subplots(figsize=(7, 5), sharey=True)
    parts = ax2.violinplot(
            data_list, showmeans=False, showmedians=False,
            showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    quartile1_1, medians_1, quartile3_1 = np.percentile(data_list[0], [25, 50, 75])
    quartile1_2, medians_2, quartile3_2 = np.percentile(data_list[1], [25, 50, 75])
    quartile1=[quartile1_1,quartile1_2]
    quartile3=[quartile3_1,quartile3_2]
    medians=[medians_1,medians_2]
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data_list, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # set style for the axes
    set_axis_style(ax2, labels)
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.savefig(os.path.join(results_path,results_path , 'violin_'+type_plot+'_'+str(represent)+'.png'),dpi=300)
    plt.close()
# create test data
data_cost = []
data_emissions= []
labels = []
for represent in range(num_clusters):
    data_cost.append(cost[represent])
    data_emissions.append(emissions[represent])
    labels.append('representative='+str(represent))
violin_plot(data_cost,'cost',labels,represent)
#violin_plot(data_emissions,'emissions',labels,represent)
### Parallel coordinates plot of the sizings
#Ref: https://coderzcolumn.com/tutorials/data-science/how-to-plot-parallel-coordinates-plot-in-python-matplotlib-plotly
def parallel_plots(type_plot,path_test,represent):
    cost = {}
    emissions = {}
    label_data[represent]['CHP (kWh)'] = label_data[represent]['CHP Operation (kWh)']
    label_data[represent]['Boilers (kWh)'] = label_data[represent]['Boilers Operation (kWh)']
    label_data[represent]['Battery (kWh)'] = label_data[represent]['Battery Operation (kWh)']
    label_data[represent]['Grid (kWh)'] = label_data[represent]['Grid Operation (kWh)']
    label_data[represent]['Solar (kWh)'] = label_data[represent]['Solar Generation (kWh)']
    label_data[represent]['Wind (kWh)'] = label_data[represent]['Wind Generation (kWh)']
    label_data[represent]['Emissions (kg)'] = label_data[represent]['Emission (kg CO2)']
    if type_plot == 'cost':
        cols = ['Boilers (kWh)', 'CHP (kWh)', 'Battery (kWh)','Grid (kWh)','Solar (kWh)','Wind (kWh)','Emissions (kg)','Cost ($)']
        color_plot = px.colors.sequential.Blues
    if type_plot == 'emissions':
        cols = ['Boilers (kWh)', 'CHP (kWh)', 'Battery (kWh)','Grid (kWh)','Solar (kWh)','Wind (kWh)','Cost ($)','Emissions (kg)']
        color_plot = px.colors.sequential.Reds
    fig_new = px.parallel_coordinates(label_data[represent], color=cols[-1], dimensions=cols,color_continuous_scale=color_plot)
    fig_new.update_layout(
        font=dict(
            size=22,
        )
    )
    fig_new.write_image(os.path.join(results_path,'Parallel_coordinates_'+type_plot+'_'+str(represent)+'.png'),width=1200, height=350,scale=3)
for represent in range(num_clusters):
    parallel_plots('cost',path_test,represent)
    ParetoFront_EFs(path_test,represent)
#violin_plot(data_cost,'cost',labels,represent)
