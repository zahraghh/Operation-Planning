import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import sklearn.datasets, sklearn.decomposition
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn_extra
from scipy import stats
from scipy.stats import kurtosis, skew
from collections import defaultdict
import statistics
from itertools import chain
from scipy.interpolate import interp1d
from collections import defaultdict
from nested_dict import nested_dict
def kmedoid_clusters(path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city = editable_data['city']
    save_path = os.path.join(path_test, str('Scenario Generation') , city)
    representative_days_path =  os.path.join(save_path,'Representative days')
    if not os.path.exists(representative_days_path):
        os.makedirs(representative_days_path)
    folder_path = os.path.join(path_test,str(city))
    GTI_distribution = pd.read_csv(os.path.join(folder_path,'best_fit_GTI.csv'))
    wind_speed_distribution = pd.read_csv(os.path.join(folder_path,'best_fit_wind_speed.csv'))
    range_data = ['low','medium','high']
    scenario_genrated = {}
    scenario_probability = defaultdict(list)
    solar_probability = defaultdict(list)
    wind_probability = defaultdict(list)
    for i in range(8760):
        if GTI_distribution['Mean'][i] == 0:
            solar_probability['low'].append(1/3)
            solar_probability['medium'].append(1/3)
            solar_probability['high'].append(1/3)
        ## If Solar GTI is normal: from Rice & Miller  low = 0.112702 = (x-loc)/scale -->  =tick
        elif GTI_distribution['Best fit'][i] == 'norm':
            solar_probability['low'].append(0.166667)
            solar_probability['medium'].append(0.666667)
            solar_probability['high'].append(0.166667)
        ## If Solar GTI is uniform: from Rice & Miller low = 0.112702 (i - loc)/scale
        elif GTI_distribution['Best fit'][i] == 'uniform':
            solar_probability['low'].append(0.277778)
            solar_probability['medium'].append(0.444444)
            solar_probability['high'].append(0.277778)
        ## If Solar GTI is expon: from Rice & Miller low = 0.415775 (i - loc)/scale, scale/scale)
        elif GTI_distribution['Best fit'][i] == 'expon':
            solar_probability['low'].append(0.711093)
            solar_probability['medium'].append(0.278518)
            solar_probability['high'].append(0.010389)

        if wind_speed_distribution['Mean'][i] == 0:
            wind_probability['low'].append(1/3)
            wind_probability['medium'].append(1/3)
            wind_probability['high'].append(1/3)
        ## If Solar GTI is normal: from Rice & Miller  low = 0.112702 = (x-loc)/scale -->  =tick
        elif wind_speed_distribution['Best fit'][i] == 'norm':
            wind_probability['low'].append(0.166667)
            wind_probability['medium'].append(0.666667)
            wind_probability['high'].append(0.166667)
        ## If Solar GTI is uniform: from Rice & Miller low = 0.112702 (i - loc)/scale
        elif wind_speed_distribution['Best fit'][i] == 'uniform':
            wind_probability['low'].append(0.277778)
            wind_probability['medium'].append(0.444444)
            wind_probability['high'].append(0.277778)
        ## If Solar GTI is expon: from Rice & Miller low = 0.415775 (i - loc)/scale, scale/scale)
        elif wind_speed_distribution['Best fit'][i] == 'expon':
            wind_probability['low'].append(0.711093)
            wind_probability['medium'].append(0.278518)
            wind_probability['high'].append(0.010389)
    p_solar = nested_dict()
    p_wind = nested_dict()
    scenario_number = {}
    num_scenario = 0
    #laod the energy deamnd, solar, wind, and electricity emissions from scenario generation file
    for i_demand in range_data:
        for i_solar in range_data:
            for i_wind in range_data:
                for i_emission in range_data:
                    if i_demand=='low':
                        p_demand = 0.277778
                    elif i_demand=='medium':
                        p_demand = 0.444444
                    elif i_demand=='high':
                        p_demand = 0.277778
                    if i_emission=='low':
                        p_emission = 0.166667
                    elif i_emission=='medium':
                        p_emission = 0.666667
                    elif i_emission=='high':
                        p_emission = 0.166667
                    for day in range(365):
                        p_solar[i_solar][day] = sum(solar_probability[i_solar][day*24:(day+1)*24])/(sum(solar_probability[range_data[0]][day*24:(day+1)*24])+sum(solar_probability[range_data[1]][day*24:(day+1)*24])+sum(solar_probability[range_data[2]][day*24:(day+1)*24]))
                        p_wind[i_wind][day] = sum(wind_probability[i_wind][day*24:(day+1)*24])/(sum(wind_probability[range_data[0]][day*24:(day+1)*24])+sum(wind_probability[range_data[1]][day*24:(day+1)*24])+sum(wind_probability[range_data[2]][day*24:(day+1)*24]))
                        scenario_probability['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission].append(p_demand*p_solar[i_solar][day]*p_wind[i_wind][day]*p_emission)

                    scenario_number['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission]=  num_scenario
                    num_scenario = num_scenario + 1
                    scenario_genrated['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission] = pd.read_csv(os.path.join(save_path, 'D_'+i_demand+'_S_'+i_solar+'_W_'+i_wind+'_C_'+i_emission+'.csv'), header=None)
    features_scenarios = defaultdict(list)
    features_scenarios_list = []
    features_probability_list = []
    features_scenarios_nested = nested_dict()
    k=0
    days= 365
    for scenario in scenario_genrated.keys():
        scenario_genrated[scenario]=scenario_genrated[scenario]
        for i in range(days):
            if i==0:
                data = scenario_genrated[scenario][1:25]
            else:
                data = scenario_genrated[scenario][25+(i-1)*24:25+(i)*24]
            #Total electricity, heating, solar, wind, EF.
            daily_list =list(chain(data[0].astype('float', copy=False),data[1].astype('float', copy=False),
            data[2].astype('float', copy=False),data[3].astype('float', copy=False),data[6].astype('float', copy=False)))
            features_scenarios[k*days+i] = daily_list
            features_scenarios_nested[scenario][i] = features_scenarios[k*days+i]
            features_scenarios_list.append(features_scenarios[k*days+i])
            features_probability_list.append(scenario_probability[scenario][i])
        k = k+1
    A = np.asarray(features_scenarios_list)
    #Convert the dictionary of features to Series
    standardization_data = StandardScaler()
    A_scaled = standardization_data.fit_transform(A)
    # Create a PCA instance: pca
    pca = PCA(n_components=int(editable_data['PCA numbers']))
    principalComponents = pca.fit(A_scaled)
    scores_pca = pca.transform(A_scaled)
    #print('Score of features', scores_pca)
    #print('Explained variance ratio',pca.explained_variance_ratio_)
    # Plot the explained variances
    # Save components to a DataFrame
    features = range(pca.n_components_)
    search_optimum_feature= editable_data['Search optimum PCA']
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.edgecolor'] = 'black'
    if search_optimum_feature == 'yes':
        print('Defining the optimum number of features in the PCA method: ')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(features, pca.explained_variance_ratio_.cumsum(), color='tab:blue')
        ax.set_xlabel('PCA features',fontsize=BIGGER_SIZE)
        ax.set_ylabel('Cumulative explained variance',fontsize=BIGGER_SIZE)
        ax.set_xticks(features)
        ax.set_title('The user should set a limit on the explained variance value and then, select the optimum number of PCA features',fontsize=BIGGER_SIZE)
        plt.savefig(os.path.join(sys.path[0], 'Explained variance vs PCA features.png'),dpi=300,facecolor='w')
        plt.close()
        print('"Explained variance vs PCA features" figure is saved in the directory folder')
        print('You can use the figure to select the optimum number of features' )
        print('You should enter the new optimum number of features in EditableFile.csv file and re-run this part')
        plt.close()
    PCA_components = pd.DataFrame(scores_pca)
    inertia_list = []
    search_optimum_cluster = editable_data['Search optimum clusters'] # if I want to search for the optimum number of clusters: 1 is yes, 0 is no
    cluster_range = range(2,20,1)
    if search_optimum_cluster=='yes':
        print('Defining the optimum number of clusters: ')
        fig, ax = plt.subplots(figsize=(12, 6))

        for cluster_numbers in cluster_range:
            kmedoids = KMedoids(n_clusters=cluster_numbers, init="random",max_iter=1000,random_state=0).fit(scores_pca)
            inertia_list.append(kmedoids.inertia_)
            plt.scatter(cluster_numbers,kmedoids.inertia_)
            print('Cluster number:', cluster_numbers, '  Inertia of the cluster:', int(kmedoids.inertia_))
        ax.set_xlabel('Number of clusters',fontsize=BIGGER_SIZE)
        ax.set_ylabel('Inertia',fontsize=BIGGER_SIZE)
        ax.set_title('The user should use "Elbow method" to select the number of optimum clusters',fontsize=BIGGER_SIZE)
        ax.plot(list(cluster_range),inertia_list)
        ax.set_xticks(np.arange(2,20,1))
        plt.savefig(os.path.join(sys.path[0], 'Inertia vs Clusters.png'),dpi=300,facecolor='w')
        plt.close()
        print('"Inertia vs Clusters" figure is saved in the directory folder')
        print('You can use the figure to select the optimum number of clusters' )
        print('You should enter the new optimum number of clusters in EditableFile.csv file and re-run this part')

    cluster_numbers= int(editable_data['Cluster numbers'])
    kmedoids_org = KMedoids(n_clusters=cluster_numbers, init="random",max_iter=1000,random_state=4).fit(A)
    kmedoids = KMedoids(n_clusters=cluster_numbers, init="random",max_iter=1000,random_state=4).fit(scores_pca)
    label = kmedoids.fit_predict(scores_pca)
    #filter rows of original data
    probability_label = defaultdict(list)
    index_label = defaultdict(list)
    index_label_all = []
    filtered_label={}
    for i in range(cluster_numbers):
        filtered_label[i] = scores_pca[label == i]
        index_cluster=np.where(label==i)
        if len(filtered_label[i])!=0:
            index_cluster = index_cluster[0]
            for j in index_cluster:
                probability_label[i].append(features_probability_list[j])
                index_label[i].append(j)
                index_label_all.append(j)
        else:
            probability_label[i].append(0)
    sum_probability = []

    for key in probability_label.keys():
        sum_probability.append(sum(probability_label[key]))

    plt.scatter(filtered_label[i][:,0] , filtered_label[i][:,1] )
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    #plt.show()
    plt.close()
    plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    #plt.show()
    plt.close()

    #print(kmedoids.predict([[0,0,0], [4,4,4]]))
    #print(kmedoids.cluster_centers_,kmedoids.cluster_centers_[0],len(kmedoids.cluster_centers_))
    scores_pca_list={}
    clusters={}
    clusters_list = []
    label_list = []
    data_labels={}
    data_all_labels = defaultdict(list)
    for center in range(len(kmedoids.cluster_centers_)):
        clusters['cluster centers '+str(center)]= kmedoids.cluster_centers_[center]
        clusters_list.append(kmedoids.cluster_centers_[center].tolist())
    for scenario in range(len(scores_pca)):
        scores_pca_list[scenario]=scores_pca[scenario].tolist()
        data_all_labels[kmedoids.labels_[scenario]].append(standardization_data.inverse_transform(pca.inverse_transform(scores_pca_list[scenario])))
        scores_pca_list[scenario].insert(0,kmedoids.labels_[scenario])
        data_labels['labels '+str(scenario)]= scores_pca_list[scenario]
        label_list.append(scores_pca[scenario].tolist())
    df_clusters= pd.DataFrame(clusters)
    df_labels = pd.DataFrame(data_labels)
    df_clusters.to_csv(os.path.join(representative_days_path , 'cluster_centers_C_'+str(len(kmedoids.cluster_centers_))+'_L_'+str(len(kmedoids.labels_))+'.csv'), index=False)
    df_labels.to_csv(os.path.join(representative_days_path , 'labels_C_'+str(len(kmedoids.cluster_centers_))+'_L_'+str(len(kmedoids.labels_))+'.csv'), index=False)
    return data_all_labels
