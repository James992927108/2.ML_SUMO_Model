
import csv
import os
import sys

import numpy as np
import pandas as pd


def load_data():
    filename_list = ["population.csv", "fitnesses.csv", "complete_duration.csv",
                     "waitingTime.csv", "incomplete_Veh_num.csv", "complete_Veh_num.csv"]

    if check_file_is_exist(filename_list):
        rootdir = os.getcwd() + '/get_data/output_file'
        for foldercount in range(len(os.listdir(rootdir))):
            for filename in filename_list:
                all_to_csv = pd.read_csv(
                    '{}/{}/{}'.format(rootdir, foldercount, filename))
                all_to_csv.to_csv("all_"+str(filename), mode='a')

    c_data_set = integration_csv_to_data_set(filename_list)
    return c_data_set


def check_file_is_exist(filename_list):
    for filename in filename_list:
        if os.path.isfile("all_"+str(filename)):
            print("exist")
            return False
    print("not exist")
    return True

# ['index', 'complete_Veh_num', 'complete_duration', 'fitnesses', 'incomplete_Veh_num', 'population', 'waitingTime']
# 'complete_Veh_num', 'complete_duration', 'fitnesses', 'incomplete_Veh_num', 'index', 'waitingTime'
def integration_csv_to_data_set(filename_list):
    _dict = {}
    for filename in filename_list:
        csv_file = pd.read_csv("all_"+str(filename))
        print("{},{}".format(filename,csv_file.shape))
        _dict[str(filename).replace('.csv', '')] = np.array( csv_file.iloc[:, 1:]).flatten()

    data_set = pd.DataFrame(_dict)
    # remove the same column,reset_index is important
    data_set = data_set.drop_duplicates().reset_index()


    c_array = parse_poplations(data_set['population'])
    c_data_set = pd.DataFrame(data=c_array)
    # remove origin column "population" and "index"

    data_set = data_set.drop(columns=['population'])
    data_set = data_set.drop(columns=['index'])

    final_data_set = pd.concat([data_set,c_data_set],axis=1,sort=False)
    # c_data_set = pd.DataFrame(columns=[
    #     'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'], data=c_array)
    # c_data_set['fitnesses'] = data_set['fitnesses']

    return final_data_set

def parse_poplations(data_set):
    c_list = []
    for i in range(data_set.shape[0]):
        # print(i)
        data_set_i = data_set[i].replace(r'[', '').replace(r']', '')
        data_set_i = np.fromstring(data_set_i, dtype=int, sep=' ')
        # c = classity_population_element(data_set_i)
        # c_list.append(c)
        c_list.append(data_set_i)
    c_array = np.asarray(c_list)
    return c_array


def classity_population_element(data_set_i):
    c = np.zeros(11, dtype=int)
    for i in range(data_set_i.shape[0]):
        index = (data_set_i[i] // 5) - 1
        if(data_set_i[i] == 60):
            index -= 1
        # print(data_set_i[i], index)
        c[index] += 1
    return c