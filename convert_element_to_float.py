import pandas as pd
import numpy as np


def read(filename):
    data = pd.read_csv(filename)
    n_data = np.array(data)
    return n_data


def conver(n_data):
    n_data = conver_to_correct_float(n_data)
    return n_data


def write(n_data, filename):
    df_n_data = pd.DataFrame(n_data, index=None)
    df_n_data.to_csv(filename, mode='w', index=False)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def conver_to_correct_float(n_data):
    for i in range(n_data.shape[0]):
        for j in range(n_data.shape[1]):
            is_number(n_data[i][j])
            if not is_number(n_data[i][j]):
                n_data[i][j] = n_data[i][j][:-2]
                if n_data[i][j][:-1] == '.':
                    print(a)
    return n_data


if __name__ == "__main__":
    model_name_list = ["complete_duration","waitingTime", "incomplete_Veh_num", "complete_Veh_num"]
    for filename in model_name_list:
        filename = 'all_' + str(filename)+'.csv'
        n_data = read(filename)
        print("before {} shape {}".format(filename,n_data.shape))
        n_data = conver(n_data)
        write(n_data, filename)
        print("after {} shape {}".format(filename,n_data.shape))

