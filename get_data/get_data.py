
from __future__ import absolute_import
from __future__ import print_function

from xml.etree import ElementTree as ET
from multiprocessing import Pool
import os
import sys
import time
import subprocess
import random
import string
import numpy as np 
import matplotlib.pyplot as plt
import shutil
import errno
import csv


import FitnessClass
# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
import traci


# def _get_SG_List(originfile):
#     tree = ET.parse(originfile)
#     root = tree.getroot()
#     str_g = 'g'
#     str_r = 'r'
#     SG_list = []
#     for child in root.iter('phase'):
#         state_str_tolower = child.get('state').lower()
#         checkhasyellow = string.find(state_str_tolower, 'y') != -1
#         if checkhasyellow != 1:
#             g_count = state_str_tolower.count(str_g)
#             r_count = state_str_tolower.count(str_r)
#             total = g_count + r_count
#             SG_list.append(g_count / total)
#     return SG_list


# def _getFitness(population, SG_list):
#     assert len(population) == len(
#         SG_list), 'the (population and SG_list) length are not the same'
#     fitness = 0
#     for i in range(len(population)):
#         fitness = fitness + population[i] * SG_list[i]
#     return fitness


def _modify_add_xml_phase_duration(population, inputfile):
    tree = ET.parse(inputfile)
    root = tree.getroot()
    index = 0
    for child in root.iter('phase'):
        checkhasyellow = string.find(child.get('state').lower(), 'y') != -1
        if checkhasyellow == 1:
            child.set('duration', '5')
        else:
            child.set('duration', str(population[index]))
            index += 1
    tree.write(inputfile)

# get Population Size from xml (number of phase)


def _getSize(originfile):
    tree = ET.parse(originfile)
    root = tree.getroot()

    # for child in root.iter('tlLogic'):
    #     print(child.get('id'))
    #     # print(child.iter('phase'))
    #     size1 = 0
    #     for i in child.iter('phase'):
    #         size1 += 1
    #     print(size1)

    size = 0
    for child in root.iter('phase'):
        checkhasyellow = string.find(child.get('state').lower(), 'y') != -1
        if checkhasyellow != 1:
            size += 1
    return size


def _getdurationlist():
    tree = ET.parse('osm.add.xml')
    root = tree.getroot()
    durationlist = []
    for child in root.iter('phase'):
        duration = int(child.get('duration'))
        durationlist.append(duration)
    return durationlist


def _getFitnessElement(outputfile, inputfile):
    tree = ET.parse(outputfile)
    root = tree.getroot()
    complete_duration = 0.0
    incomplete_Veh_num = 0
    complete_Veh_num = 0
    waitingTime = 0
    for child in root.iter('tripinfo'):
        if child.get('arrival') == "-1.00":  # vaporized = 0 mean vehicle's not finish on journey
            incomplete_Veh_num += 1
        else:
            complete_duration += float(child.get('duration'))
            complete_Veh_num += 1
        waitingTime += int(child.get('waitSteps'))
    Cr = _getCr(inputfile)
    element = FitnessClass.Fitness(complete_duration, waitingTime, incomplete_Veh_num, Cr, complete_Veh_num)
    return element


def _getCr(inputfile):
    tree = ET.parse(inputfile)
    root = tree.getroot()
    Cr = 0
    for child in root.iter('phase'):
        duration = int(child.get('duration'))
        state = child.get('state').lower().translate(None, "y")
        if state != 0:
            r_count = state.count('r') if state.count('r') is not 0 else 1
            g_count = state.count('g') if state.count('g') is not 0 else 1
            rate = g_count / r_count
            Cr += duration * rate
    return Cr

def _getFitness(element):
    # print(element.__str__())
    fitness = FitnessClass.Fitness._calFitness(element)
    return fitness


def run_simulate(i, sumoCmd):
    print('Run task %s ...' % i)
    subprocess.call(sumoCmd, stdout=sys.stdout, stderr=sys.stderr)
    print('Task %s finish ' % i)


def _PSO(roundtime, population_counts, iterations):
    np.set_printoptions(precision=3, suppress=True)

    w = 0.5
    c1 = 2.05
    c2 = 2.05
    upper = 60
    lower = 5

    gloabpopulation = np.zeros(1)
    gloabfitness = np.array(
        [np.finfo(np.float32).max for _ in range(iterations)], dtype=float)

    originfile = "sumo_input/copy_osm.add.xml"
    # Initial
    # ----------------------------------------------------------------------
    population = np.array(
        [np.random.randint(lower, upper, _getSize(originfile)) for _ in range(population_counts)])
    
    fitnesses = np.zeros(population_counts)

    complete_duration = np.zeros(population_counts)
    waitingTime = np.zeros(population_counts)
    incomplete_Veh_num = np.zeros(population_counts)
    complete_Veh_num = np.zeros(population_counts)

    # ----------------------------------------------------------------------
    bestpopulation = np.zeros(population.shape)
    bestfitnesses = np.full(len(population), 1000, dtype=float)

    speed = np.zeros(population.shape)

    inputfile = []
    outputfile = []
    for i in range(len(population)):
        inputfile.append("sumo_input/addfile/%d_osm.add.xml" % i)
        outputfile.append("sumo_output/%d_out.xml" % i)
        shutil.copyfile(originfile, inputfile[i])

    # start run algo
    for iteration in range(iterations):
        print("---iteration", iteration)
        # start simulate
        p = Pool(len(population))
        for i in range(len(population)):
            _modify_add_xml_phase_duration(population[i], inputfile[i])
            sumoCmd = [checkBinary('sumo'), "-c", "3osm.sumocfg",
                       "-a", inputfile[i], "--tripinfo-output", outputfile[i]]
            p.apply_async(run_simulate, args=(i, sumoCmd))
        # close pool ,do not accept new task
        p.close()
        # let parent wait ,after child all have finish will open
        p.join()
        print("all done")

        # cal each population fitness and save to fitnesses
        for i in range(len(population)):

            element = _getFitnessElement(outputfile[i], inputfile[i])

            fitnesses[i] =  _getFitness(element)
            complete_duration[i] = element.complete_duration
            waitingTime[i] = element.waiting_time
            incomplete_Veh_num[i] = element.incomplete_Veh_num
            complete_Veh_num[i] = element.complete_Veh_num

            if(fitnesses[i] < bestfitnesses[i]):
                bestfitnesses[i] = fitnesses[i]
                bestpopulation[i] = population[i]

        file_path_list = _create_output_file(iteration,roundtime,population_counts)

        output_value_list = [population, fitnesses,complete_duration,waitingTime,incomplete_Veh_num,complete_Veh_num]
        _output_file(file_path_list,output_value_list)

        print("bfitnesses : {}".format(np.array(bestfitnesses)))

        # record gloab fitness and population
        if(np.min(bestfitnesses) < np.min(gloabfitness)):
            gloabfitness[iteration] = np.min(bestfitnesses)
            gloabpopulation = bestpopulation[np.argmin(bestfitnesses)]

        else:
            gloabfitness[iteration] = np.min(gloabfitness)

        # update speed
        for i in range(len(population)):
            speed[i] = w * speed[i] + c1 * random.uniform(0, 1) * (
                bestpopulation[i] - population[i]) + c2 * random.uniform(0, 1) * (gloabpopulation - population[i])
        # update population
        for i in range(len(population)):
            population[i] = population[i] + speed[i]

        # check upper and lower bound(i thinks is cost lot,increase much time)
        np.clip(population, 5, 60, out=population)

        print("gloabfitness : {}".format(np.array(gloabfitness)))

    return np.min(gloabfitness)


def _output_file(file_path_list,output_value_list):
    assert len(file_path_list) == len(output_value_list)
    for i in range(len(file_path_list)):
        with open(file_path_list[i], 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(output_value_list[i])


    # with open(population_filename, 'a') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(population)

    # with open(fitnesses_filename, 'a') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(fitnesses)


def _create_output_file(iteration,roundtime,population_counts):
    # create a filename list 
    filename_list = ["population.csv","fitnesses.csv","complete_duration.csv","waitingTime.csv","incomplete_Veh_num.csv","complete_Veh_num.csv"]
    file_path_list = []
    for filename in filename_list:
        file_path = "output_file/{}/{}".format(str(roundtime),filename)
        Is_file_exist(file_path)
        file_path_list.append(file_path)
    
    if roundtime == 0 and iteration == 0:
        for filename in file_path_list:
            with open(filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(np.arange(population_counts))

    return file_path_list


def Is_file_exist(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


if __name__ == "__main__":
    # originfile = "sumo_input/copy_osm.add.xml"
    
    # _getSize(originfile)
    start = time.time()

    sumoBinary = checkBinary('sumo')
    roundtimes = 5
    
    for roundtime in range(roundtimes):
        print("roundtime{}".format(roundtime))
        population_counts = 100
        iteration_amounts = 300
        fitness = _PSO(roundtime, population_counts, iteration_amounts)
        result = np.array2string(np.array([str(roundtime), str(population_counts), str(
            iteration_amounts), str(fitness)]), separator=',')
        print(result)

    end = time.time()
    time_taken = end - start
    print("cost time : {}".format(time_taken))
