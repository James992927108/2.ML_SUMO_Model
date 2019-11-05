import numpy as np
class Fitness:

    simulationTime = 500
    def __init__(self, complete_duration, waitingTime, incomplete_Veh_num, Cr, complete_Veh_num):
        self.complete_duration = complete_duration
        self.waiting_time = waitingTime
        self.incomplete_Veh_num = incomplete_Veh_num
        self.complete_Veh_num = complete_Veh_num
        
        self.incomplete_time = incomplete_Veh_num * self.simulationTime
        self.Cr = Cr
        self.sqar_complete_Veh_num = np.square(complete_Veh_num)

    def _calFitness(self):
        numerator = self.complete_duration + self.waiting_time + self.incomplete_time
        denominator = self.sqar_complete_Veh_num + self.Cr
        fitness = numerator / denominator
        return fitness

    def __str__(self):
        return '({0}, {1}, {2},{3},{4})'.format(self.complete_duration,
                                                self.waiting_time,
                                                self.incomplete_time,
                                                self.Cr,
                                                self.sqar_complete_Veh_num)
