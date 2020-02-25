# code is modified from opencv python example
# https://github.com/tobybreckon/python-examples-cv/blob/master/kalman_tracking_live.py

import cv2
import math
import numpy as np

class kalmanFilter:
    def __init__(self):

        self.measureNum = 2 # we can only measure x, y from the sensor
        self.stateNum = 2 * self.measureNum # constraints the state and velocity model[x,y,dx,dy]
        self.kalman = cv2.KalmanFilter(self.stateNum,self.measureNum,0) # creating the kalman filter

        transitionMatrix = np.zeros((self.stateNum,self.stateNum),dtype= np.float32) # denoted as F ,to extrapolate the next state from the current state
        for i in range(0,self.stateNum):
            for j in range(0,self.stateNum):
                if (i == j or (j - self.measureNum) == i):
                    transitionMatrix[i, j] = np.float32(1.0)
                else:
                    transitionMatrix[i, j] = np.float32(0.0)

        self.kalman.transitionMatrix = transitionMatrix

        # create the measurement matrix
        # measurement matrix is of size measureNum * stateNum i.e) 2 x 4
        measurementMatrix = np.eye(self.stateNum,dtype= np.float32)[:self.measureNum,:]
        self.kalman.measurementMatrix = measurementMatrix        
        # print('measurement Matrix',measurementMatrix )


    def setFilterParams(self, process_uncertainty = 0.03, noise_uncertainity = 0.03):
        self.kalman.processNoiseCov = np.eye(self.stateNum,dtype= np.float32) * process_uncertainty  # process uncertainty
        self.kalman.measurementNoiseCov = np.eye(self.measureNum,dtype= np.float32) * noise_uncertainity # measurement noise uncertainity

    def filterPoints(self, input_points):
        # get new kalman filter predictionS
        predicted_stateVector = self.kalman.predict() # it predicts the next state vector
        prediction_keypoints = np.matmul(self.kalman.measurementMatrix, predicted_stateVector)
        self.kalman.correct(input_points)

        return prediction_keypoints

def plotSinFunction(x,y,noise,prediction=None):
    # function adapted from the stackoverflow comment
    # https://stackoverflow.com/questions/22566692/python-how-to-plot-graph-sine-wave/34442729
    import matplotlib.pyplot as plt
    # plt.plot(x, y)
    plt.plot(x, y+noise)
    if prediction is not None:
        plt.plot(prediction[:,0], prediction[:,1])
    plt.xlabel('sample(n)')
    plt.ylabel('voltage(V)')
    plt.show()

def SampleSinFunction():
    # function adapted from the stackoverflow comment
    # https://stackoverflow.com/questions/22566692/python-how-to-plot-graph-sine-wave/34442729
    import numpy as np
    f = 1
    Fs = sample = 100
    noise = np.random.normal(0,1,sample) * 0.2
    x = np.arange(sample)
    y = np.sin(2 * np.pi * f * x / Fs)
    return x, y,noise


if (__name__ == "__main__"):
    print('Basic kalman filter')
    x, y,noise = SampleSinFunction()
    plotSinFunction(x,y,noise)

    KF = kalmanFilter()
    KF.setFilterParams()
    KF_filtered_point = []
    
    # for every sampled sin function, run the kalman filter
    for i in range(len(x)):
        data_points =  np.array([x[i],y[i]],dtype= np.float32)
        predicted_points = KF.filterPoints(data_points)
        KF_filtered_point.append(predicted_points.reshape(2,))

    plotSinFunction(x,y,noise,np.asarray(KF_filtered_point))

   
    