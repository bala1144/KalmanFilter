import cv2
import math
import numpy as np

class kalmanFilter:
    def __init__(self, numOf2DPoints = 1):

        self.measureNum = 2 * numOf2DPoints # we can only measure x, y from the sensor
        self.stateNum = 2 * self.measureNum # constraints the state and velocity model[x,y,dx.dy]
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


    def setFilterParams(self):
        self.kalman.processNoiseCov = np.eye(self.stateNum,dtype= np.float32) * 0.03  # process uncertainty
        self.kalman.measurementNoiseCov = np.eye(self.measureNum,dtype= np.float32) * 0.03 # measurement noise uncertainity
        # self.kalman.errorCovPost = np.repeat(1,(self.stateNum,self.stateNum))

    def predictKeyPoints(self, input_points):
        # get new kalman filter predictionS
        predicted_stateVector = self.kalman.predict() # it predicts the next state vector
        prediction_keypoints = np.matmul(self.kalman.measurementMatrix, predicted_stateVector)
        self.kalman.correct(input_points)

        return prediction_keypoints

def displayResults(x,y,noise,prediction=None):
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.plot(x, y+noise)
    if prediction is not None:
        plt.plot(prediction[:,0], prediction[:,1])
    plt.xlabel('sample(n)')
    plt.ylabel('voltage(V)')
    plt.show()

def SampleSinfunction():
    import numpy as np
    f = 1
    Fs = sample = 100
    noise = np.random.normal(0,1,sample) * 0.2
    x = np.arange(sample)
    y = np.sin(2 * np.pi * f * x / Fs)
    return x, y,noise

if (__name__ == "__main__"):
    print('Test kalman filter')
    x, y,noise = SampleSinfunction()
    displayResults(x,y,noise)
    KF = kalmanFilter()
    KF.setFilterParams()
    KF_filtered_point = []
    
    for i in range(len(x)):
        dlib_points =  np.array([x[i],y[i]],dtype= np.float32)
        prediction_keypoints = KF.predictKeyPoints(dlib_points)
        KF_filtered_point.append(prediction_keypoints.reshape(2,))


    displayResults(x,y,noise,np.asarray(KF_filtered_point))

   
    