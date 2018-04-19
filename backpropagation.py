# !/usr/bin/python

# BP On Feed-Forward Network for Time Series

# based on: https://github.com/rohitash-chandra/FNN_TimeSeries
# based on: https://github.com/rohitash-chandra/mcmc-randomwalk

# Rohitash Chandra, Centre for Translational Data Science
# University of Sydey, Sydney NSW, Australia.  2017 c.rohitash@gmail.conm
# https://www.researchgate.net/profile/Rohitash_Chandra

# Gary Wong
# University of the South Pacific, Fiji
# gary.wong.fiji@gmail.com


import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import os
import shutil


class Network:
    def __init__(self, Topo, Train, Test):
        self.lrate = 0.1
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed()

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def RMSE(self, actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired, vanilla):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta[0])

        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta[0])

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]

    def evaluate_solution(self, data):


        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for pat in xrange(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            Desired[:] = data[pat, self.Top[0]:]

            self.ForwardPass(Input)
            try:
                fx[pat] = self.out
            except:
               print 'Error'

        return fx

    def ForwardFitnessPass(self, data, w):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.

        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)
        actual = np.zeros(size)

        for pat in xrange(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            Desired[:] = data[pat, self.Top[0]:]

            actual[pat] = data[pat, self.Top[0]:]

            self.ForwardPass(Input)
            try:
                fx[pat] = self.out
            except:
               print 'Error'


        # FX holds calculated output

        return self.RMSE(actual, fx)

    def ForwardFitnessPassBP(self, data):  # BP with SGD (Stocastic BP)

        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)
        actual = np.zeros(size)

        for pat in xrange(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            Desired[:] = data[pat, self.Top[0]:]

            actual[pat] = data[pat, self.Top[0]:]

            self.ForwardPass(Input)
            try:
                fx[pat] = self.out
            except:
               print 'Error'


        # FX holds calculated output

        return self.RMSE(actual, fx)



class Backpropagation:
    def __init__(self):
        print 'initialize'
    def Run(self, Data, Network, Topology):
        NetworkSize = (Topology[0] * Topology[1]) + (Topology[1] * Topology[2]) + Topology[1] + Topology[2]
        InitialWeights =  np.random.uniform(-5,5,NetworkSize)

        Network.decode(InitialWeights)

        size = Data.shape[0]

        Input = np.zeros((1, Topology[0]))  # temp hold input
        Desired = np.zeros((1, Topology[2]))
        fx = np.zeros(size)


        Gen = 0
        while(Gen < 5000):
            for sample in Data:
                Gen += 1
                Input[:] = sample[0:Topology[0]]
                Desired[:] = sample[Topology[0]:]

                Network.ForwardPass(Input)
                Network.BackwardPass(Input, Desired, 0)

                print 'Epoch: ' + str(Gen) + ' Fitness: ' + str(Network.ForwardFitnessPassBP(Data))

class Graphing:
    def __init__(self):
        print 'Graphing Initialized'
    def PlotGraphs(self, net, run, testdata, traindata, topology):

        testsize = testdata.shape[0]
        trainsize = traindata.shape[0]


        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        ytestdata = testdata[:, topology[0]]
        ytraindata = traindata[:, topology[0]]

        trainout = net.evaluate_solution(traindata)
        testout = net.evaluate_solution(testdata)

        plt.plot(x_train, ytraindata, label='actual')
        plt.plot(x_train, trainout, label='predicted (train)')


        plt.legend(loc='upper right')
        plt.title("Plot of Train Data vs MCMC Uncertainty ")

        if not os.path.exists('mcmcresults/run' + str(run) + '/'):
            os.makedirs('mcmcresults/run' + str(run) + '/')

        plt.savefig('mcmcresults/run' + str(run) + '/dmcmc_train.png')
        plt.savefig('mcmcresults/run' + str(run) + '/dmcmc_train.svg', format='svg', dpi=600)
        plt.clf()

        plt.plot(x_test, ytestdata, label='actual')
        plt.plot(x_test, testout, label='predicted (test)')

        plt.legend(loc='upper right')
        plt.title("Plot of Test Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/run' + str(run) + '/dmcmc_test.png')
        plt.savefig('mcmcresults/run' + str(run) + '/dmcmc_test.svg', format='svg', dpi=600)
        plt.clf()




def main():

    if os.path.exists('mcmcresults'):
        shutil.rmtree('mcmcresults/', ignore_errors=True)

    else:
        os.makedirs('mcmcresults')

    start = time.time()


    hidden = 5
    input = 4  #
    output = 1

    traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
    testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt")  #

    # traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
    # testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")


    # if problem == 1:
    #     traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
    #     testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")  #
    # if problem == 2:

    # if problem == 3:
    #     traindata = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
    #     testdata = np.loadtxt("Data_OneStepAhead/Mackey/test.txt")  #

    print(traindata)

    topology = [input, hidden, output]

    random.seed(time.time())

    runs = 5

    RUNRMSE = []

    net = Network(topology, traindata, testdata)

    graph = Graphing()
    for i in range(0, runs):
        Procedure = Backpropagation()
        Procedure.Run(traindata, net, topology)
        graph.PlotGraphs(net, i,testdata,traindata,topology)


    print 'End simulation'
    end = time.time()
    print str(end - start) + ' Seconds'


if __name__ == "__main__": main()
