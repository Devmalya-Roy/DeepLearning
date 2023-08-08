# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.datasets import fashion_mnist,mnist
import numpy as np
import matplotlib.pyplot as plt
import wandb
import math
import random
import matplotlib.pyplot as plt
from sklearn import metrics


from scipy.special import expit

wandb.login()



class SimpleClassifier:
    # params = {"w": [], "b": []}
    # layers = {"a":[], "h": []}
    
    def __init__(self, conf):
        # start a new wandb run to track this script
        self.conf = conf
        print("from backprop")
        print(conf)

        # Start a run, tracking hyperparameters
        # run = wandb.init(
        #     # set the wandb project where this run will be logged
        #     project = self.conf["wandb_project"],
        #     # track hyperparameters and run metadata with wandb.config
        #     config = self.conf
        # )


        if(conf["dataset"] == "fashion_mnist"):
            (self.train_images, self.train_labels),(self.test_images, self.test_labels) = fashion_mnist.load_data()
        elif(conf["dataset"] == "mnist"):
            (self.train_images, self.train_labels),(self.test_images, self.test_labels) = mnist.load_data()
        else:
            raise Exception("no matching Dataset found")
            
        #print(self.train_images)
    def draw_samples(self):
        num_row = 2
        num_col = 5
        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))

        for i in range(len(np.unique(self.train_labels))):
            ax = axes[i//num_col, i%num_col]
            index = np.where(self.train_labels == i)
            index = index[0][0]
            ax.imshow(self.train_images[index], cmap='gray')
            ax.set_title('Label: {}'.format(i))
        plt.tight_layout()
        plt.show()
    
    def init_params_zeros(self, dimensions):
        params = {"w" : [[] for i in range(len(dimensions) - 1)], 
        "b": [[] for i in range(len(dimensions) - 1)]}

        for index in range(len(dimensions) - 1):
            w = np.zeros([dimensions[index + 1], dimensions[index]])
            b = np.zeros(dimensions[index + 1])
            params["w"][index] = w
            params["b"][index] = b
        
        return params
    
    def int_param_xavier(self, dimensions):
        params = {"w" : [[] for i in range(len(dimensions) - 1)], 
        "b": [[] for i in range(len(dimensions) - 1)]}
        
        for index in range(len(dimensions) - 1):
            lower = -1/dimensions[index]
            upper = 1/dimensions[index]

            w = lower + np.random.random([dimensions[index + 1], dimensions[index]]) * (upper - lower)
            b = lower + np.zeros(dimensions[index + 1]) * (upper - lower)
            params["w"][index] = w
            params["b"][index] = b
        
        return params

    def init_params_random(self, dimensions):
        params = {"w" : [[] for i in range(len(dimensions) - 1)], 
        "b": [[] for i in range(len(dimensions) - 1)]}

        for index in range(len(dimensions) - 1):
            w = np.array([random.uniform(-0.5, 0.5) for _ in range(dimensions[index + 1] * dimensions[index])]).reshape((dimensions[index + 1], dimensions[index]))
            #np.random.rand(dimensions[index + 1], dimensions[index])
            b = np.array([random.uniform(-0.5, 0.5) for _ in range(dimensions[index + 1])])
            #np.random.rand(dimensions[index + 1])
            params["w"][index] = w
            params["b"][index] = b
        
        return params

    def sigmoid(self, a):
        b = np.array(a)
        return 1 / (1 + np.exp(-b))
    
    def relu(self, a):
        np.maximum(0,a)
    
    def softmax(self, a):
        b = np.array(a)
        #c = b / np.max(b)
        b = np.exp(b)
        b = b / b.sum()
        return b
        #e = d / d.sum()
    
    def derivativeSigmoid(self, a):
        return (self.sigmoid(a) * ( 1 - self.sigmoid(a)))
  
    def encode(self, y):
        e = np.zeros(len(np.unique(self.train_labels)))
        e[y] = 1
        return e

    def product(self, a , b):
        c = np.zeros(len(a))
        for i in range(len(a)):
            c[i] = a[i] * b[i]
        return c

    def activate(self, a, activation):
        if(activation == "sigmoid"):
            return self.sigmoid(a)
        elif(activation == "ReLU"):
            return self.relu(a)

    def mean_squared_error(self, label, pred):
        x = self.encode(label)
        y = pred
        return (x*x - y*y).mean()

    def calc_accuracy(self,a,b):
        return (len(a) - np.count_nonzero(a - b)) * 100 / len(a)

    def forward_propagation(self, data, outputOriginal, params, dimensions):

            layers = self.init_layers(dimensions)

            layers["h"][0] = data.flatten().astype(np.float32)
            layers["h"][0] /= 255
            for index in range(1, len(dimensions)):
                #reshape**
                layers["a"][index] = np.matmul(params["w"][index - 1], layers["h"][index - 1]) + params["b"][index - 1]
                if(index != len(dimensions) - 1):
                    layers["h"][index] = self.activate(layers["a"][index], self.conf["activation"])
            
            layers["h"][len(dimensions)-1] = self.softmax(layers["a"][len(dimensions) - 1])
            #print(sum(layers["h"][len(dimensions) - 1]))

            if(self.conf["loss"] == "cross_entropy"):
                loss = -1 * math.log(layers["h"][len(dimensions) - 1][outputOriginal])
            elif(self.conf["loss"] == "mean_squared_error"):
                loss = self.mean_squared_error(outputOriginal, layers["h"][len(dimensions) - 1])

            p = np.argmax(layers["h"][len(dimensions) - 1])
            return [layers, loss, p]

    def backward_propagation(self, layers, params, dimensions, y):
        dParam = self.init_params_zeros(dimensions)
        dLayers = self.init_layers(dimensions)
        k = dimensions[len(dimensions) - 1]

        dLayers["a"][len(dimensions) - 1] = -1 * (self.encode(y) - layers["h"][len(dimensions) - 1])
        
        for index in range(len(dimensions)-2, -1, -1):
            #dParam["w"][index] = np.matmul(dLayers["a"][index+1].reshape((dimensions[index + 1]), 1),layers["h"][index].reshape((1, dimensions[index])))
            dParam["w"][index] = np.outer((dLayers["a"][index+1]).T,layers["h"][index])
            dParam["b"][index] = dLayers["a"][index+1]
            if(index != 0):
                dLayers["h"][index] = np.matmul((params["w"][index]).T, dLayers["a"][index + 1])
                dLayers["a"][index] = self.product(dLayers["h"][index],self.derivativeSigmoid(layers["a"][index]))
        return dParam


    def init_layers(self, dimensions):
        layers = {"a": [[] for i in range(len(dimensions))], 
        "h" : [[] for i in range(len(dimensions))]}

        for index in range(len(dimensions)):
            layers["a"][index] = np.zeros(dimensions[index])
            layers["h"][index] = np.zeros(dimensions[index])
        return layers

    def calc_params_momentum(self, params, batchSize, dimensions, eta, beta):

        index = 0
        predictions = []
        lossTrain = 0

        while(index < len(self.test_images)):

            dParam = self.init_params_zeros(dimensions)
            uold = self.init_params_zeros(dimensions)

            for i in range(index, index + batchSize):
               [layers, loss, p] = self.forward_propagation(self.train_images[i], self.train_labels[i], params, dimensions)
               temp = self.backward_propagation(layers, params, dimensions, self.train_labels[i])
               lossTrain += loss
               predictions.append(p)
              

               for l in range(len(dimensions)):
                    dParam["w"][l] += temp["w"][l]
                    dParam["b"][l] += temp["b"][l]
            
            unew = self.init_params_zeros(dimensions)
            for i in range(len(dimensions) - 1):
                unew["w"][i] = beta * uold["w"][i] + dParam["w"][i]
                unew["b"][i] = beta * uold["b"][i] + dParam["b"][i]

            for i in range(len(dParam)):
                params["w"][i] = params["w"][i] - eta * unew["w"][i]
                params["b"][i] = params["b"][i] - eta * unew["b"][i]
            
            uold = unew

        accTrain = self.calc_accuracy(self.train_labels, predictions)
        return [params, accTrain, lossTrain]



    def calc_params_sgd(self, params, batchSize, dimensions, eta):
        
        index = 0
        lossTrain = 0
        predictions = []

        while(index < len(self.train_images)):
            print(index)
            
            dParam = self.init_params_zeros(dimensions)

            for i in range(index, index + batchSize):
                if(i >= len(self.train_images) ):
                    break
                [layers, loss, p] = self.forward_propagation(self.train_images[i], self.train_labels[i], params, dimensions)
                temp = self.backward_propagation(layers, params, dimensions, self.train_labels[i])
                lossTrain += loss
                predictions.append(p)

                for i in range(len(dimensions) - 1):
                    dParam["w"][i] = dParam["w"][i] + temp["w"][i]
                    dParam["b"][i] = dParam["b"][i] + temp["b"][i]
                
            for i in range(len(dimensions) - 1):
                dParam["w"][i] = dParam["w"][i] / batchSize
                dParam["b"][i] = dParam["b"][i] / batchSize
            
            for i in range(len(dimensions) - 1):
                params["w"][i] = params["w"][i] - eta * dParam["w"][i]
                params["b"][i] = params["b"][i] - eta * dParam["b"][i]

            index += batchSize

        accTrain = self.calc_accuracy(self.train_labels, predictions)
        
        return [params, accTrain, lossTrain]

            


    def calc_params(self, params, batchSize, dimensions, eta, beta):
        updateRule = self.conf["optimizer"]
        if(updateRule == "sgd"):
            [params, accTrain, lossTrain] = self.calc_params_sgd(params, batchSize, dimensions, eta)
        if(updateRule == "momentum"):
            [params, accTrain, lossTrain] = self.calc_params_momentum(params, batchSize, dimensions, eta, beta)
        # if(updateRule == "nag"):
        #     [params, accTrain, lossTrain] = self.calc_params_nag(params, batchSize, dimensions)
        # if(updateRule == "rmsprop"):
        #     [params, accTrain, lossTrain] = self.calc_params_rmsprop(params, batchSize, dimensions)
        # if(updateRule == "nadam"):
        #     [params, accTrain, lossTrain] = self.calc_params_nadam(params, batchSize, dimensions)
        
        return [params, accTrain, lossTrain]

    def gradientDescend(self):


        batchSize = int(self.conf["batch_size"])
        epochs = int(self.conf["epochs"])
        eta = float(self.conf["learning_rate"])
        levels = int(self.conf["num_layers"])
        hidden_size = int(self.conf["hidden_size"])

        dimensions = []
        for dim in range(levels):
            dimensions.append(hidden_size)

        beta = float(self.conf["beta"])

        input_dimension = [self.train_images[0].shape[0] * self.train_images[0].shape[1]]
        output_dimension = [len(np.unique(self.train_labels))]

        dimensions = input_dimension + dimensions + output_dimension
        dimensions = np.array(dimensions)

        #preparing network initializing with random parameters
        params = self.init_params_random(dimensions)
        #layers = self.init_layers(self.dimensions)
        
        #acc = 0
        TrainLosses = []
        TrainAccuracy = []

        TestLosses = []

        for itr in range (epochs):
            
            
            [params, accTrain, lossTrain] = self.calc_params(params, batchSize, dimensions, eta, beta)
            
            #wandb.log({"lossTrain": lossTrain})
            TrainLosses.append(lossTrain)
            TrainAccuracy.append(accTrain)
            
            actual = []
            predicted = []
            lossTest = 0
            for index in range(len(self.test_images)):
                [layers, loss, p] = self.forward_propagation(self.test_images[index], self.test_labels[index], params, dimensions)
                lossTest += loss
                predicted.append(p)
                actual.append(self.test_labels[index])
            
            confusion_matrix = metrics.confusion_matrix(actual, predicted)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [i for i in range(0,10)])

            actual = np.array(actual)
            predicted = np.array(predicted)
            accTest = self.calc_accuracy(actual, predicted)
            

            cm_display.plot()
            plt.show()
            #wandb.log({"loss": lossTest})
            TestLosses.append(lossTest)
            print("training loss: " , lossTrain)
            print("test loss: ", lossTest)
            print("test accuracy: ", accTest)
            print("train accuracy: ", TrainAccuracy)






            



# #sc.draw_samples()
argstore = {
    "batch_size" : "100",
    "dataset" : "fashion_mnist",
    "activation" : "sigmoid",
    "loss" : "cross_entropy",
    "optimizer": "sgd",
    "epochs": "100",
    "learning_rate": "0.01",
    "num_layers": "3",
    "hidden_size" : "128",
    "beta" : "0.4"

}
sc = SimpleClassifier(argstore)
sc.gradientDescend()
