import numpy as np
from random import random
import copy
from math import e

class NeuralNetwork:

    def __init__(self, structure, learning_rate, iterations):
        self.layers = [int(x) for x in structure.split('-')]
        
        self.iterations = iterations
        self.weights = [[[random() for k in range(self.layers[i] + 1)] for j in range(self.layers[i + 1])] for i in range(len(self.layers) - 1)]
        self.lr = learning_rate

        self.activation_function = lambda x: self.__activate(x)

    def __activate(self, x):
        return 1/(1+e**(-x))
    
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        for iteration in range(self.iterations):
            graph_outputs = []
            hidden_outputs = inputs[:]
            for i in range(len(self.weights)):
                hidden_outputs = np.insert(hidden_outputs, 0, 1)
                hidden_outputs = np.array(hidden_outputs, ndmin=2).T
                hidden_inputs = np.dot(self.weights[i], hidden_outputs)
                hidden_outputs = self.activation_function(hidden_inputs)

                graph_outputs.append(hidden_outputs.tolist())
                if i == len(self.weights) - 2:
                    graph_outputs[i] = np.insert(graph_outputs[i], 0, 1).tolist()

            inputs_for_last_step = inputs[:].tolist()
            inputs_for_last_step.insert(0, [1])
            inputs_for_last_step = np.array(inputs_for_last_step, ndmin=2)

            weights_corrected = copy.deepcopy(self.weights)
            weights_for_hidden_errors = copy.deepcopy(self.weights)
            for i in range(len(weights_for_hidden_errors)):
                for j in range(len(weights_for_hidden_errors[i])):
                   weights_for_hidden_errors[i][j] = weights_for_hidden_errors[i][j][1:]
            
            delta_wy = self.lr * (targets - hidden_outputs)
            delta_s = delta_wy * hidden_outputs * (1.0 - hidden_outputs)
            
            for i in range(len(self.weights) - 1, -1, -1):
                if i == len(self.weights) - 1:
                    delta_w = delta_s * graph_outputs[i - 1]
                    weights_corrected[i] += delta_w
                elif i == 0:
                    output = copy.deepcopy(graph_outputs[i])
                    output = output[output != 1.0]
                    output = np.array(output)
                    
                    if len(self.layers)> 3:
                        delta_wy = np.dot(delta_s, weights_for_hidden_errors[i+1])
                        
                        delta_s = delta_wy * output * (1.0 - output)
                        delta_s = np.array(delta_s, ndmin=2).T

                        delta_w = np.dot(delta_s, inputs_for_last_step.T)
                        weights_corrected[i] += delta_w
                    else:
                        delta_wy = delta_s * weights_for_hidden_errors[i + 1]
                        graph_outputs[i].pop(0)

                        graph_outputs[i] = np.array(graph_outputs[i])
                        result = np.sum(delta_wy, axis=0)

                        delta_s = (result * graph_outputs[i] * (1.0 - graph_outputs[i])).tolist()
                        delta_s = np.array(delta_s, ndmin=2)

                        delta_w = np.dot(delta_s.T, inputs_for_last_step.T)
                        weights_corrected[i] += delta_w

                else:
                    delta_wy = delta_s * weights_for_hidden_errors[i + 1]
                    graph_outputs[i].pop(0)
                    
                    graph_outputs[i] = np.array(graph_outputs[i])
                    result = np.sum(delta_wy, axis=0)
                    
                    delta_s = (result * graph_outputs[i] * (1.0 - graph_outputs[i])).tolist()
                    delta_s = np.array(delta_s, ndmin=2)
                    
                    graph_outputs[i - 1] = np.insert(graph_outputs[i - 1], 0, 1).tolist()
                    graph_outputs[i - 1] = np.array(graph_outputs[i - 1], ndmin=2)
                    
                    delta_w = np.dot(delta_s.T, graph_outputs[i - 1])
                    weights_corrected[i] += delta_w
            
            for i, j  in zip(weights_corrected, range(len(self.weights))):
                i = i.tolist()
                self.weights[j] = i
                
            
    def query(self, inputs_list, weights=None):
        if weights == None:
            weights = self.weights
            
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_outputs = inputs[:]
        for i in range(len(weights)):
            
            hidden_outputs = np.insert(hidden_outputs, 0, 1)
            hidden_outputs = np.array(hidden_outputs, ndmin=2).T
            
            hidden_inputs = np.dot(weights[i], hidden_outputs)
        
            hidden_outputs = self.activation_function(hidden_inputs)
        
        return hidden_outputs







