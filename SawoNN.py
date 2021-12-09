import tensorflow as tf
import numpy as np
import json
import copy
import re
from ast import literal_eval
import math

from Node import Node
from Architecture import Architecture

class SawoNN:

    def __init__(self, equation, X_train, y_train, X_test, y_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        equation = equation.split(" $ ")

        equation[0] = json.loads(equation[0])
        layersizes = equation[0]
        equation[1] = equation[1].split(" ~ ")
        wues = equation[1]
        equation[2] = literal_eval(equation[2])
        primary = equation[2]

        orig_weights = ["io", "ia", "ib", "ic", "ab", "ba", "ac", "ca", "bc", "cb", "ao", "bo", "co"]


        #print("STEP 1")
        nodes = [Node(x) for x in orig_weights]


        #print("STEP 2")
        for i in range(len(orig_weights)):
            node_xy = nodes[i]
            weight_xy = orig_weights[i]
            wue_xy = wues[i]

            alpha_xy = self.getArchsFromWUE(wue_xy)
            for d in alpha_xy:
                weights_in_d = self.getWeightsInArch(d)
                for weight_ij in weights_in_d:
                    node_ij = nodes[orig_weights.index(weight_ij)]
                    node_xy.addEdge(node_ij)


        #print("STEP 3")
        needed_weights = []


        #print("STEP 4")
        g = primary
        weights_in_g = self.getWeightsInArch(g)
        self.num_weights_in_primary = len(weights_in_g)
        visited = [False for x in orig_weights]
        for weight_xy in weights_in_g:
            node_xy = nodes[orig_weights.index(weight_xy)]

            nodes_con, visited = node_xy.nodesConNotInc(orig_weights, visited)

            for n in nodes_con:
                if n.label not in needed_weights:
                    needed_weights.append(n.label)

            needed_weights = sorted(needed_weights)


        #print("STEP 5")
        needed_archs = [primary]


        #print("STEP 6")
        for weight_xy in needed_weights:
            wue_xy = wues[orig_weights.index(weight_xy)]
            alpha_xy = self.getArchsFromWUE(wue_xy)

            for d in alpha_xy:
                if d not in needed_archs:
                      needed_archs.append(d)
                      

        layersizes["i"] = 10 if self.X_train is None else self.X_train.shape[1]
        layersizes["o"] = 1 if self.y_train is None else self.y_train.shape[1]

        
        self.layersizes = layersizes
        self.wues = wues
        self.primary = primary
        self.needed_weights = needed_weights
        self.needed_archs = needed_archs
        self.orig_weights = orig_weights
        
    
    def printEssentialInfo(self):
        build = "(V) "
        if len(self.needed_weights) == self.num_weights_in_primary:
            build = "(N) "
        build += str(self.layersizes) + " $ "
        for weight_xy in self.needed_weights:
            wue_xy = self.wues[self.orig_weights.index(weight_xy)]
            build += wue_xy + " ~ "
        build = build[:-3] + " $ " + str(self.primary)
        return build

    
    def getArchsFromWUE(self, wue):
        archs = [x.group() for x in re.finditer(r' \[[\' ,\w]+\] ', wue)]
        archs = [x.strip() for x in archs]
        archs = [literal_eval(x) for x in archs]
        new_archs = []
        for arch in archs:
            if arch not in new_archs:
                new_archs.append(arch)
        return new_archs

    def weightInArch(self, weight, arch):
        for i in range(len(arch)-1):
            if weight[0] in arch[i] and weight[1] in arch[i+1]:
                return True
        return False

    def getWeightsInArch(self, arch):
        weights = []
        for i in range(len(arch)-1):
            front = arch[i]
            end = arch[i+1]
            for j in range(len(front)):
                for k in range(len(end)):
                    if end[k] != "i":
                        weights.append(str(front[j]) + "" + str(end[k]))
        return weights

    
    def constant(self, value, weight_for):
        return [np.full_like(self.weights[weight_for][0], value), np.full_like(self.weights[weight_for][1], value)]
    
    def add(self, lhs, rhs):
        result = None
        lhs[0] = np.add(lhs[0], rhs[0])
        lhs[1] = np.add(lhs[1], rhs[1])
        result = lhs
        return result
    
    def subtract(self, lhs, rhs):
        result = None
        lhs[0] = np.subtract(lhs[0], rhs[0])
        lhs[1] = np.subtract(lhs[1], rhs[1])
        result = lhs
        return result
      
    def multiply(self, lhs, rhs):
        result = None
        lhs[0] = np.multiply(lhs[0], rhs[0])
        lhs[1] = np.multiply(lhs[1], rhs[1])
        result = lhs
        return result
      
    def divide(self, lhs, rhs):
        result = None
        lhs[0] = np.divide(lhs[0], rhs[0])
        lhs[1] = np.divide(lhs[1], rhs[1])
        result = lhs
        return result
      
    def sqrt(self, exp):
        result = None
        exp[0] = np.sqrt(exp[0])
        exp[1] = np.sqrt(exp[1])
        return result
      
    def abs(self, exp):
        result = None
        exp[0] = np.abs(exp[0])
        exp[1] = np.abs(exp[1])
        return result
      
    def exp(self, e):
        result = None
        e[0] = np.exp(e[0])
        e[1] = np.exp(e[1])
        return result
      
    def log(self, exp):
        result = None
        exp[0] = np.log(exp[0])
        exp[1] = np.log(exp[1])
        return result
                
    
    def train(self, epochs, num_batches, class_T_reg_F):
    
        if class_T_reg_F:
            reporting_metric = 'acc'
        else:
            reporting_metric = 'mse'
    
        batch_size = self.X_train.shape[0] / num_batches


        #print("STEP 7")
        weights = {}       

        for weight_label in self.needed_weights:

            x_size = self.layersizes[weight_label[0]]
            y_size = self.layersizes[weight_label[1]]

            normal_weights_xy = np.random.normal(size=(x_size, y_size), loc=0.0, scale=0.2)
            bias_xy = np.zeros(y_size)
            weight_xy = [normal_weights_xy, bias_xy]

            weights[weight_label] = weight_xy 
        
        self.weights = weights
        
        
        #print("STEP 1")
        self.constructed_archs = [Architecture(d, self.weights, self.layersizes) for d in self.needed_archs]


        #print("STEP 2")
        perfs = []
        primary_perf = self.constructed_archs[0].error(reporting_metric, self.X_test, self.y_test)
        print("primary_perf", primary_perf)
        
        perfs.append(primary_perf)
        
        best_perf = primary_perf
        best_weights = copy.deepcopy(self.weights)
        
        last_perf = primary_perf.numpy()
        num_since_last_best = 0


        #print("STEP 3")
        for epoch in range(epochs):
            print("EPOCH ", epoch)
            
            randomize = np.arange(len(self.X_train))
            np.random.shuffle(randomize)
            self.X_train = self.X_train[randomize]
            self.y_train = self.y_train[randomize]
            
            for batch in range(num_batches):
                print("BATCH ", batch)
                
                start_batch = math.ceil( batch_size * batch )
                end_batch = math.ceil( batch_size * (batch+1) )
                self.X_train_batch = self.X_train[start_batch:end_batch]
                self.y_train_batch = self.y_train[start_batch:end_batch]
                
                for weight_xy in self.needed_weights:
                    wue_xy = self.wues[self.orig_weights.index(weight_xy)]
                    result = eval(wue_xy)
                    self.weights[weight_xy] = result
                
                self.constructed_archs = [Architecture(d, self.weights, self.layersizes) for d in self.needed_archs]

                primary_perf = self.constructed_archs[0].error(reporting_metric, self.X_test, self.y_test)
                print("primary_perf (", primary_perf.numpy(), ") best_perf", best_perf.numpy(), "num_since_last_best", num_since_last_best)
                perfs.append(primary_perf)
                
                if np.isnan(primary_perf.numpy()):
                    return self.primary, [x.numpy() for x in perfs], best_weights

                if class_T_reg_F:
                    if primary_perf > best_perf:
                        best_perf = primary_perf
                        best_weights = copy.deepcopy(self.weights)
                        num_since_last_best = 0
                    else:
                        num_since_last_best += 1
                else:
                    if primary_perf < best_perf:
                        best_perf = primary_perf
                        best_weights = copy.deepcopy(self.weights)
                        num_since_last_best = 0
                    else:
                        num_since_last_best += 1
                    
                if num_since_last_best == int(math.ceil(num_batches*int(epochs/1))):
                    return self.primary, [x.numpy() for x in perfs], best_weights
                
                last_perf = primary_perf.numpy()
        
        
        #print("STEP 4")
        return self.primary, [x.numpy() for x in perfs], best_weights




    
    def d(self, error_type, arch, deriv):

        index = self.needed_archs.index(arch)
        constructed_arch = self.constructed_archs[index]

        with tf.GradientTape() as tape:
            error = constructed_arch.error(error_type, self.X_train_batch, self.y_train_batch)

        names = [(constructed_arch.model.layers[i].name.replace("2", ""), i) for i in range(len(constructed_arch.model.layers)) if constructed_arch.model.layers[i].name.endswith(deriv[1])]
        layer_num = None
        if len(names) == 1:
            layer_num = names[0][1]
        else:
            names = [x for x in names if x[0] == deriv]
            layer_num = names[0][1]

        grads = tape.gradient(error, constructed_arch.model.layers[layer_num].trainable_weights)

        return grads
        
    
    def gu(self, weight, const, error_type, arch):
        return self.subtract ( self.weights[weight] , self.multiply ( self.constant( const , weight) , self.d( error_type , arch , weight ) ) )
            


