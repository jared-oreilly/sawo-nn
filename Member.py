import random
import copy
import json
import re

from SawoNN import SawoNN

class Member():

    def __init__(self, chromosome = None):
        if(chromosome == None):
            self.chromosome = self.randomChromosome()
        else:
            self.chromosome = copy.deepcopy(chromosome)

    def copy(self):
        mem = Member(self.chromosome)
        return mem

    def randomChromosome(self):
        num_codons = 31
        build = []
        for i in range(num_codons):
            build.append("")
            for j in range(7):
                rand_char = int(round(random.random()))
                build[i] += str(rand_char)
        return build
            
    def to_string(self):
        return str(self.chromosome)
    def __str__(self):
        return self.to_string()
    def __unicode__(self):
        return self.to_string()
    def __repr__(self):
        return self.to_string()
        
    def construct(self, rs):
        codons = []
        for codon in self.chromosome:
            dec = int(codon, 2)
            codons.append(dec)
        
        this_sentence = rs.buildSentence(codons)
        return this_sentence
    
    def fitness(self, rs, X_train, X_test, y_train, y_test):
        from tensorflow import keras
        
        equation = self.construct(rs)
        print("equation", equation)
        if("'ERR" in equation):
            return float('inf')
        
        gnn = SawoNN(equation, X_train, y_train, X_test, y_test)
        
        num_epochs = 20
        num_batches = 5
        num_to_avg = 2
        last_min_loss = None
        sum_fit = 0
        for j in range(num_to_avg):
            primary, errors, weights = gnn.train(num_epochs, num_batches, class_T_reg_F=False)
            #print(primary)
            print(errors)
            #print(weights)

            first_loss = errors[0]
            min_loss = errors[0]
            count_min = 0
            for k in range(1, len(errors)):
                if errors[k] < min_loss:
                    min_loss = errors[k]
                    count_min += 1
            
            print("first_loss", first_loss)
            print("min_loss", min_loss)
            print("count_min", count_min)
            
            if j == 0:
                last_min_loss = min_loss
            else:
                if last_min_loss == min_loss:
                    print("(single) avg_fit:", sum_fit)
                    return sum_fit
                last_min_loss = min_loss
                
            top_for_count = int(num_epochs*num_batches)
            count_min = (top_for_count - count_min) / (top_for_count - 0)
            improvement = (min_loss / first_loss)
            optimality = min_loss
            
            print("c, i, o:", count_min, improvement, optimality)
            
            fit = (count_min*0.0) + (improvement*0.0) + (optimality*1.0)
            print("fit:", fit)
            
            sum_fit += fit


        print("avg_fit:", sum_fit / num_to_avg)
        return (sum_fit / num_to_avg)