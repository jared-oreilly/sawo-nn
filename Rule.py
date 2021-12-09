import copy

class Rule():

    def __init__(self, base):
        self.base = base
        self.outputs = []

    def copy(self):
        rule = Rule(self.base)
        rule.outputs = copy.deepcopy(self.outputs)
        return rule

    def add(self, sequence):
        sequence = sequence.split(" ")
        self.outputs.append(sequence)
            
    def to_string(self):
        build = self.base + " := "
        for i in range(len(self.outputs)):
            ele = " ".join(self.outputs[i])
            build += ele + " | "
        return build[:-3]
        
    def __str__(self):
        return self.to_string()
    def __unicode__(self):
        return self.to_string()
    def __repr__(self):
        return self.to_string()