import copy
from Rule import Rule

class RuleSet():

    def __init__(self, filename = None):
        self.rules = []
        if(filename != None):
            self.initFromFile(filename)

    def copy(self):
        rs = RuleSet()
        for i in range(len(self.rules)):
            rs.rules.append(self.rules[i].copy())
        return rs
    
    def initFromFile(self, filename):
        file = open(filename)
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            line = line.strip()
            if(line != ""):
                self.add(line)
        file.close()
        
    def findRule(self, base):
        for i in range(len(self.rules)):
            if(self.rules[i].base == base):
                return self.rules[i]
        
    def add(self, full_sequence):
        full_sequence = full_sequence.split(" := ")
        start = full_sequence[0]
        end = full_sequence[1]
        
        for i in range(len(self.rules)):
            if(self.rules[i].base == start):
                self.rules[i].add(end)
                return;
                
        rule = Rule(start)
        rule.add(end)
        self.rules.append(rule)
            
    def to_string(self):
        build = "RULE SET\n--------\n"
        for i in range(len(self.rules)):
            build += str(self.rules[i]) + "\n"
        return build + "--------"
        
    def __str__(self):
        return self.to_string()
    def __unicode__(self):
        return self.to_string()
    def __repr__(self):
        return self.to_string()
    
    def chooseOption(self, rule, codon):
        num_options = len(rule.outputs)
        codon = codon % num_options
        output = rule.outputs[codon]
        return output
        
    def buildAnOutput(self, rule):
        self.recursion_guard += 1
        if(self.recursion_guard == 2000):
            return False
            
        self.codon_pos += 1
        if(self.codon_pos == len(self.codons)):
            self.codon_pos = 0
        
        output = self.chooseOption(rule, self.codons[self.codon_pos])
        
        
        result = []
        for i in range(len(output)):
            cur = output[i]
            
            if(cur[0].isupper()):
                
                rule = self.findRule(cur)
                new = self.buildAnOutput(rule)
                
                if(new == False):
                    return False
                
                new = " ".join(new)
            else:
                new = cur
                
            result.append(new)
        
        return result
                
        
    def buildSentence(self, codons):
        self.recursion_guard = 0
        self.codons = codons
        self.codon_pos = -1
        result = self.buildAnOutput(self.rules[0])
        if(result == False):
            return "'ERR - RECURSION DEPTH REACHED'"
        else:
            return " ".join(result)
            


    def constructEquation(self, chromosome):
        codons = []
        for codon in chromosome:
            dec = int(codon, 2)
            codons.append(dec)
        
        this_sentence = self.buildSentence(codons)
        return this_sentence
                
        
    def buildANodePrint(self, rule):
            
        self.recursion_guard += 1
        if(self.recursion_guard == 2000):
            return False
            
        self.codon_pos += 1
        if(self.codon_pos == len(self.codons)):
            self.codon_pos = 0
        
        output = self.chooseOption(rule, self.codons[self.codon_pos])
        
        result = []
        for i in range(len(output)):
            cur = output[i]
            
            if(cur[0].isupper()):

                rule = self.findRule(cur)
                new = self.buildANodePrint(rule)
                
                if(new == False):
                    return False
                
                new = " ".join(new)
            else:
                new = cur
                
            result.append(new)
        
        
        self.global_node_prints.append(" ".join(result))
        
        return result            
                
                
        
    def buildNodePrints(self, codons):
        self.global_node_prints = []
        
        self.recursion_guard = 0
        self.codons = codons
        self.codon_pos = -1
        result = self.buildANodePrint(self.rules[0])
        if(result == False):
            return "'ERR - RECURSION DEPTH REACHED'"
        else:
            node_prints_obj = {}
            for node_print in self.global_node_prints:
                if node_print in node_prints_obj:
                    node_prints_obj[node_print] = node_prints_obj[node_print] + 1
                else:
                    node_prints_obj[node_print] = 1
            return node_prints_obj
    
    
    def produceNodePrints(self, chromosome):
        codons = []
        for codon in chromosome:
            dec = int(codon, 2)
            codons.append(dec)
        
        these_node_prints = self.buildNodePrints(codons)
        return these_node_prints
    