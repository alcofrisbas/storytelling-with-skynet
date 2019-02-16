"""
 A start at an evolutionary algorithm to optimize hyperparameters.

 Method of reproduction:
    bitwise crossover.

use code from run.py 134-148 as a fitness function.
"""
import sys, os
import random
from utils import *


class Individual:
    def __init__(self, labels, values, code="", type_markers=[]):
        self.labels = labels
        self.values = values
        self.code = code
        self.type_markers = type_markers
        if self.code and len(self.values) == 0:
            print("decodeing")
            self.decode()
            print("done decoding")
        else:
            self.encode()

    def encode(self):
        for v in self.values:
            if isinstance(v, int):
                self.type_markers.append("int")
                self.code += int_to_bin(v)
            elif isinstance(v, float):
                self.type_markers.append("float")
                self.code += float_to_bin(v)

    def decode(self):
        for i,t in enumerate(self.type_markers):
            start = i*32
            end = start+32
            if end > len(self.code):
                break
            if t == "int":
                self.values.append(bin_to_int(self.code[start: end]))
            elif t == "float":
                self.values.append(bin_to_float(self.code[start: end]))
            else:
                print(t)
        print(self.values)

def mutate(s):
    prob = 1.0/len(s)
    new_s = ""
    for i in range(len(s)):
        if random.random() < prob:
            if s[i] == "0":
                new_s += "1"
            else:
                new_s += "0"
        else:
            new_s += s[i]
    return s

def crossover(a, b, k=1):
    l = len(a.code)
    s1 = s2 = ""
    switch = 0
    rand = 0
    randList = sorted(random.sample(range(0, l), k))

    for i in range(len(randList)):
        if i == 0:
            prev = 0
            rand = randList[i]
        else:
            prev = randList[i-1]
            rand = randList[i]
        if switch%2 == 0:
            s1 += a.code[prev:rand]
            s2 += b.code[prev:rand]
        else:
            s1 += b.code[prev:rand]
            s2 += a.code[prev:rand]
        switch += 1
    prev = int(rand)
    rand = int(l)
    if switch%2 == 0:
        s1 += a.code[prev:rand]
        s2 += b.code[prev:rand]
    else:
        s1 += b.code[prev:rand]
        s2 += a.code[prev:rand]
    s1 = mutate(s1)
    s2 = mutate(s2)
    return Individual(a.labels, [], s1, a.type_markers), Individual(a.labels, [], s2, a.type_markers)



def reproduce(a, b, k_bounds=(0.25,0.5)):
    k = random.uniform(k_bounds[0], k_bounds[1])
    offspring1 = linear_interpolate(a,b,k)
    offspring2 = linear_interpolate(a,b,1.0-k)
    return offspring1, offspring2

def test_fitness(d):
    d_new = {}
    return d_new

def load_individuals(fname):
    pass

def save_individuals(fname,d):
    return

if __name__ == '__main__':
    labels = ["init_scale","learning_rate", "max_grad_norm", "num_layers",
            "num_steps", "hidden_size", "max_epoch", "max_max_epoch", "keep_prob",
            "lr_decay", "batch_size", "vocab_size"]

    values1 = [0.1, 1.0, 5, 2, 20, 200, 4, 20, 1.0, 0.5, 1, 10011]
    values2 = [0.05, 1.0, 5, 2, 35, 650, 6, 39, 0.8, 0.8, 1, 10011]
    i1 = Individual(labels, values2)
    i2 = Individual(labels, values1)

    i3, i4 = crossover(i1,i2, k=250)
    for i in range(len(i3.labels)):
        print("{}:\t{}".format( i3.labels[i], str(i3.values[i])))
