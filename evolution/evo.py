"""
 A start at an evolutionary algorithm to optimize hyperparameters.

 Method of reproduction:
    bitwise crossover.

Questions as of 19 February:
    - how to evaluate fitness...
        * obviously, we use an amalgamation of loss and accuracy,
          but should I include an evaluate_individual function, or
          let the user define a way to best evaluate an individual...

"""
import sys, os
import random
import json
from utils import *


class Individual:
    """
    An Individual is a member of a population. It contains genetic information
    that will affect the outcome of our problem: the loss after one epoch of
    training of our RNN.

    :: labels: a list of attribute names
    :: values: a list of attribute values
    :: code: a binary representation of these values(32bit ints and floats)
        this code is by default empty. If it is not, then init will
        automatically convert it into values.
    :: type_markers: a list of attribute types(either "float" or "int") used by
        encode and decode.
    :: exempt: a dictionary of labeled values that are passed down regardless of
        performance. Used primarily for vocab_size in our case.

    Two Individuals can reproduce, which involves both k-random recombination
    and 1.0/l mutation. By encoding our values into binary, we provide a greater
    diversity and potential for mutation. Each pair produces a pair of children,
    both of which, combnined together contain ALL of both parents' genetic
    material.

    The other method that we considered is simply interpolating, or weigthing
    an average between parents for each attribute, based on a random
    distribution, but that only provides piece-wise variance, as opposed to bit-
    wise variance.
    """
    def __init__(self, labels, values, code="", type_markers=[], exempt={}):
        self.labels = labels
        self.values = values
        self.code = code
        self.type_markers = type_markers
        if self.code and len(self.values) == 0:
            self.decode()
        else:
            self.encode()
        self.exempt = exempt

    def __str__(self):
        s = "== inherited traits ==\n"
        for l in range(len(self.labels)):
            s += "{}\t{:.5}\n".format(self.labels[l], str(self.values[l]))
        s += "== exempt traits ==\n"
        for e in self.exempt:
            s += ("{}\t{:.5}\n".format(e, str(self.exempt[e])))
        s += "== code ==\n"
        s += self.code
        return s

    def encode(self):
        """
        Convert the list of values into bits and a list of types for decoding
        """
        for v in self.values:
            if isinstance(v, int):
                self.type_markers.append("int")
                self.code += int_to_bin(v)
            elif isinstance(v, float):
                self.type_markers.append("float")
                self.code += float_to_bin(v)

    def decode(self):
        """
        convert a string of bits into a list of values based on type markers
        """
        for i,t in enumerate(self.type_markers):
            start = i*32
            end = start+32
            if end > len(self.code):
                break
            if t == "int":
                self.values.append(bin_to_int(self.code[start: end]))
            elif t == "float":
                self.values.append(bin_to_float(self.code[start: end]))

    def attributes(self):
        """
        return a dictionary of attributes, including exempt attributes
        """
        d = {}
        for i in range(len(self.values)):
            d[self.labels[i]] = self.values[i]
        d.update(self.exempt)
        return d


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


def reproduce(a, b, k=1):
    """
    first recombination, then mutation
    """
    l = len(a.code)
    s1 = s2 = ""
    switch = 0
    rand = 0
    randList = random_list_int(0, l, k)

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
    return Individual(a.labels, [], s1, a.type_markers, exempt=a.exempt),Individual(a.labels, [], s2, a.type_markers, exempt=b.exempt)

def population_from_file(fname):
    with open(fname+".traits", 'r') as f:
        indivs = f.read().strip().split('\n')
        labels = indivs.pop(0).split(",")
    population = []
    for i in indivs:
        values = [num(j) for j in i.split(",")]
        population.append(Individual(labels,values))
    with open(fname+".exempt", 'r') as f:
        exempts = f.read().strip().split('\n')
    for i in range(len(exempts)):
        population[i].exempt = json.loads(exempts[i])
    return population

def file_from_population(fname, population):
    with open(fname+".traits",'w') as w:
        w.write(",".join(population[0].labels)+"\n")
        for p in population:
            w.write(",".join([str(i) for i in p.values]))
            w.write("\n")
    with open(fname+".exempt", 'w') as w:
        for p in population:
            w.write(json.dumps(p.exempt))
            w.write("\n")


if __name__ == '__main__':
    pass
