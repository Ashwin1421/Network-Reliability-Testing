from matplotlib import pyplot as plt, pylab
from math import ceil, pow
from operator import add
import numpy as np
from random import shuffle, sample
from itertools import product, accumulate
import networkx as nx
from collections import defaultdict
from sys import argv


class Main:
    def __init__(self, N, p, k):
        self.N = int(N)
        self.p = float(p)
        self.k = int(k)
        self.ID = ''
        self.d = list(map(int, self.ID))

        self.graph = nx.complete_graph(self.N)
        self.edges = self.graph.edges(nbunch=None, data=True)
        """
        Shuffling the list ensures that the edges are always indexed randomly.
        """
        shuffle(self.edges)
        for index, edge in enumerate(self.edges):
            edge[2]['reliability'] = pow(self.p, ceil(self.d[index] / 3))

        self.sys_cond = defaultdict(list)

        self.sys_states = None

    def genSysCombo(self):
        lst = [list(tup) for tup in product([0, 1], repeat=len(self.edges))]

        for tup in lst:
            if tup.count(1) == 1:
                self.sys_cond['1'].append(tup[::-1])
            if tup.count(1) == 2:
                self.sys_cond['2'].append(tup[::-1])
            if tup.count(1) == 3:
                self.sys_cond['3'].append(tup[::-1])
            if tup.count(1) == 4:
                self.sys_cond['4'].append(tup[::-1])
            if tup.count(1) == 5:
                self.sys_cond['5'].append(tup[::-1])
            if tup.count(1) == 6:
                self.sys_cond['6'].append(tup[::-1])
            if tup.count(1) == 7:
                self.sys_cond['7'].append(tup[::-1])
            if tup.count(1) == 8:
                self.sys_cond['8'].append(tup[::-1])
            if tup.count(1) == 9:
                self.sys_cond['9'].append(tup[::-1])
            if tup.count(1) == 10:
                self.sys_cond['10'].append(tup[::-1])

    def set_sys_states(self):
        min_edges = nx.edge_connectivity(self.graph)
        self.sys_states = ['up'] * min_edges
        self.sys_states += ['down'] * (len(self.edges) - min_edges)

    def get_total_rel(self):
        total_rel = defaultdict(list)
        for i, state in enumerate(self.sys_states):
            if state == 'up':
                for t in self.sys_cond[str(i + 1)]:
                    for s in t:
                        if s == 0:
                            total_rel[str(i + 1)].append(self.edges[t.index(s)][2]['reliability'])
                        elif s == 1:
                            total_rel[str(i + 1)].append(1 - self.edges[t.index(s)][2]['reliability'])

        total_sum = 0
        for i, s in enumerate(self.sys_states):
            if s == 'up':
                total_sum += sum(total_rel[str(i + 1)]) / len(total_rel[str(i + 1)])

        return total_sum / self.sys_states.count('up')

    def list_complement(self, array):
        for i in range(len(array)):
            if array[i] == 1:
                array[i] = 0
            else:
                array[i] = 1

        return array

    def kOutOfN(self, k):

        flat_list = [item for sublist in list(self.sys_cond.values()) for item in sublist]
        random_combo = sample(flat_list, k)
        """
        Changing the states of randomly selected configuration to 'up' or 'down'.
        """
        for i, val in enumerate(random_combo):
            random_combo[i] = self.list_complement(val)

        index_list = []
        for i in range(len(self.sys_cond.values())):
            index_list.append(len(self.sys_cond[str(i + 1)]))

        index_list = list(accumulate(index_list, add))

        total_rel = defaultdict(list)
        for i, val in enumerate(random_combo):
            if val.count(1) < nx.edge_connectivity(self.graph):
                for key, index in enumerate(index_list):
                    if flat_list.index(val) < index:
                        if self.sys_states[key] == 'up':
                            self.sys_states[key] = 'down'
                        else:
                            self.sys_states[key] = 'up'

        rel = self.get_total_rel()

        return rel

    def run(self):
        self.genSysCombo()
        self.set_sys_states()
        if self.k > 0:
            return self.kOutOfN(self.k)
        else:
            return self.get_total_rel()


if __name__ == '__main__':
    if len(argv) == 3:
        N = argv[1]
        p = argv[2]
        main_program = Main(N, p, 0)
        main_program.run()
    elif len(argv) == 6 and argv[1] == '-k':
        N = argv[2]
        p = argv[3]
        k1 = argv[4]
        k2 = argv[5]
        data = []
        steps = list(range(int(k1), int(k2)))
        for step in steps:
            data.append(Main(N, p, step).run())

        plt.plot(steps, data)
        plt.xlabel('k over [0,1,2...20]')
        plt.ylabel('Network Reliability')
        plt.axis()
        plt.ticklabel_format(style='plain', axis='x', useOffset=False)
        plt.savefig('k_out_of_N.png')

    elif len(argv) == 5 and argv[1] == '-i':
        N = argv[2]
        p1 = argv[3]
        p2 = argv[4]
        steps = np.arange(float(p1), float(p2), float(p1))
        data = []
        for step in steps:
            data.append(Main(N, step, 0).run())

        plt.plot(steps, data)
        plt.xlabel('p run over interval [' + p1 + ',' + p2 + ']')
        plt.ylabel('Network Reliability')
        plt.axis([min(steps), max(steps), min(data), max(data)])
        plt.savefig('p_with_intervals.png')


    else:
        print('Usage :\n Main.py N p.')
        print('Main.py -i N p1 p2.')
        print('Use intervals for a range [p1,p2] with steps of p1.')
        print('Main.py -k N p k1 k2.')
        print('Use fixed p value with k values in range [k1,k2].')
