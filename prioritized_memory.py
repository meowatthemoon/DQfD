import random
import numpy as np
import torch
import torch.nn as nn


class Memory:  # stored as ( s, a, r, s_ )
    e_a = 0.01
    e_d = 0.6
    beta = 0.0
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, epoch=150):
        # self.tree = SumTree(capacity)
        # self.capacity = capacity
        self.transitions = []
        self.td_errors = []
        self.n_demonstrations = 0

    def add(self, transition, demonstration):
        """
        if error is None:
            p = self.tree.tree[0]  # max priority for new data
            if p == 0:
                p = 0.1
            else:
                p = self.tree.get(p * 0.9)[1]
        else:
            p = self._get_priority(error)
        self.tree.add(p, sample)
        """
        if demonstration:
            self.n_demonstrations += 1
        self.transitions.append(transition)
        self.td_errors.append(0)

    def sample(self, batch_size):
        """
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight
        """
        abs = np.absolute(self.td_errors)
        priorities = []
        for index in range(len(self.transitions)):
            if index <= self.n_demonstrations:
                priority = abs[index] + self.e_d
            else:
                priority = abs[index] + self.e_a
            priorities.append(priority)

        priorities = np.array(priorities)
        probabilities = priorities / np.sum(priorities)
        idxs = np.random.choice(len(probabilities), batch_size, p=probabilities, replace=False)
        batch = np.array(self.transitions)[idxs]
        return batch, idxs

    def update(self, idx, error):
        # p = self._get_priority(error)
        # self.tree.update(idx, p)
        self.td_errors[idx] = error
