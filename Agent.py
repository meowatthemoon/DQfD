from prioritized_memory import Memory
from Policy import Policy
import torch
import math
import random
from torch import optim
from collections import defaultdict as ddict
from functools import reduce
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Agent:  # todo change name to Agent
    def __init__(self, eps, lr, gamma, batch_size, tau, max_memory, lambda_1, lambda_2, lambda_3, n_steps,
                 l_margin):
        # Input Parameters
        self.eps = eps  # eps-greedy
        self.gamma = gamma  # discount factor
        self.batch_size = batch_size
        self.tau = tau  # frequency of target replacement
        self.ed = 0.005  # bonus for demonstration # todo they aren't used
        self.ea = 0.001  # todo they aren't used
        self.l_margin = l_margin
        self.n_steps = n_steps
        self.lambda1 = lambda_1  # n-step return
        self.lambda2 = lambda_2  # supervised loss
        self.lambda3 = lambda_3  # L2

        self.counter = 0  # target replacement counter # todo change to iter_counter
        self.replay = Memory(capacity=max_memory)
        self.loss = nn.MSELoss()
        self.policy = Policy()  # todo change not have to pass architecture
        self.opt = optim.Adam(self.policy.predictNet.parameters(), lr=lr, weight_decay=lambda_3)

        self.replay.e = 0
        self.demoReplay = ddict(list)

        self.noisy = hasattr(self.policy.predictNet, "sample")

    def choose_action(self, state):
        state = torch.Tensor(state)
        A = self.policy.sortedA(state)

        if self.noisy:
            self.policy.predictNet.sample()
            return A[0]

        if np.random.random() < self.eps:
            return random.sample(A, 1)[0]
        return A[0]

    def sample(self):
        return self.replay.sample(self.batch_size)

    def store_demonstration(self, s, a, r, s_, done, episode):
        s = torch.Tensor(s)
        s_ = torch.Tensor(s_)
        episodeReplay = self.demoReplay[episode]  # replay of certain demo episode
        index = len(episodeReplay)
        data = (s, a, r, s_, done, (episode, index))
        episodeReplay.append(data)
        self.replay.add(transition=data, demonstration=True)

    def store_transition(self, s, a, r, s_, done):
        s = torch.Tensor(s)
        s_ = torch.Tensor(s_)
        data = (s, a, r, s_, done, None)
        self.replay.add(transition=data, demonstration=False)

    def calculate_td_errors(self, samples):
        if self.noisy:
            self.policy.predictNet.sample()  # for choosing action
        alls, alla, allr, alls_, alldone, *_ = zip(*samples)
        maxA = [self.policy.sortedA(s_)[0] for s_ in alls_]
        if self.noisy:
            self.policy.predictNet.sample()  # for prediction
            self.policy.targetNet.sample()  # for target

        Qtarget = torch.Tensor(allr)
        Qtarget[torch.tensor(alldone) != 1] += self.gamma * self.policy.calcQ(self.policy.targetNet, alls_, maxA)[
            torch.tensor(alldone) != 1]
        Qpredict = self.policy.calcQ(self.policy.predictNet, alls, alla)
        return Qpredict, Qtarget

    def JE(self, samples):
        loss = torch.tensor(0.0)
        count = 0  # number of demo
        for s, aE, *_, isdemo in samples:
            if isdemo is None:
                continue
            A = self.policy.sortedA(s)
            if len(A) == 1:
                continue
            QE = self.policy.calcQ(self.policy.predictNet, s, aE)
            A1, A2 = np.array(A)[:2]  # action with largest and second largest Q
            maxA = A2 if (A1 == aE).all() else A1
            Q = self.policy.calcQ(self.policy.predictNet, s, maxA)
            if (Q + self.l_margin) < QE:
                continue
            else:
                loss += (Q - QE)
                count += 1
        return loss / count if count != 0 else loss

    def Jn(self, samples, Qpredict):
        # wait for refactoring, can't use with noisy layer
        loss = torch.tensor(0.0)
        count = 0
        for i, (s, a, r, s_, done, isdemo) in enumerate(samples):
            if isdemo is None:
                continue
            episode, idx = isdemo
            nidx = idx + self.n_steps
            lepoch = len(self.demoReplay[episode])
            if nidx > lepoch:
                continue
            count += 1
            ns, na, nr, ns_, ndone, _ = zip(*self.demoReplay[episode][idx:nidx])
            ns, na, ns_, ndone = ns[-1], na[-1], ns_[-1], ndone[-1]
            discountedR = reduce(lambda x, y: (x[0] + self.gamma ** x[1] * y, x[1] + 1), nr, (0, 0))[0]
            maxA = self.policy.sortedA(ns_)[0]
            target = discountedR if ndone else discountedR + self.gamma ** self.n_steps * self.policy.calcQ(
                self.policy.targetNet, ns_,
                maxA)
            predict = Qpredict[i]
            loss += (target - predict) ** 2
        return loss / count

    def L2(self, parameters):
        loss = 0
        for p in parameters:
            loss += (p ** 2).sum()
        return loss

    def learn(self):
        self.opt.zero_grad()
        samples, idxs,  = self.sample()
        Qpredict, Qtarget = self.calculate_td_errors(samples)

        for i in range(self.batch_size):
            error = math.fabs(float(Qpredict[i] - Qtarget[i]))
            self.replay.update(idxs[i], error)

        JDQ = self.loss(Qpredict, Qtarget)
        JE = self.JE(samples)
        Jn = self.Jn(samples, Qpredict)
        L2 = self.L2(self.policy.predictNet.parameters())
        J = JDQ + self.lambda2 * JE + self.lambda1 * Jn + self.lambda3 * L2
        J.backward()
        self.opt.step()

        self.counter += 1
        if self.counter % self.tau == 0:
            self.policy.updateTargetNet()
