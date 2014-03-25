#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import csv
import random
import glob
import os
import sys

import numpy
import pylab

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from svmutil import *

N_MONTH = 4
N_DAY_PER_MONTH = 31
BASE_MONTH = 4
TYPE_LENGTH = 4

#Don't use a python long as this don't work on 32 bits computers.
numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False

class User(object):
    def __init__(self, id, info):
        self.id = id;
        self.bands = info.keys()
        self.data = numpy.zeros((len(info), N_MONTH * 3 * TYPE_LENGTH * 8), dtype=int)
        self.label = []
        for idx, brandID in enumerate(self.bands):
            band = info[brandID]
            row = [0 for n in range(48)]
            label = 0
            for month, day, action in band:
                p = (month - BASE_MONTH) * 12
                if day > 10:
                    p += 4
                elif day > 20:
                    p += 8
                row[p + action] = min(255, row[p + action] + 1)

                if month == BASE_MONTH + N_MONTH - 1 and action == 1:
                    label = 1

            self.label.append(label)
            self.data[idx, :] = numpy.mat([list(format(num, '08b')) for num in row]).flatten()
        self.data = self.data.astype(numpy.float32)

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))

def build_rbm(v, W, bv, bh, k):
    '''Construct a k-step Gibbs chain starting at v for an RBM.

v : Theano vector or matrix
  If a matrix, multiple chains will be run in parallel (batch).
W : Theano matrix
  Weight matrix of the RBM.
bv : Theano vector
  Visible bias vector of the RBM.
bh : Theano vector
  Hidden bias vector of the RBM.
k : scalar or Theano scalar
  Length of the Gibbs chain.

Return a (v_sample, cost, monitor, updates) tuple:

v_sample : Theano vector or matrix with the same shape as `v`
  Corresponds to the generated sample(s).
cost : Theano scalar
  Expression whose gradient with respect to W, bv, bh is the CD-k approximation
  to the log-likelihood of `v` (training example) under the RBM.
  The cost is averaged in the batch case.
monitor: Theano scalar
  Pseudo log-likelihood (also averaged in the batch case).
updates: dictionary of Theano variable -> Theano variable
  The `updates` object returned by scan.'''

    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates


def shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent):
    '''Construct a symbolic RNN-RBM and initialize parameters.

n_visible : integer
  Number of visible units.
n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.

Return a (v, v_sample, cost, monitor, params, updates_train, v_t,
          updates_generate) tuple:

v : Theano matrix
  Symbolic variable holding an input sequence (used during training)
v_sample : Theano matrix
  Symbolic variable holding the negative particles for CD log-likelihood
  gradient estimation (used during training)
cost : Theano scalar
  Expression whose gradient (considering v_sample constant) corresponds to the
  LL gradient of the RNN-RBM (used during training)
monitor : Theano scalar
  Frame-level pseudo-likelihood (useful for monitoring during training)
params : tuple of Theano shared variables
  The parameters of the model to be optimized during training.
updates_train : dictionary of Theano variable -> Theano variable
  Update object that should be passed to theano.function when compiling the
  training function.
v_t : Theano matrix
  Symbolic variable holding a generated sequence (used during sampling)
updates_generate : dictionary of Theano variable -> Theano variable
  Update object that should be passed to theano.function when compiling the
  generation function.'''

    W = shared_normal(n_visible, n_hidden, 0.01)
    bv = shared_zeros(n_visible)
    bh = shared_zeros(n_hidden)
    Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bu = shared_zeros(n_hidden_recurrent)

    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu  # learned parameters as shared
                                                # variables

    v = T.matrix()  # a training sequence
    u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden
                                         # units

    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence(v_t, u_tm1, generate=False):
        bv_t = bv + T.dot(u_tm1, Wuv)
        bh_t = bh + T.dot(u_tm1, Wuh)
        if generate:
            v_t, _, _, updates = build_rbm(v_t, W, bv_t,
                                           bh_t, k=25)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.
    (u_t, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1),
        sequences=v, outputs_info=[u0, None, None], non_sequences=params)
    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:],
                                                     k=15)
    updates_train.update(updates_rbm)

    # symbolic loop for sequence generation
    (v_t, u_t), updates_generate = theano.scan(
        lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1, True),
        sequences=v, outputs_info=[None, u0], non_sequences=params, n_steps=1)

    return (v, v_sample, cost, monitor, params, updates_train, v_t,
            updates_generate)


class RnnRbm:
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
sequences.'''

    def __init__(self, n_visible=96, n_hidden=150, n_hidden_recurrent=100, lr=0.001, dt=0.3):
        '''Constructs and compiles Theano functions for training and sequence
generation.

n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.
lr : float
  Learning rate
r : (integer, integer) tuple
  Specifies the pitch range of the piano-roll in MIDI note numbers, including
  r[0] but not r[1], such that r[1]-r[0] is the number of visible units of the
  RBM at a given time step. The default (21, 109) corresponds to the full range
  of piano (88 notes).
dt : float
  Sampling period when converting the MIDI files into piano-rolls, or
  equivalently the time difference between consecutive time steps.'''

        self.dt = dt
        (v, v_sample, cost, monitor, params, updates_train, v_t,
         updates_generate) = build_rnnrbm(n_visible, n_hidden,
                                           n_hidden_recurrent)

        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update(((p, p - lr * g) for p, g in zip(params,
                                                                gradient)))
        self.train_function = theano.function([v], monitor,
                                               updates=updates_train)
        self.generate_function = theano.function([v], v_t,
                                                 updates=updates_generate)

    def train(self, userTrain, userVal, batch_size=100, num_epochs=200):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
files converted to piano-rolls.

files : list of strings
  List of MIDI files that will be loaded as piano-rolls for training.
batch_size : integer
  Training sequences will be split into subsequences of at most this size
  before applying the SGD updates.
num_epochs : integer
  Number of epochs (pass over the training set) performed. The user can
  safely interrupt training with Ctrl+C at any time.'''

        dataTrain = numpy.vstack([user.data for user in userTrain])
        labelTrain = []
        for user in userTrain:
            labelTrain.extend(user.label)

        try:
            for epoch in xrange(num_epochs):
                combine = zip(dataTrain, labelTrain)
                numpy.random.shuffle(combine)

                costs = []

                for s, sequence in enumerate(dataTrain):
                    cost = self.train_function(numpy.reshape(sequence, (4, 96)))
                    costs.append(cost)

                print 'Epoch %i/%i' % (epoch + 1, num_epochs),
                print numpy.mean(costs)
                sys.stdout.flush()

                generateData = []
                for s, sequence in enumerate(dataTrain):
                    sample = self.generate_function(numpy.reshape(sequence, (4, 96))[0:3, :])
                    generateData.append(sample.flatten().tolist())

                self.svm = svm_train(labelTrain, generateData, '-t 0 -q')
                print 'Training:  %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % self.eval(userTrain)
                print 'Validiate: %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % self.eval(userVal)
                print ''
                sys.stdout.flush()

        except KeyboardInterrupt:
            print 'Interrupted by user.'

    def eval(self, evalUser):
        pBands = []
        bBands = []
        hitBands = []
        for user in evalUser:
            bBands.append(sum(user.label))

            hit = 0
            total = 0
            for idx, label in enumerate(user.label):
                data = self.generate_function(numpy.reshape(user.data[idx, :], (4, 96))[0:3, :])
                predict, acc, prob = svm_predict([label], [data.flatten().tolist()], self.svm, '-q')
                if predict == 1:
                    total += 1
                    if label == 1:
                        hit += 1

            hitBands.append(hit)
            pBands.append(total)

        precision = float(sum(hitBands)) / sum(pBands) if not sum(pBands) == 0 else 0
        recall = float(sum(hitBands)) / sum(bBands) if not sum(bBands) == 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if not precision + recall == 0 else 0

        return precision, recall, f1

    def generate(self, filename, show=True):
        '''Generate a sample sequence, plot the resulting piano-roll and save
it as a MIDI file.

filename : string
  A MIDI file will be created at this location.
show : boolean
  If True, a piano-roll of the generated sequence will be shown.'''

        piano_roll = self.generate_function()
        midiwrite(filename, piano_roll, self.r, self.dt)
        if show:
            extent = (0, self.dt * len(piano_roll)) + self.r
            pylab.figure()
            pylab.imshow(piano_roll.T, origin='lower', aspect='auto',
                         interpolation='nearest', cmap=pylab.cm.gray_r,
                         extent=extent)
            pylab.xlabel('time (s)')
            pylab.ylabel('MIDI note number')
            pylab.title('generated piano-roll')


def test_rnnrbm(userTrain, userVal, batch_size=100, num_epochs=200):
    model = RnnRbm()
    model.train(userTrain, userVal, 
                batch_size=batch_size, num_epochs=num_epochs)
    return model


if __name__ == '__main__':
    userInfo = dict()
    with open('/home/pumpkin/Documents/project/tmall/dataset/t_alibaba_data.csv', 'rb') as csvfile:
    # with open('/home/pumpkin/Documents/project/tmall/dataset/demo.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            userID, brandID, actionType, month, day = [int(field) for field in row]
            if not userID in userInfo:
                userInfo[userID] = dict()

            user = userInfo[userID]
            if brandID not in user:
                user[brandID] = []

            if month in (4, 5, 6):
                day = day - 14
            else:
                day = day - 15
            if day <= 0:
                month -= 1
                day += 31

            band = user[brandID]
            band.append((month, day, actionType))

    users = []
    for (userID, info) in userInfo.iteritems():
        users.append(User(userID, info))

    nUsers = len(users)
    nTrain = int(0.6 * nUsers)
    nVal = int(0.2 * nUsers)
    nTest = nUsers - nTrain - nVal
    print 'Num of users: ', len(users)
    print 'Num for train: ', nTrain
    print 'Num for validiate: ', nVal
    print 'Num for test: ', nTest

    random.shuffle(users)
    userTrain = users[0: nTrain]
    userVal = users[nTrain: nTrain+nVal]
    userTest = users[nTrain+nVal: nUsers]

    model = test_rnnrbm(userTrain, userVal, batch_size=4)
    # numpy.savetxt("foo.csv", users[0].data, delimiter=",", fmt='%d')


