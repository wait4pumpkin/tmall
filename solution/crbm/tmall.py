#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import csv
import random
import glob
import os
import sys
import time

import numpy
import pylab

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


N_MONTH = 4
N_DAY_PER_MONTH = 31
BASE_MONTH = 4
TYPE_LENGTH = 4

class User(object):
    def __init__(self, id, info, total):
        self.id = id;
        self.brands = info.keys()
        self.data = numpy.zeros((total, 4), dtype=int)
        self.data_mask = numpy.zeros((total, 4), dtype=int)
        self.test_label = set()
        self.train_label = set()
        for brandID in self.brands:
            brand = info[brandID]
            for month, day, action in brand:
                p = (month - BASE_MONTH) * 12
                if day > 10:
                    p += 4
                elif day > 20:
                    p += 8

                if month == BASE_MONTH + N_MONTH - 1:
                    if action == 1:
                        self.test_label.add(brandID)
                else:
                    # self.data[brandID, action] += 1
                    self.data[brandID, action] = 1
                    self.data_mask[brandID, :] = 1
                    self.train_label.add(brandID)
            
        self.data = self.data.flatten().astype(theano.config.floatX)
        self.data_mask = self.data_mask.flatten().astype(theano.config.floatX)

    def __str__(self):
        return str(self.id) + ' ' + str(len(self.bands))


class RBM(object):
    def __init__(self, input, input_mask, 
        n_visible=784, n_hidden=500, weight_decay=0.001, 
        W_vh=None, W_uh=None, hbias=None, vbias=None, numpy_rng=None,
        theano_rng=None):
        """
        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weight_decay = weight_decay

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W_vh is None:
            initial_W_vh = numpy.asarray(numpy_rng.uniform(
                          low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                          high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                          size=(n_visible, n_hidden)),
                      dtype=theano.config.floatX)
            W_vh = theano.shared(value=initial_W_vh, name='W_vh', borrow=True)

        if W_uh is None:
            initial_W_uh = numpy.asarray(0.01 * numpy_rng.randn(
                            n_visible, n_hidden),
                        dtype=theano.config.floatX)
            W_uh = theano.shared(value=initial_W_uh, name='W_uh', borrow=True)

        if hbias is None:
            hbias = theano.shared(value=numpy.zeros(n_hidden,
                                                    dtype=theano.config.floatX),
                                  name='hbias', borrow=True)

        if vbias is None:
            vbias = theano.shared(value=numpy.zeros(n_visible,
                                                    dtype=theano.config.floatX),
                                  name='vbias', borrow=True)

        self.input = input
        self.input_mask = input_mask
        
        self.W_vh = W_vh
        self.W_uh = W_uh
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W_vh, self.W_uh, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample * self.input_mask, self.W_vh) + T.dot(self.input_mask, self.W_uh) + self.hbias
        vbias_term = T.dot(v_sample * self.input_mask, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units
        '''
        pre_sigmoid_activation = T.dot(vis, self.W_vh) \
                               + T.dot(self.input_mask, self.W_uh) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample * self.input_mask)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units
        '''
        pre_sigmoid_activation = T.dot(hid, self.W_vh.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean * self.input_mask,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, momentum=0.9, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh,
                    outputs_info=[None,  None,  None, None, None, ph_sample],
                    n_steps=k)
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) \
             - T.mean(self.free_energy(chain_end)) \
             + self.weight_decay * T.sum(self.W_vh ** 2)
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            # if param in updates:
                # updates[param] = param + momentum * updates[param] - gparam * T.cast(lr, dtype=theano.config.floatX)
            # else:
                updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
        
        monitoring_cost = self.get_reconstruction_cost(updates,
                                                       pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        cross_entropy = T.mean(
                T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                      axis=1))

        return cross_entropy



if __name__ == '__main__':
    userInfo = dict()
    with open('/home/pumpkin/Documents/project/tmall/dataset/t_alibaba_data.csv', 'rb') as csvfile:
    # with open('/home/pumpkin/Documents/project/tmall/dataset/demo.csv', 'rb') as csvfile:
        brand_table = dict()
        brand_counter = 0
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            userID, brandID, actionType, month, day = [int(field) for field in row]
            if not userID in userInfo:
                userInfo[userID] = dict()

            if not brandID in brand_table:
                brand_table[brandID] = brand_counter
                brand_counter += 1
            brandID = brand_table[brandID]

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
        users.append(User(userID, info, brand_counter))

    nUsers = len(users)
    train_data_raw = numpy.asarray([user.data for user in users], dtype=theano.config.floatX)
    train_data = theano.shared(train_data_raw, borrow=True)
    train_data_mask_raw = numpy.asarray([user.data_mask for user in users], dtype=theano.config.floatX)
    train_data_mask = theano.shared(train_data_mask_raw, borrow=True)
    print 'Num of users: ', len(users)
    
    learning_rate = 0.01
    training_epochs = 150
    batch_size = 100
    n_hidden = 100

    n_train_batches = nUsers / batch_size

    index = T.lscalar()
    k_step = T.lscalar()
    x = T.matrix('x')
    mask = T.matrix('mask')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    rbm = RBM(input=x, input_mask=mask, n_visible=4 * brand_counter,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate, k=k_step)

    #################################
    #     Training the RBM          #
    #################################
    train_rbm = theano.function([index, k_step], cost,
           updates=updates,
           givens={x: train_data[index * batch_size:
                                 (index + 1) * batch_size], 
                   mask: train_data_mask[index * batch_size:
                                 (index + 1) * batch_size]},
           name='train_rbm')

    start_time = time.clock()
    for epoch in xrange(training_epochs):
        random.shuffle(zip(train_data_raw, train_data_mask_raw))
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            if epoch < 120:
                mean_cost += [train_rbm(batch_index, 1)]
            else:
                mean_cost += [train_rbm(batch_index, 3)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

    end_time = time.clock()

    training_time = (end_time - start_time)
    print ('Training took %f minutes' % (training_time / 60.))


    #################################
    #     Sampling from the RBM     #
    #################################
    top_n = 100
    pBands = []
    bBands = []
    hitBands = []
    data = T.matrix('data')
    presig_hids, hid_mfs, hid_samples, presig_vis, vis_mfs, vis_samples = \
        rbm.gibbs_vhv(data)

    for user in users:
        sample = vis_mfs.eval({data: numpy.mat([user.data]), mask: numpy.mat([user.data_mask])})
        result = sorted(zip(sample[:, 1:-1:4].tolist(), [num for num in range(brand_counter)]), reverse=True)[0:top_n]
        bBands.append(len(user.test_label))

        hit = 0
        total = top_n
        for prob, brand_id in result:
            if brand_id in user.test_label:
                hit += 1

        hitBands.append(hit)
        pBands.append(total)

    print sum(hitBands), ' ', sum(pBands), ' ', sum(bBands)
    precision = float(sum(hitBands)) / sum(pBands) if not sum(pBands) == 0 else 0
    recall = float(sum(hitBands)) / sum(bBands) if not sum(bBands) == 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if not precision + recall == 0 else 0
    print 'All:  %.02f%% (Precision) %.02f%% (Recall) %.02f%% (F1)' % (precision * 100, recall * 100, f1 * 100)
