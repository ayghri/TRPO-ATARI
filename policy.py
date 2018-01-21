# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:07:46 2018

@author: gamer
"""

import tensorflow as tf
import prettytensor as pt
import config
import numpy as np
import random
import utils

class Policy(object):
    
    def __init__(self,env,session):
        
        self.session = session

        self.actions_set = env.actions_set

        self.actions_dim = env.actions_dim        

        #self.state = tf.placeholder(tf.float32, shape=[None,env.states_dim[0],env.states_dim[1],env.states_dim[2]], name="state")
        self.state = tf.placeholder(tf.float32, shape=[None,np.prod(env.states_dim)], name="state")

        self.pi_theta_old = tf.placeholder(tf.float32, shape=[None, self.actions_dim], name="pi_theta_old")
        #self.pi_theta, _ = (pt.wrap(self.state).fully_connected(64, activation_fn=tf.nn.tanh).softmax_classifier(self.actions_dim))

        self.create_policy()

        self.explore = True
        
    def actions_dist(self,state):
  #      if self.explore:
   #         return np.ones(4)/4
    #    else:
        return self.session.run(self.pi_theta, {self.state: state})[0]
        

    def create_policy(self):
        self.pi_theta, _ = (pt.wrap(self.state).fully_connected(64, activation_fn=tf.nn.tanh).
                                    softmax_classifier(self.actions_dim))

    def create_policy2(self):
    
        self.first_layer = tf.layers.conv2d(inputs=self.state, filters=4,
                    kernel_size=[4, 4], activation=tf.nn.relu)
        
        self.second_layer = tf.layers.conv2d(inputs=self.first_layer, filters=4,
                    kernel_size=[4, 4], activation=tf.nn.tanh)
        
        self.pi_theta =  tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.second_layer),
                                                           num_outputs = self.actions_dim,
        activation_fn=tf.nn.softmax)# weights_initializer=tf.zeros_initializer(),biases_initializer=tf.zeros_initializer())
