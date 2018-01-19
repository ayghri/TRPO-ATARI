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

class Policy(object):
    
    def __init__(self,env,session):
        self.session = session
        self.actions_set = env.actions_set

        self.actions_dim = env.actions_dim        

        self.state = tf.placeholder(tf.float32, shape=[None,env.states_dim], name="state")

        self.pi_theta_old = tf.placeholder(tf.float32, shape=[None, self.actions_dim], name="pi_theta_old")
        self.pi_theta, _ = (pt.wrap(self.state).fully_connected(64, activation_fn=tf.nn.tanh)
            #.fully_connected(64, activation_fn=tf.nn.relu)
                            .softmax_classifier(self.actions_dim))
        self.explore = True
        
    def actions_dist(self,state):
  #      if self.explore:
   #         return np.ones(4)/4
    #    else:
        return self.session.run(self.pi_theta, {self.state: state})[0]
        

        
