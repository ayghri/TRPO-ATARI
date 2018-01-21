# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:50:11 2018

@author: gamer
"""

import ale_environment
import trpo
import img_functions as imf
import time
import tensorflow as tf
import numpy as np
import utils

def main():
    session = tf.Session()
    env = ale_environment.ALE_ENVIRONMENT('./roms/breakout.bin',session)
    agent = trpo.TRPO(env,session)
    #session.run(tf.global_variables_initializer())
    utils.initialize_zeros(session)
    for _ in range(100): agent.train()

