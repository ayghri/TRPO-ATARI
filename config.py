# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:01:39 2018

@author: gamer
"""



SKIP_FRAMES = 3
USE_SDL = True
STATE_DIM = (64,64)
EPS = 1e-8

STEP_PER_BATCH = 1000
MAX_EPISODE_LEN = 5000
MAX_KL = .01
CG_DAMP = .1
GAMMA = .95
NUM_EPISODES = 15
LN_ACCEPT_RATE = 0.1