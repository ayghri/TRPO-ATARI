# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:01:39 2018

@author: gamer
"""



SKIP_FRAMES = 4
USE_SDL = True
STATE_DIM=(96,96)
EPS=1e-6

STEP_PER_BATCH = 1000
MAX_EPISODE_LEN = 10000 
MAX_KL = .04
CG_DAMP = .1
GAMMA = .95
NUM_EPISODES = 20