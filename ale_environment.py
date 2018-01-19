# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:59:20 2018

@author: gamer
"""
import config
import numpy as np
import tensorflow as tf
from ale_python_interface import ALEInterface
from collections import deque
import img_functions as imf



class ALE_ENVIRONMENT(object):
    
    def __init__(self,rom_file,session):
        
        self.load_rom(rom_file)
        self.session = session        
        
        self.actions_set = self.ale.getMinimalActionSet().tolist()
        self.action_to_index = {a:i for i,a in enumerate(self.actions_set)}
        self.state_dummy = np.zeros((1,np.prod(config.STATE_DIM)))
        
        self.states_dim = np.prod(config.STATE_DIM)
        self.states_dimF = tf.cast(self.states_dim,tf.float32)
        self.actions_dim = len(self.actions_set)
        
        self.preprocess_stack = deque([], 2)

    def load_rom(self,rom_file):

        self.ale = ALEInterface()
        self.ale.setInt(str.encode('random_seed'), 123)
        self.ale.setFloat(str.encode('repeat_action_probability'), 0.0)        
        self.ale.setBool(str.encode('sound'), False)
        self.ale.setBool(str.encode('display_screen'), config.USE_SDL)
        self.ale.loadROM(str.encode(rom_file))
        
    def generate_episodes(self,num_episodes,agent):
        
        episodes = []
        lives = self.ale.lives()
        for i in range(num_episodes):
            print("Starting episode %d"%i)
            episode_done = False
            
            states = []
            actions_dist =[]
            actions = [0]
            rewards = [0]

            self.skip_frames(states,config.SKIP_FRAMES)
            
            while not episode_done:
                action_idx,action_dist = agent.act(states[-1])
                reward = 0
                for _ in range(config.SKIP_FRAMES):
                    reward = reward + self.ale.act(self.actions_set[action_idx])
                    self.preprocess_stack.append(self.ale.getScreenRGB())
                state = imf.preprocess(self.preprocess_stack)
                states.append(state)
                actions_dist.append([action_dist])
                actions.append(action_idx)
                rewards.append(reward)
                
                episode_done = self.ale.game_over() or (self.ale.lives() < lives)

            episodes.append({'states':np.concatenate(states[1:]),
                            'actions_dist':np.concatenate(actions_dist),
                            'actions':np.array(actions[1:]),
                            'rewards':np.array(rewards[1:])})
            self.ale.reset_game()
        return episodes
        
    def reset(self):
        self.ale.reset_game()
        self.preprocess_stack = deque([], 2)
        
    def skip_frames(self,states,num_frames):
    #perform nullops
        for _ in range(num_frames):
            self.ale.act(3)
            self.preprocess_stack.append(self.ale.getScreenRGB())
        
        if len(self.preprocess_stack) < 2:
            self.ale.act(3)
            self.preprocess_stack.append(self.ale.getScreenRGB())
        states.append(imf.preprocess(self.preprocess_stack))
            
            