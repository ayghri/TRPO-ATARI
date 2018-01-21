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
import config



class ALE_ENVIRONMENT(object):
    
    def __init__(self,rom_file,session):
        
        self.load_rom(rom_file)
        self.session = session        
        
        self.actions_set = self.ale.getMinimalActionSet().tolist()
        self.action_to_index = {a:i for i,a in enumerate(self.actions_set)}
        #self.state_dummy = 255*np.ones((1,config.STATE_DIM[0],config.STATE_DIM[1],config.SKIP_FRAMES))
        self.states_dim = (config.STATE_DIM[0],config.STATE_DIM[1],config.SKIP_FRAMES)        
        self.state_dummy = 255*np.ones((1,np.prod(self.states_dim)))
        
        self.actions_dim = len(self.actions_set)
        
        self.preprocess_stack = deque([], config.SKIP_FRAMES)
        
        self.max_episode_len = config.MAX_EPISODE_LEN
        

    def load_rom(self,rom_file):

        self.ale = ALEInterface()
        self.ale.setInt(str.encode('random_seed'), 123)
        self.ale.setFloat(str.encode('repeat_action_probability'), 0.0)        
        self.ale.setBool(str.encode('sound'), False)
        self.ale.setBool(str.encode('display_screen'), config.USE_SDL)
        self.ale.loadROM(str.encode(rom_file))
        
    def generate_episodes(self,num_episodes,agent):
        print("Generateting %d episodes"%num_episodes)
        episodes = []
        lives = self.ale.lives()
        _,dist = agent.act(self.state_dummy)
        print("pi_theta for blank white state:",dist)
        for i in range(num_episodes):
            print("Starting episode %d"%i)
            episode_done = False
            
            states = []
            actions_dist =[]
            actions = [0]
            rewards = [0]

            self.skip_frames(states,config.SKIP_FRAMES)
            i = 0
            while not episode_done and i<self.max_episode_len:
                action_idx,action_dist = agent.act(states[-1])
                
                reward = 0
                for _ in range(config.SKIP_FRAMES):
                    reward = reward + np.clip(self.ale.act(self.actions_set[action_idx]),-1,1)
                    self.preprocess_stack.append(self.ale.getScreenRGB())
                    i += 1
                state = imf.preprocess(self.preprocess_stack,True)
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
        self.preprocess_stack = deque([], config.SKIP_FRAMES)
        
    def skip_frames(self,states,num_frames):
    #perform nullops
        for _ in range(num_frames):
            self.ale.act(0)
            self.preprocess_stack.append(self.ale.getScreenRGB())
        
        if len(self.preprocess_stack) < config.SKIP_FRAMES:
            self.ale.act(0)
            self.preprocess_stack.append(self.ale.getScreenRGB())
        states.append(imf.preprocess(self.preprocess_stack,True))
            
            