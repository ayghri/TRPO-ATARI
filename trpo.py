# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:58:51 2018

@author: gamer
"""

import policy
import utils
import config
import tensorflow as tf
import numpy as np
import time

class TRPO(object):
    
    def __init__(self,env,session):
        
        self.env = env
        self.train_phase = True
        self.session = session
        self.img_size = config.STATE_DIM
        
        self.policy= policy.Policy(env,self.session)
        
        self.action = tf.placeholder(tf.int64, shape=[None], name="action")
        self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")
        
        self.flat_tangent = tf.placeholder(tf.float32, shape=[None])
        self.N= tf.shape(self.policy.state)[0]
        self.Nf = tf.cast(self.N, tf.float32)
        
        self.create_functions()
        
        
    
    def create_functions(self):
        
        
        
        eps = config.EPS
        self.var_list = tf.trainable_variables()
        
        #print("Before Surr Ok !")
        self.create_surr()

        #self.KL = (tf.reduce_sum(self.policy.pi_theta_old * 
        #            tf.log((self.policy.pi_theta_old + eps) /
        #            (self.policy.pi_theta + eps))) / self.Nf)

        self.KL = (tf.reduce_sum(self.policy.pi_theta * 
                    tf.log((self.policy.pi_theta + eps) /
                    (self.policy.pi_theta_old + eps)))) / self.Nf
                    
        self.entropy = (tf.reduce_sum(-self.policy.pi_theta *
                        tf.log(self.policy.pi_theta + eps)) / self.Nf)
                        
        
        """
        self.KL_firstfixed = tf.reduce_sum(tf.stop_gradient(self.policy.pi_theta)*
                tf.log(tf.stop_gradient(self.policy.pi_theta + eps) /
                (self.policy.pi_theta + eps))) / self.Nf
            """
        
        self.KL_firstfixed = tf.reduce_sum(self.policy.pi_theta*
                tf.log((self.policy.pi_theta + eps) /
                (tf.stop_gradient(self.policy.pi_theta + eps)))) / self.Nf
                
        self.KL_firstfixed_grad = tf.gradients(self.KL_firstfixed, self.var_list)
        
        shapes = map(utils.var_shape, self.var_list)
        
        start = 0
        self.tangents = []

        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            self.tangents.append(param)
            start += size
        
        

        self.fisher_vect_prod = (utils.flatgrad([tf.reduce_sum(g * t) for (g, t) in 
                            zip(self.KL_firstfixed_grad, self.tangents)],
                     self.var_list))
        
        self.current_theta = utils.GetFlat(self.session, self.var_list)
        
        self.set_theta = utils.SetFromFlat(self.session, self.var_list)
        
        self.value_func = utils.ValueFunction(self.session)
        self.stats = []
        self.saver = tf.train.Saver()

        
    def create_surr(self):
        p_n = utils.slice_2d(self.policy.pi_theta,tf.range(0, self.N), self.action)
        p_n_old = utils.slice_2d(self.policy.pi_theta_old,tf.range(0, self.N), self.action)
        
        # Surrogate Loss        
        self.surr_loss = - tf.reduce_mean(p_n/p_n_old * self.advantage)
        self.surr_loss_grad = utils.flatgrad(self.surr_loss, self.var_list)
        
    def act(self,state):
        pi = self.policy.actions_dist(state)
        action = utils.choice_weighted(pi)
        return action,pi
    
    def train(self):
        
        start_time = time.time()

        self.episodes = self.env.generate_episodes(config.NUM_EPISODES,self)

        # Computing returns and estimating advantage function.
        for episode in self.episodes:
            episode["baseline"] = self.value_func.predict(episode)
            episode["returns"] = utils.discount(episode["rewards"], config.GAMMA)
            episode["advantage"] = episode["returns"] - episode["baseline"]

        # Updating policy.
        actions_dist_n = np.concatenate([episode["actions_dist"] for episode in self.episodes])
        states_n = np.concatenate([episode["states"] for episode in self.episodes])
        actions_n = np.concatenate([episode["actions"] for episode in self.episodes])
        baseline_n = np.concatenate([episode["baseline"] for episode in self.episodes])
        returns_n = np.concatenate([episode["returns"] for episode in self.episodes])

        # Standardize the advantage function to have mean=0 and std=1.
        advantage_n = np.concatenate([episode["advantage"] for episode in self.episodes])
        advantage_n -= advantage_n.mean()
        advantage_n /= (advantage_n.std() + 1e-8)
        
        
        # Computing baseline function for next iter.
        print(states_n.shape, actions_n.shape, advantage_n.shape,actions_dist_n.shape)
        feed = {self.policy.state: states_n, self.action: actions_n, self.advantage: advantage_n,
                self.policy.pi_theta_old: actions_dist_n}


        episoderewards = np.array([episode["rewards"].sum() for episode in self.episodes])

        #print("\n********** Iteration %i ************" % i)
        
        self.value_func.fit(self.episodes)
        self.theta_old = self.current_theta()

        def fisher_vector_product(p):
            feed[self.flat_tangent] = p
            return self.session.run(self.fisher_vect_prod, feed) + config.CG_DAMP * p

        self.g = self.session.run(self.surr_loss_grad, feed_dict=feed)

        self.grad_step = utils.conjugate_gradient(fisher_vector_product, -self.g)
        
        self.sAs = .5 * self.grad_step.dot(fisher_vector_product(self.grad_step))
        
        self.beta_inv = np.sqrt(self.sAs/config.MAX_KL)
        self.full_grad_step = self.grad_step/self.beta_inv
        
        self.negdot_grad_step = -self.g.dot(self.grad_step)

        def loss(th):
            self.set_theta(th)
            return self.session.run(self.surr_loss, feed_dict=feed)
            
        self.theta = utils.line_search(loss, self.theta_old, self.full_grad_step, self.negdot_grad_step / self.beta_inv)
        self.set_theta(self.theta)

        surr_loss_new = - self.session.run(self.surr_loss, feed_dict=feed)
        KL_old_new = self.session.run(self.KL, feed_dict=feed)
        entropy = self.session.run(self.entropy, feed_dict=feed)
        
        old_new_norm = np.sum((self.theta-self.theta_old)**2)
        
        if np.abs(KL_old_new) > 2.0 * config.MAX_KL:
            print("Keeping old theta")
            self.set_theta(self.theta_old)

        stats = {}
        stats["L2 of old - new"] = old_new_norm
        stats["Total number of episodes"] = len(self.episodes)
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["Entropy"] = entropy
        exp = utils.explained_variance(np.array(baseline_n), np.array(returns_n))
        stats["Baseline explained"] = exp
        stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
        stats["KL between old and new distribution"] = KL_old_new
        stats["Surrogate loss"] = surr_loss_new
        self.stats.append(stats)
        utils.write_dict(stats)
        save_path = self.saver.save(self.session, "./checkpoints/model.ckpt")
        print('Saved checkpoint to %s'%save_path)
        for k, v in stats.items():
            print(k + ": " + " " * (40 - len(k)) + str(v))
