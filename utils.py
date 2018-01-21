# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:24:22 2018

@author: gamer
"""
import numpy as np
import tensorflow as tf
import prettytensor as pt
import scipy.signal
import config

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    
    
def line_search(f, x, fullstep, expected_improve_rate):
    """ We perform the line search in direction of fullstep, we shrink the step 
        exponentially (multi by beta**n) until the objective improves.
        Without this line search, the algorithm occasionally computes
        large steps that cause a catastrophic degradation of performance

        f : callable , function to improve    
        x : starting evaluation    
        fullstep : the maximal value of the step length
        expected_improve_rate : stop if 
                    improvement_at_step_n/(expected_improve_rate*beta**n)>0.1
    """
    
    accept_ratio = config.LN_ACCEPT_RATE
    max_backtracks = 10
    fval = f(x)
    stepfrac=1
    stepfrac=stepfrac*0.5
    for stepfrac in .5**np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew

    return x


def conjugate_gradient(f_Ax, b, n_iters=10, gtol=1e-10):
    """Search for Ax-b=0 solution using conjugate gradient algorithm
       
        f_Ax : callable, f(x, *args) (returns A.dot(x) with A Symetric Definite)
        b : b such we search for Ax=b
        cg_iter : max number of iterations
        gtol: iterations stop when norm(residual) < gtol
    """
    
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for _ in range(n_iters):
        if rdotr < gtol:
            break
        z = f_Ax(p)
        alpha = rdotr / p.dot(z)
        x += alpha * p
        r -= alpha * z
        newrdotr = r.dot(r)
        beta = newrdotr / rdotr
        p = r + beta * p
        rdotr = newrdotr
        
    return x
    
def choice_weighted(pi):
#    np.random.seed(np.random.randint(0,2**10))
    #print(pi.shape)
    return np.random.choice(np.arange(len(pi)), 1, p=pi)[0]
        
        
class ValueFunction(object):
    def __init__(self, session):
        self.net = None
        self.session = session

    def create_net(self, shape):
        print(shape)
        self.x = tf.placeholder(tf.float32, shape=[None, shape], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.net = (pt.wrap(self.x).fully_connected(64,activation_fn=tf.nn.relu).
                    fully_connected(64, activation_fn=tf.nn.relu).fully_connected(1))
                    
        self.net = tf.reshape(self.net, (-1, ))
        l2 = (self.net - self.y) * (self.net - self.y)
        self.train = tf.train.AdamOptimizer().minimize(l2)
        
        initialize_uninitialized(self.session)
        

    def _features(self, episode):
        o = episode["states"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        act = episode["actions_dist"].astype('float32')
        l = len(episode["rewards"])
        al = np.arange(l).reshape(-1, 1) / 10.0
        ret = np.concatenate([o, act, al, np.ones((l, 1))], axis=1)
        return ret

    def fit(self, episodes):
        featmat = np.concatenate([self._features(episode) for episode in episodes])
        if self.net is None:
            self.create_net(featmat.shape[1])
        returns = np.concatenate([episode["returns"] for episode in episodes])
        for _ in range(50):
            self.session.run(self.train, {self.x: featmat, self.y: returns})

    def predict(self, episode):
        if self.net is None:
            return np.zeros(len(episode["rewards"])) 
        else:
            ret = self.session.run(self.net, {self.x: self._features(episode)})
            return np.reshape(ret, (ret.shape[0], ))
            
def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(g, [numel(v)])for (v, g) in zip(var_list, grads)],0)
    
class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list],0)

    def __call__(self):
        return self.op.eval(session=self.session)

def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        shapes = list(map(var_shape, var_list))
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        i = 0
        for v in var_list:
            size = np.prod(shapes[i])
            #assigns.append(tf.assign(v,tf.reshape(self.theta[start:start +size],shapes[i])))
            assigns.append(tf.assign(v,tf.reshape(self.theta[start:start +size],shapes[i])))
            i = i+1
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})
        

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
        
def initialize_zeros(sess):
    global_vars          = tf.global_variables()
    assigns = []
    for g in global_vars:
        assigns.append(tf.assign(g, tf.zeros_like(g)))
    sess.run(assigns)
    
def write_dict(dic):

    fout = "./here.txt"
    fo = open(fout, "a+")
    fo.write('\n'+'-'*10+'\n')
    for k, v in dic.items():
        fo.write(str(k) + ' >>> '+ str(v) + '\n')
    fo.close()



