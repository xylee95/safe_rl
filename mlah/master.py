import gym
import gym_mlah
import tensorflow as tf
import rollouts
from policy_network import Policy
from subpolicy_network import SubPolicy
#from observation_network import Features
from learner import Learner
import rl_algs.common.tf_util as U
import numpy as np
# from tinkerbell import logger
import pickle
#from grad_attack import *
from adversaries.adversaries import *
from collections import deque
import csv
import os
import errno

def start(args, workerseed, rank, comm):
    env = gym.make(args.task)
    env_eval = gym.make(args.task)

    env.seed(workerseed)
    env.set_experiment_id(args.id_number)

    ob_space = env.observation_space
    master_ob = gym.spaces.Box(np.array([-100,-100],dtype=np.float32),np.array([100,100],dtype=np.float32))
    ac_space = env.action_space

    num_subs = args.num_subs
    num_rollouts = args.num_rollouts
    train_time = args.train_time

    num_batches = int(num_rollouts/64)
    print(num_batches)

    # observation in.
    ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])
    adv_ob = U.get_placeholder(name="adv_ob",dtype=tf.float32, shape=[None,master_ob.shape[0]])

    master_policy = Policy(name="master", ob=adv_ob, ac_space=0, hid_size=16, num_hid_layers=2, num_subpolicies=2)
    old_master_policy = Policy(name="old_master", ob=adv_ob, ac_space=0, hid_size=16, num_hid_layers=2, num_subpolicies=2)

    sub_policies = [SubPolicy(name="sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]
    old_sub_policies = [SubPolicy(name="old_sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]

    learner = Learner(env,master_policy,old_master_policy,sub_policies, old_sub_policies, comm, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64)

    adv_generator = adv_gen(1.0, ob_space, perturb_func= grid_reflect_bias, delay=num_rollouts*args.warmup_time,augmented=args.augment)
    adv_generator_eval = adv_gen(-1.0, ob_space, perturb_func= grid_reflect_bias)

    override=None

    rollout = rollouts.traj_segment_generator(adv_generator, master_policy, sub_policies, env, num_rollouts, stochastic=True, args=args)
    rollout_eval = rollouts.traj_segment_generator(adv_generator_eval, master_policy, sub_policies, env_eval, 1, stochastic=False, args=args)

    ret_buffer = deque(maxlen=20)
    ret_buffer_eval = deque(maxlen=20)

    fname = './data/'+args.filename +'.csv'
    file  = open(fname,'w')
    writer = csv.writer(file)
    if args.load is not None:
        fname = osp.join("./savedir/",args.load, args.load)
        U.load_state(fname)
    #saver = tf.train.Saver()

    #callback(0)


    learner.syncSubpolicies()
    print("synced subpols")

    master_train = True
    sub_train = [True, True]
    goal_t = 0
    mini_ep=0
    totalmeans = []
    while mini_ep < args.warmup_time + train_time:

        mini_ep += 1
        if(mini_ep==args.warmup_time):
            print("===================")
            print("START TRAINING WITH")
            print("===================")
            args.pretrain = -1
            sub_train = [False,True]
        #if(mini_ep == 200):
         #   adv_generator.perturb_func = stoch_bias

        rolls = rollout.__next__()
        allrolls = []
        allrolls.append(rolls)
        # train theta
        rollouts.add_advantage_macro(rolls, 0.99, 0.98)
        if args.pretrain < 0 and master_train:
            gmean, lmean = learner.updateMasterPolicy(rolls)
        # train phi
        test_seg = rollouts.prepare_allrolls(allrolls, 0.99, 0.98, num_subpolicies=num_subs)
        learner.updateSubPolicies(test_seg, num_batches, sub_train)
        rolls_eval = rollout_eval.__next__()
        # learner.updateSubPolicies(test_seg,
        # log
        ret_buffer.extend(rolls['ep_rets'])
        ret_buffer_eval.extend(rolls_eval['ep_rets'])
        ret_mean = np.mean(ret_buffer)
        ret_eval_mean = np.mean(ret_buffer_eval)
        if len(ret_buffer_eval) == 0: ret_eval_mean =  -100
        fields = [mini_ep, ret_mean,ret_eval_mean,rolls['latent_counts'][0],rolls['latent_counts'][1],rolls['real_counts'][0],rolls['real_counts'][1]]
        writer.writerow(fields)

        print("rollout: {}, avg ep r: {}, avg eval ep r: {}".format(mini_ep,ret_mean, ret_eval_mean))
        print("--------------------------------------------------")
    if args.save is not None:
        fname = osp.join("savedir/", args.save, args.save)
        U.save_state(fname)
        #saver.save(U.get_session(), "./savedir/test/mc30")
            #if args.s:
             #   totalmeans.append(gmean)
             #   with open('outfile'+str(x)+'.pickle', 'wb') as fp:
              #      pickle.dump(totalmeans, fp)
