import tensorflow as tf
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from environments import env1, env2


gamma =0.9
alpha=0.2
mem_size=10000000
batch_size=100
update_counter=1
prob=1
replay =1
STDDEV=0
n_states=10
n_actions=2

W=tf.Variable(tf.random_normal([n_states,n_actions],stddev=STDDEV),name="W")
W_=tf.Variable(tf.random_normal([n_states,n_actions],stddev=STDDEV),name="W_")
b=tf.Variable(tf.zeros([n_actions]),name="b")
b_=tf.Variable(tf.zeros([n_actions]),name="b_")
WeightCopy = W_.assign(W)
BiasCopy = b_.assign(b)

S= tf.placeholder("float", shape = [None, 10], name="State")
A= tf.placeholder("float", shape = [None, n_actions], name="Action")
R= tf.placeholder("float", shape = [None, 1], name="Reward")
S_= tf.placeholder("float", shape = [None, n_states], name="Next_State")

Q=tf.sigmoid(tf.add(tf.matmul(S,W),b))
Q_=tf.sigmoid(tf.add(tf.matmul(S_, W_),b_))

Y= tf.add(R, tf.mul(gamma, Q_))
Y_= tf.placeholder("float", shape=[None, n_actions], name="TargetQ")
cost= tf.reduce_sum(tf.square(tf.sub(Y_, tf.mul(A,Q))))
init=tf.initialize_all_variables()
optimizer=tf.train.GradientDescentOptimizer(alpha).minimize(cost)

sess=tf.InteractiveSession()
tf.get_default_graph().finalize()

#############################################

memory= np.zeros([4,mem_size])
Stable=np.identity(10)
Atable=np.identity(2)

Tot_Reward=np.zeros([20])

for tot_episode in range(10,201,10):
    for epoch in range(1,101,1):
        sess.run(init)
        with tf.device("/gpu:0"):
            counter=1
            tot_time_step = 1
            for episode in range(tot_episode+1):
                start_time = time.time()
                epsilon = 1-float(episode) / float(tot_episode)
                St=Stable[[1],:]
                T=0
                time_step=1
                while(not T):
                    Qt= sess.run(Q, feed_dict={S: St})
                    if np.random.uniform()>0:
                        if Qt[0,0]>Qt[0,1]:
                            At=0
                        elif Qt[0,0]<Qt[0,1]:
                            At=1
                        else:
                            At=np.random.randint(0,2)
                    else:
                        At=np.random.randint(0,2)
                    if At==0:
                        At=np.array([[1,0]])
                    else:
                        At=np.array([[0,1]])
                    Sn, Rn, T=env1(St,At)

                    memory= np.roll(memory, 1, axis=1)
                    memory[0,0]=np.argmax(St)
                    memory[1,0]=np.argmax(At)
                    memory[2,0]=Rn
                    memory[3,0]=np.argmax(Sn)

                    if counter==update_counter:
                        if replay ==1:
                            Sampling = np.random.randint(min(tot_time_step,mem_size),size=(batch_size))
                        else:
                            Sampling=range(batch_size)
                        Sindices=memory[0,Sampling].astype(int).tolist()
                        Supdate=Stable[Sindices,:]
                        Aindices=memory[1,Sampling].astype(int).tolist()
                        Aupdate= Atable[Aindices,:]
                        Rupdate= memory[[2], [Sampling]].T
                        S_indices= memory[3,Sampling].astype(int).tolist()
                        S_update=Stable[S_indices,:]
                        Y_target=Y.eval({R: Rupdate, S_: S_update})
                        sess.run(optimizer,feed_dict={S: Supdate, A: Aupdate,
                            Y_:Y_target})
                        counter =1
                        sess.run(WeightCopy)
                        sess.run(BiasCopy)
                    else:
                        counter = counter+1
                    Stmp=St
                    St=Sn
                    os.system('clear')
                    time_step=time_step+1
                    tot_time_step=tot_time_step+1

                    print "=============================="
                    print "Q learning"
                    print "Annealing:", tot_episode, ", Epoch:", epoch, ", Episode:",episode
                    #print" Time Step:", time_step(time.time()-start_time),"seconds"
                    print "=============================="
                    print "curr State:", Stmp
                    if At[0,0]==1:
                        print "Action: Left"
                    else: 
                        print "Action: Right"
                    print "Reward", Rn
                    print "next State:", Sn
                    print "Q values:"
                    print sess.run(W)
                    print "Trained reward with %d-episode annealing: %d" % (tot_episode, Tot_Reward[tot_episode/10-1])
                    print "Epsilon:", epsilon
                    if(Rn==100):
                        print "============================"
                        print "100 Reward"
                        print "==========================="
                    if(time_step>19):
                        break
                Tot_Reward[tot_episode/10-1]=Tot_Reward[tot_episode/10-1]+Rn

    Episode_Index = np.arrange(10,201,10)
    np.savetxt('Tot_Reward_EPS_STDDEV_'+str(STDDEV)+'.txt', Tot_Reward)
    plt.plot(Episode_Index, Tot_Reward/100, 'bs')
    plt.xlabel('Num. of Annealing Episodes')
    plt.ylabel('Average Reward over 100 Run')
    plt.savefig('Avg_Reward_EPS_STDDEV_'+str(STDDEV)+'.png')

sess.close()

