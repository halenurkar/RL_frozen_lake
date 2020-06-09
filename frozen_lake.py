import gym
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make('FrozenLake-v0').env

q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.8
gama = 0.95
epsilon = 0.1

episode_number = 75000

reward_list = []

for i in range(1, episode_number):
    #initialize q table
    state = env.reset()
    reward_count=0
    
   
    
    while True:
        #choose an action
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        #perform action
        next_state, reward, done, _= env.step(action)
            
        #q learning fanction
        old_value = q_table[state, action]
        next_value = np.max(q_table[next_state])
        new_value = (1-alpha)*old_value + alpha*(reward + gama*next_value)
            
        #q_table update
        q_table[state, action] = new_value
            
        #state update
        state = next_state
            
        reward_count += reward
        
        if done:
            break
        
   
    reward_list.append(reward_count)
    print("episode: {}, reward: {}".format(i, reward_count))
        
plt.plot(reward_list)          
        
        
        
        
        
        
        
        