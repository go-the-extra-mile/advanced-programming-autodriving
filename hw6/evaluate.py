import os
import sys
import numpy as np
import torch
import gym.envs.box2d.car_racing as cr

def evaluate(trained_network_file, device):
    
    env = cr.CarRacing(render_mode="human")
    screen_to_action = torch.load(trained_network_file, map_location=device)
    screen_to_action.eval()

    scenario_n = 10
    time_limit = 500

    for episode in range(scenario_n):

        observation = env.reset()[0]
        reward_per_episode = 0 
        
        for t in range(time_limit):

            env.render()
            action_scores = screen_to_action(torch.Tensor(np.ascontiguousarray(observation[None])).to(device))
            steer, gas, brake = screen_to_action.scores_to_action(action_scores)
            observation, reward, done, info, _ = env.step([steer, gas, brake])
            reward_per_episode += reward

        print('episode %d \t reward %f' % (episode, reward_per_episode))

    env.close()
