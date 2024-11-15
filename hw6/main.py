import os
import sys
import numpy as np
import torch
import gym
import argparse
import gym.envs.box2d.car_racing as cr

from train import train
from demonstration import record_demonstration
from evaluate import evaluate

def calculate_score_for_leaderboard(trained_network_file):
    """
    Evaluate the performance of the network. This is the function to be used for
    the final ranking on the course-wide leader-board, only with a different set
    of seeds. Better not change it.
    """
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    infer_action = torch.load(trained_network_file, map_location=device)
    
    infer_action.eval()
    env = cr.CarRacing(render_mode="human")


    seeds = [22597174, 68545857, 75568192, 91140053, 86018367,
            49636746, 66759182, 91294619, 84274995, 31531469]
    # seeds = [81232882, 65020368, 13601687, 57953090, 37989439,
    #          84877679, 5664502, 98291373, 28454024, 87598965]

    total_reward = 0

    for episode in range(10):
        env.reset(seed=seeds[episode])
        observation = env.reset()[0]

        reward_per_episode = 0
        for t in range(600):
            env.render()
            action_scores = infer_action(torch.Tensor(
                np.ascontiguousarray(observation[None])).to(device))

            steer, gas, brake = infer_action.scores_to_action(action_scores)
            observation, reward, done, info,_ = env.step([steer, gas, brake])
            reward_per_episode += reward

        print('episode %d \t reward %f' % (episode, reward_per_episode))
        total_reward += np.clip(reward_per_episode, 0, np.infty)

    print('---------------------------')
    print(' total score: %f' % (total_reward / 10))
    print('---------------------------')

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--train",
        action="store_true",
    )
    main_parser.add_argument(
        "--test",
        action="store_true",
    )
    main_parser.add_argument(
        "--score",
        action="store_true",
    )
    main_parser.add_argument(
        "--teach",
        action="store_true",
    )
    main_parser.add_argument(
        "--agent_load_path",
        type=str,
        default="data/agent.pt",
        help="Path to the .pt file of the trained agent."
    )
    main_parser.add_argument(
        "--agent_save_path",
        type=str,
        default="data/agent.pt",
        help="Save path of the trained model."
    )
    main_parser.add_argument(
        "--training_data_path",
        type=str,
        default="data/teacher",
        help="Save path of the trained model."
    )

    args = main_parser.parse_args()

    if args.teach:
        print('Teach: You can collect training data now.')
        record_demonstration(args.training_data_path)
    elif args.train:
        print('Train: Training your network with the collected data.')
        train(args.training_data_path, args.agent_save_path)
    elif args.test:
        print('Test: Your trained model will be tested now.')
        evaluate(args.agent_load_path, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    elif args.score:
        calculate_score_for_leaderboard(args.agent_load_path)