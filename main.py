from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from typing import Dict
import torch
import matplotlib.pyplot as plt

from model import QNetwork

import PARAMETERS as PARAM
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from ddqn_agent_prioritized_experience import DDQNAgentPrioExpReplay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_MODE = False


def main():
    env = UnityEnvironment(file_name="/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print_env_state(env=env, brain_name=brain_name, brain=brain)

    if not TRAIN_MODE:
        model = QNetwork(PARAM.STATE_SIZE, PARAM.ACTION_SIZE, 0).to(device)
        model.load_state_dict(torch.load("/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p1_navigation/banana_project/saved_checkpoints/ddq_checkpoint_800.pth"))
        agent = DQNAgent(state_size=PARAM.STATE_SIZE, action_size=PARAM.ACTION_SIZE, seed=0)
        agent.qnetwork_target = model
        agent.qnetwork_local = model
    else:
        agent = DQNAgent(state_size=PARAM.STATE_SIZE, action_size=PARAM.ACTION_SIZE, seed=0)
        #agent = DDQNAgent(state_size=PARAM.STATE_SIZE, action_size=PARAM.ACTION_SIZE, seed=0)
        #agent = DDQNAgentPrioExpReplay(state_size=PARAM.STATE_SIZE, action_size=PARAM.ACTION_SIZE, seed=0)

    # TODO maybe use hyperparameter class for setting rest of run method
    scores = run_agent(agent=agent,
                       env=env,
                       brain_name=brain_name)
    plot_scores(scores=scores)


def print_env_state(env: UnityEnvironment, brain_name: str, brain: Dict):
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)


def run_agent(agent, env: UnityEnvironment, brain_name: str):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = PARAM.EPS_START  # initialize epsilon
    for i_episode in range(1, PARAM.N_EPISODES + 1):

        env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        for t in range(PARAM.MAX_T):
            if TRAIN_MODE:
                action = agent.act(state, eps)
            else:
                action = agent.act(state, 0)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            if TRAIN_MODE:
                agent.step(state, action, reward, next_state, done)

            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(PARAM.EPS_END, PARAM.EPS_DECAY * eps)  # decrease epsilon
        #print('\rEpisode {}\tAverage Score: {:.2f} \tepsilon: {:.2f} \tbeta: {:.2f}'.format(i_episode, np.mean(scores_window), eps, agent.memory.beta), end="")
        print('\rEpisode {}\tAverage Score: {:.2f} \tepsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps), end="")
        if i_episode % 100 == 0 and TRAIN_MODE:
            #print('\rEpisode {}\tAverage Score: {:.2f} \tepsilon: {:.2f} \tbeta: {:.2f}'.format(i_episode, np.mean(scores_window), eps, agent.memory.beta))
            print('\nEpisode {}\tAverage Score: {:.2f} \tepsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps), end="")
            torch.save(agent.qnetwork_local.state_dict(), f'ddq_checkpoint_{i_episode}.pth')
            plot_scores(scores_window)
        if np.mean(scores_window) >= 13.0 and TRAIN_MODE:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} '.format(i_episode - 100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), f'ddq_checkpoint_{np.round(np.mean(scores_window),2)}.pth')
            plot_scores(scores_window)
    return scores


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'score_plot_{scores}.jpg')
    plt.show()


if __name__ == "__main__":
    main()

