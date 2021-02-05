from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from typing import Dict
import torch
import matplotlib.pyplot as plt

from agent import Agent


def main():
    env = UnityEnvironment(file_name="/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print_env_state(env=env, brain_name=brain_name, brain=brain)
    # TODO implement loading of model

    agent = Agent(state_size=37, action_size=4, seed=0)
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


def run_agent(agent: Agent, env: UnityEnvironment, brain_name: str, n_episodes: int=2000, max_t: int=1000, eps_start: float=1.0, eps_end: float=0.01, eps_decay: float=0.995):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), f'ddq_checkpoint_{i_episode}.pth')
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'ddq_checkpoint.pth')
    return scores


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == "__main__":
    main()

