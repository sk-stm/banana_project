import os
import shutil

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from typing import Dict, List
import torch
import matplotlib.pyplot as plt
import datetime

from model import QNetwork

import PARAMETERS as PARAM
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from ddqn_agent_prioritized_experience import DDQNAgentPrioExpReplay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_MODE = False
MODEL_TO_LOAD = "DQN/best_model_overall/dq_checkpoint_15.39.pth"


def main():
    """
    Main method runs the whole experiment.
    """
    env = UnityEnvironment(file_name="../Banana_Linux/Banana.x86")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print_env_state(env=env, brain_name=brain_name, brain=brain)

    agent = DQNAgent(state_size=PARAM.STATE_SIZE, action_size=PARAM.ACTION_SIZE, seed=0)
    # agent = DDQNAgent(state_size=PARAM.STATE_SIZE, action_size=PARAM.ACTION_SIZE, seed=0)
    # agent = DDQNAgentPrioExpReplay(state_size=PARAM.STATE_SIZE, action_size=PARAM.ACTION_SIZE, seed=0)

    if not TRAIN_MODE:
        load_model_into_agent(agent)

    scores = run_agent(agent=agent, env=env, brain_name=brain_name)
    save_score_plot(scores=scores)


def load_model_into_agent(agent):
    """
    Loads a pretrained network into the created agent.
    """
    model = QNetwork(PARAM.STATE_SIZE, PARAM.ACTION_SIZE, 0).to(device)
    model.load_state_dict(torch.load(MODEL_TO_LOAD))
    agent.qnetwork_target = model
    agent.qnetwork_local = model


def print_env_state(env: UnityEnvironment, brain_name: str, brain: Dict):
    """
    Print the environment properties.

    :param env:         environment to be printed
    :param brain_name:  Name of the brain of the environment
    :param brain:       Brain of the environment
    """
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
    """
    Runs the agent in the environment.
    When TRAIN_MODE == True, the agent is trained and acts according to epsilon-greedy policy, else the agent
    acts according to full greedy policy.
    :param agent:       Agent to run
    :param env:         Environment to run the agent in (must be UnityEnvironment)
    :param brain_name:  Name of the environment brain
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = PARAM.EPS_START  # initialize epsilon
    score_max = 0
    for i_episode in range(1, PARAM.N_EPISODES + 1):

        # reset environment
        env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]
        # get first state
        state = env_info.vector_observations[0]
        score = 0

        # iterate over MAX_T many steps
        for t in range(PARAM.MAX_T):

            if TRAIN_MODE:
                # use epsilon.greedy policy when training
                action = agent.act(state, eps)
            else:
                # use fully greedy policy when testing
                action = agent.act(state, 0)

            # send action to the environment and receive next_state, reward and if the episode is done
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            # when in training mode, let agent learn
            if TRAIN_MODE:
                agent.step(state, action, reward, next_state, done)

            # keep track of the score
            score += reward

            # use next state for next iteration
            state = next_state

            if done:
                break

        # keep track of scores
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        # update epsilon
        eps = max(PARAM.EPS_END, PARAM.EPS_DECAY * eps)

        # print according to replay buffer type
        #print('\rEpisode {}\tAverage Score: {:.2f} \tepsilon: {:.2f} \tbeta: {:.2f}'.format(i_episode, np.mean(scores_window), eps, agent.memory.beta), end="")
        print('\rEpisode {}\tAverage Score: {:.2f} \tepsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps), end="")

        # plot and save the agent if necessary
        plot_and_save_agent(agent, eps, i_episode, score_max, scores, scores_window)

        score_max = np.mean(scores_window)

    return scores


def plot_and_save_agent(agent, eps, i_episode, score_max, scores, scores_window):
    """
    Plots and saves the agent each 100th episode.
    Saves the agent, the current scores, the episode number, the trained parameters of the NN model and the hyper
    parameters of the agent to a folder with the current date and time if the mean average of the last 100 scores
    are > 13 and if a new maximum average was reached.

    :param agent:           agent to saved
    :param eps:             current value of epsilon
    :param i_episode:       number of current episode
    :param score_max:       max_score reached by the agent so far
    :param scores:          all scores of the agent reached so far
    :param scores_window:   mean of the last 100 scores
    """
    if i_episode % 100 == 0 and TRAIN_MODE:
        # print('\rEpisode {}\tAverage Score: {:.2f} \tepsilon: {:.2f} \tbeta: {:.2f}'.format(i_episode, np.mean(scores_window), eps, agent.memory.beta))
        print('\nEpisode {}\tAverage Score: {:.2f} \tepsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps),
              end="")
        save_score_plot(scores, i_episode)
    if np.mean(scores_window) >= 13.0 and np.mean(scores_window) > score_max and TRAIN_MODE:
        score_max = np.mean(scores_window)
        save_current_agent(agent, score_max, scores, i_episode)
        # TODO save replay buffer parameters as well if prioritized replay buffer was used
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} '.format(i_episode - 100,
                                                                                      np.mean(scores_window)))


def save_current_agent(agent, score_max, scores, i_episode):
    """
    Saves the current agent.

    :param agent:       agent to saved
    :param score_max:   max_score reached by the agent so far
    :param scores:      all scores of the agent reached so far
    :param i_episode:   number of current episode
    """
    new_folder_path = create_folder_structure_according_to_agent(agent)

    os.makedirs(new_folder_path, exist_ok=True)
    torch.save(agent.qnetwork_local.state_dict(),
               os.path.join(new_folder_path, f'checkpoint_{np.round(score_max, 2)}.pth'))
    save_score_plot(scores, i_episode, path=new_folder_path)
    shutil.copyfile("PARAMETERS.py", os.path.join(new_folder_path, "PARAMETERS.py"))


def create_folder_structure_according_to_agent(agent):
    """
    Creates a folder structure to store the current experiment according to the type of agent that was run and the
    current date and time.

    :param agent: Agent to be stored
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    if type(agent) == DQNAgent:
        new_folder_path = os.path.join('DQN', f'{date_str}')
    elif type(agent) == DDQNAgent:
        new_folder_path = os.path.join('DDQN', f'{date_str}')
    elif type(agent) == DDQNAgentPrioExpReplay:
        new_folder_path = os.path.join('DDQN_prio_replay', f'{date_str}')
    else:
        raise NotImplementedError()
    return new_folder_path


def save_score_plot(scores: List, i_episode: int, path: str = ""):
    """
    Saves a plot of numbers to a folder path. The The i_episode number is added to the name of the file.

    :param scores:      All numbers to store
    :param i_episode:   Current number of episodes
    :param path:        Path to the folder to store the plot to.
    """
    fig = plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(os.path.join(path, f'score_plot_{i_episode}.jpg'))


if __name__ == "__main__":
    main()

