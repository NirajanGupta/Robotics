import gym
from packages.Lander_on_Mars import Agents
#from utils import plot_learning_curve
import matplotlib.pyplot as plt 
import numpy as np


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    # 1) Save it
    plt.savefig(filename)
    # 2) Show it
    plt.show()


if __name__ == '__main__':
    env  = gym.make('LunarLander-v2')
    agent = Agents(gamma=0.99,epsilon=1.0, batch_size=64,n_actions=4,
                  eps_end=0.01, input_dims=[8], lr=0.003)
    scores, eps_history = [],[]
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation, info = env.reset()
        while not done:
            action  = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)  # now 5 outputs
            done = terminated or truncated                                       # collapse into a single “done”

            score += reward
            agent.store_transition(observation, action,reward,observation_,done)

            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon % .2f' % agent.epsilon)
    x = [i+1 for i in range(n_games)]
    filename = 'Lander_on_Mars.png'
    plot_learning_curve(x, scores, eps_history,filename)







