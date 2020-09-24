import matplotlib.pyplot as plt
import numpy as np 
import time
import pickle

def plot_score():
        episode_score = pickle.load(open('./dqn_LunarLander_scores.pickle', 'rb'))
        fig = plt.figure()
        episode_score = np. array (episode_score)
        ten_episode_score = episode_score [ : , 1]
        frames = episode_score [9: , 0]
        max_100epi_score =[]
        average_100epi_score = []
        episode = len  (episode_score)
        j = 0
        while  (j +10<= episode) :
              max_100epi_score.append (np.max(ten_episode_score[j:j + 9]) ) 
              average_100epi_score.append(np.mean(ten_episode_score[j:j + 9]) )
              j += 1 
        plt.plot (frames, max_100epi_score)
        plt.plot (frames, average_100epi_score)
        plt.legend (["Max score for 100 episodes","Average score for 100 episodes"])
        plt.xlabel('Time steps (x 1)')
        plt.ylabel(' Score ')
        fig.savefig('Lunar Lander Score.eps',dpi=600,format='eps')
        fig.savefig('Lunar Lander Score.png')

        episode_score = pickle.load(open('./dqn_LunarLander_loss.pickle', 'rb'))
        fig = plt.figure()
        episode_score = np. array (episode_score)
        ten_episode_score = episode_score [ : , 1]
        frames = episode_score [9: , 0]
        max_100epi_score =[]
        average_100epi_score = []
        episode = len  (episode_score)
        j = 0
        while  (j +10<= episode) :
              max_100epi_score.append (np.max(ten_episode_score[j:j + 9]) ) 
              average_100epi_score.append(np.mean(ten_episode_score[j:j + 9]) )
              j += 1 
        plt.plot (frames, max_100epi_score)
        plt.plot (frames, average_100epi_score)
        plt.legend (["Max loss for 100 episodes","Average loss for 100 episodes"])
        plt.xlabel('Time steps (x 1)')
        plt.ylabel(' Score ')
        #fig.savefig('Lunar Lander Loss.eps',dpi=600,format='eps')
        fig.savefig('Lunar Lander loss.png')



        print("Save Score Successfully")
        #plt.pause()
if __name__ == "__main__":
        plot_score()
