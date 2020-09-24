
import matplotlib.pyplot as plt
import numpy as np 
import time
import pickle

def plot_score():
        episode_score = pickle.load(open('./data/dqn_Riverraid_mean_scores.pickle', 'rb'))
        fig = plt.figure()
        episode_score = np. array (episode_score)
        five_episode_score = episode_score [ : , 1]
        frames = episode_score [19: , 0]
        max_100epi_score =[]
        average_100epi_score = []
        episode = len  (episode_score)
        j = 19
        while  (j < episode) :
              max_100epi_score.append (np.max(five_episode_score[j-19:j ]) ) 
              average_100epi_score.append(np.mean(five_episode_score[j-19:j ]) )
              j += 1       
        plt.plot (frames, max_100epi_score)
        plt.plot (frames, average_100epi_score)
        plt.legend (["Max score for 100 episodes","Average score for 100 episodes"])
        plt.xlabel('Time steps (x 1)')
        plt.ylabel(' Score ')
        fig.savefig('Riverraid Score .eps',dpi=600,format='eps')
        fig.savefig('Riverraid Score .png')
        print("Save Score Successfully")
        #plt.pause()
if __name__ == "__main__":
        plot_score()

        