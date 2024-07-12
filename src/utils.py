import matplotlib.pyplot as plt
import numpy as np
import os


def create_graph(y1:np.ndarray, y2:np.ndarray, title:str, y_label:str):
    x = np.arange(len(y1))
    
    plt.plot(x, y1, color='yellow', label = 'wave-gpt')
    plt.plot(x, y1, color='orange', label = 'gpt')

    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel(y_label)

    plt.legend()
    os.makedirs('graphs', exist_ok=True)
    plt.savefig(f'graphs/{title}.png')

    