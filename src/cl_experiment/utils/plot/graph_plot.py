import os, sys
import re
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

'''
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
'''

'''
python3 .../graph_plot.py path1 path2 
'''

def parse_metrics(json_path):
    print(json_path)
    with open(json_path, 'r') as f_p:
        exp_data = json.load(f_p)
    
    # split by task
    grouped_results = {}
    t_d = None
    for desc, val in exp_data.items():
        f = re.search('_T(\d+)', desc)
        task_desc = desc[f.start()+2:f.end()]
        print(task_desc)
        if t_d == None or t_d != task_desc:
            if t_d != None: grouped_results.update({t_d : t_a})
            t_d = task_desc
            t_a = []
        else:
            t_a.append(val)
    
    print(grouped_results)
    return grouped_results
    
def plot(a, b):
    colors = [color for color in mcolors.TABLEAU_COLORS]

    fig, ax = plt.subplots()
    # use a gray background
    #ax.set_axisbelow(True)
    # draw solid white grid lines
    
    # hide top and right ticks
    # ax.xaxis.tick_bottom()
    # ax.yaxis.tick_left()
    
    #ax.set_ylim(-300., +400.)
    #ax.yaxis.set_major_locator(plt.MaxNLocator(6)) # set number of y ticks
    #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    #ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.grid(axis='y', linestyle='solid', color='w')
    ax.grid(axis='x', linestyle='solid', color='w')
    #plt.title('GMM density estimation for E-MNIST 20-1', pad=8)
    plt.xlabel(xlabel='epoch', labelpad=4, fontsize=24)
    plt.ylabel(ylabel='train duration', labelpad=4, fontsize=24)

    
    ax.tick_params(axis='y', which='major', colors='white', direction='in', labelsize=12)
    for tick in ax.get_xticklabels(): tick.set_color('black')
    for tick in ax.get_yticklabels(): tick.set_color('black')
    
    dgr_dur = []; ar_dur = []

    max_values = 199
    for i, ((a_k, a_v), (b_k, b_v)) in enumerate(zip(a.items(), b.items())):

        if i == 0:
            for j in range(0,20):
                print(b_v[j])
                b_v[j] = b_v[20]
        print(a_k, b_k, ":", len(a_v), len(b_v))
        ar_dur.append(a_v)
        pad_arr = [a_v[-1]] * (max_values - len(a_v))
        ar_dur[i].extend(pad_arr)
        dgr_dur.append(b_v[0:len(b_v)-50])

        print(len(ar_dur[i]), len(dgr_dur[i]))

    print(len(ar_dur), len(dgr_dur))

    print(dgr_dur[-1])

    _0 = np.arange(0, 199, 1)
    _1 = np.arange(200, 399, 1)
    _2 = np.arange(400, 599, 1)
    _3 = np.arange(600, 799, 1)
    _4 = np.arange(800, 999, 1)
    _5 = np.arange(1000, 1199, 1)

    x_list = [_0, _1, _2, _3, _4, _5]

    for i, (ar_task, dgr_task, x_range) in enumerate(zip(ar_dur, dgr_dur, x_list)):
        #print(x_range.shape)
        #print(dgr_task)
        if i < 4:
            plt.plot(x_range, ar_task, color=colors[0], linewidth=5.0) 
            plt.plot(x_range, dgr_task, color=colors[1], linewidth=5.0)
        else:
            plt.plot(x_range, ar_task, label='AR', color=colors[0], linewidth=5.0)
            plt.plot(x_range, dgr_task, label='DGR (b.)', color=colors[1], linewidth=5.0)
    ax.set_xlim(-50, +1200)
    ax.set_ylim(+2500, +20000)
    plt.vlines(x=[200, 400, 600, 800], ymin=0, ymax=[16000, 18000, 18000, 18000], linestyle='dashed', label='task switch', colors=['grey'])
    ax.yaxis.set_major_locator(plt.MaxNLocator(4)) 

    """
    for i in range(0, 5):
        plt.plot(task_labels[i], logs[i], alpha=1.0, ls='--', lw=2, label=f'T{i+2}', color=color[i])
        #x_1=[i, i+1]
        #y_1=logs[i][0:2]
        #plt.fill_between(x=x_1, y1=y_1, y2=-300., alpha=0.25, color=color[i])
    """
    ax.legend(frameon=False, loc='center right', ncol=1)
    #plt.show()
    fig.tight_layout()
    fig.subplots_adjust(bottom=.15, left=.15)
    plt.savefig('/home/ak/Desktop/graph_plot.png')


if __name__ == '__main__':
    json_path_a, json_path_b = sys.argv[1], sys.argv[2]
    a = parse_metrics(json_path_a)
    b = parse_metrics(json_path_b)
    with plt.style.context('ggplot'):         # 'bmh'
        plot(a, b)
