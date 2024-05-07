import matplotlib.pyplot as plt
import numpy as np
from Plots.Plot_Helper import *

import scipy.cluster.hierarchy as sch
import pandas as pd

#execute Experiment_char.py and romove comments from:
### from Plots.DIST import *
### train_and_render_dist_plot(net, input_steps=60000)

def get_classes(net):
    neurons_in_clusters = np.zeros(net.inp_neurons.height)
    neuron_indices_in_clusters = {i: [] for i in range(net.inp_neurons.height)}

    W = net.exc_neurons1.afferent_synapses['ES'][0].W

    for i in range(net.exc_neurons1.size):
        afferents = W[:, i]
        afferents = afferents.reshape(net.inp_neurons.height, net.inp_neurons.width)
        #print(afferents.shape)
        s = np.sum(afferents, axis=1)
        ind = np.argmax(s)
        neurons_in_clusters[ind] += 1
        neuron_indices_in_clusters[ind].append(i)

    #print(np.sum(neurons_in_clusters))

    return neuron_indices_in_clusters


def get_sub_clusters(net, main_cluster_indices, sensitivity):
    W = net.exc_neurons1.afferent_synapses['EE'][0].W
    #print(len(main_cluster_indices), main_cluster_indices)
    df = pd.DataFrame(W[:, main_cluster_indices])  # .T

    corrMatrix = df.corr()
    pairwise_distances = sch.distance.pdist(corrMatrix)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / sensitivity
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')

    indices, sizes = np.unique(idx_to_cluster_array, return_counts=True)

    print(sizes)

    #result = sizes
    #print(len(idx_to_cluster_array), idx_to_cluster_array)
    return sizes

def plot_data(d, t):

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    fig.set_figwidth(6)
    fig.set_figheight(6)


    alphabet = np.array(['_', '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'x', 'y'])
    weighted_alphabet = np.array([3.98076923, 1.32692308, 0.88461538, 0.44230769, 0.88461538,
                              0.44230769, 2.65384615, 0.44230769, 0.44230769, 2.21153846,
                              0.44230769, 0.88461538, 0.44230769, 0.44230769, 1.32692308,
                              0.88461538, 0.44230769, 0.44230769, 1.32692308, 0.88461538,
                              0.88461538, 0.44230769, 0.44230769])

    sorted_alphabet = alphabet[(-weighted_alphabet).argsort()]

    alphabet[0] = ' '
    txt = [t.count(a) for a in alphabet]

    nnn = np.array(d, dtype=object)[(-weighted_alphabet).argsort()]#np.array(list(reversed(sorted(d))))
    #c = np.array([9., 3., 2., 1., 2., 1., 6., 1., 1., 5., 1., 2., 1., 1., 3., 2., 1., 1., 3., 2., 2., 1., 1.])[(-weighted_alphabet).argsort()]#np.array(list(reversed(sorted([9., 3., 2., 1., 2., 1., 6., 1., 1., 5., 1., 2., 1., 1., 3., 2., 1., 1., 3., 2., 2., 1., 1.]))))
    ccc = np.array(txt)[(-weighted_alphabet).argsort()]

    yyy = np.array(range(len(nnn)))

    max_nt = 0

    for i, (c, y, n) in enumerate(zip(ccc, yyy, nnn)):
        n_total = np.sum(n)

        if n_total > max_nt:
            max_nt = n_total

    for i, (c, y, n) in enumerate(zip(ccc, yyy, nnn)):

        color_counter = i % 2
        c_counter = 0
        for ns in np.flip(np.sort(n)):
            color_counter += 1
            ax.barh(y-0.2, ns, left=c_counter, height=0.4, color=(12/255, 5/255, 177/255, 0.8+color_counter % 2*0.2)) #colorE
            c_counter += ns

        ax.barh(y+0.2, c/np.max(ccc)*np.max(max_nt), height=0.4, color=color1)

        #p = weighted_alphabet[(-weighted_alphabet).argsort()]/np.max(weighted_alphabet)
        #p = p * np.max(n_total)
        #ax.barh(y+0.2, np.ones(len(n_total)), left=p-1, height=0.4, color=(0, 0.5, 0, 1))

    ax.set_xlim([0, np.max(max_nt)])
    ax2.set_xlim([0, np.max(ccc)])

    #ax.set_ylabel("spike possibility")
    #ax.set_xlabel("voltage")
    #ax[0].set_yticks([])
    ax.set_yticks(yyy, sorted_alphabet)
    ax2.set_yticks([])

    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')

    ax.set_xlabel('number of neurons clustered by weights', color=colorE)  # we already handled the x-label with ax1
    ax.tick_params(axis='x', labelcolor=colorE)

    ax2.set_xlabel('number of char in input', color=color1)  # we already handled the x-label with ax1
    ax2.tick_params(axis='x', labelcolor=color1)

    ##ax.spines[['top', 'bottom']].set_visible(False)
    ##ax2.spines[['top', 'bottom']].set_visible(False)

    ax.spines[['left', 'right']].set_visible(False)
    ax2.spines[['left', 'right', 'left']].set_visible(False)

    plt.text(x=-0.4, y=1.1, s='A', size=20, weight='bold')

    fig.tight_layout()
    #plt.savefig('filename'+str(nr)+'.png', dpi=600)
    plt.show()
    plt.close()
    plt.clf()

def train_and_render_dist_plot(net, input_steps):
    net.simulate_iterations(input_steps, 100)

    classes = get_classes(net)
    classes = [get_sub_clusters(net, v, 1.1) for k, v in classes.items()]

    plot_data(classes, net.inp_neurons.TextGenerator.history)



