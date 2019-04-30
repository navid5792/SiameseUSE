# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import matplotlib.pyplot as plt
import json
import argparse


# input:
#  alignment matrix - numpy array
#  shape (target tokens + eos, number of hidden source states = source tokens +eos)
# one line correpsonds to one decoding step producing one target token
# each line has the attention model weights corresponding to that decoding step
# each float on a line is the attention model weight for a corresponding source state.
# plot: a heat map of the alignment matrix
# x axis are the source tokens (alignment is to source hidden state that roughly corresponds to a source token)
# y axis are the target tokens

# http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
def plot_head_map(mma, target_labels, source_labels):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(numpy.arange(mma.shape[1]) + 0.5, minor=False)
    ax.set_yticks(numpy.arange(mma.shape[0]) + 0.5, minor=False)

    # without this I get some extra columns rows
    # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
    ax.set_xlim(0, int(mma.shape[1]))
    ax.set_ylim(0, int(mma.shape[0]))

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # source words -> column labels
    ax.set_xticklabels(source_labels, minor=False)
    # target words -> row labels
    ax.set_yticklabels(target_labels, minor=False)

    plt.xticks(rotation=45)

    # plt.tight_layout()
    plt.show()


# column labels -> target words
# row labels -> source words

def read_alignment_matrix(f):
    header = f.readline().strip().split('|||')
    if header[0] == '':
        return None, None, None, None
    sid = int(header[0].strip())
    # number of tokens in source and translation +1 for eos
    src_count, trg_count = map(int, header[-1].split())
    # source words
    source_labels = header[3].decode('UTF-8').split()
    # source_labels.append('</s>')
    # target words
    target_labels = header[1].decode('UTF-8').split()
    target_labels.append('</s>')

    mm = []
    for r in range(trg_count):
        alignment = map(float, f.readline().strip().split())
        mm.append(alignment)
    mma = numpy.array(mm)
    return sid, mma, target_labels, source_labels
import numpy as np
a = [[0.0277, 0.0196, 0.2465, 0.1027, 0.2121, 0.0612, 0.3023, 0.0280, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0116, 0.0068, 0.2672, 0.0739, 0.2134, 0.0353, 0.3799, 0.0118, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0151, 0.0092, 0.2623, 0.0801, 0.2111, 0.0412, 0.3663, 0.0146, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0067, 0.0036, 0.2766, 0.0595, 0.2022, 0.0251, 0.4196, 0.0067, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0067, 0.0036, 0.2759, 0.0605, 0.1994, 0.0250, 0.4221, 0.0068, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0057, 0.0030, 0.2716, 0.0545, 0.1987, 0.0229, 0.4377, 0.0060, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0062, 0.0031, 0.2685, 0.0562, 0.2019, 0.0237, 0.4340, 0.0064, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0095, 0.0051, 0.2681, 0.0678, 0.2057, 0.0311, 0.4030, 0.0097, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0111, 0.0059, 0.2591, 0.0713, 0.2141, 0.0351, 0.3919, 0.0114, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0101, 0.0055, 0.2676, 0.0717, 0.2093, 0.0333, 0.3922, 0.0104, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0117, 0.0065, 0.2630, 0.0733, 0.2148, 0.0357, 0.3833, 0.0117, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0216, 0.0138, 0.2455, 0.0916, 0.2099, 0.0536, 0.3420, 0.0220, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0320, 0.0226, 0.2331, 0.1052, 0.2073, 0.0681, 0.2990, 0.0327, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0185, 0.0118, 0.2551, 0.0873, 0.2098, 0.0488, 0.3499, 0.0187, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0185, 0.0118, 0.2551, 0.0873, 0.2098, 0.0488, 0.3499, 0.0187, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0185, 0.0118, 0.2551, 0.0873, 0.2098, 0.0488, 0.3499, 0.0187, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0185, 0.0118, 0.2551, 0.0873, 0.2098, 0.0488, 0.3499, 0.0187, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0185, 0.0118, 0.2551, 0.0873, 0.2098, 0.0488, 0.3499, 0.0187, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0185, 0.0118, 0.2551, 0.0873, 0.2098, 0.0488, 0.3499, 0.0187, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0185, 0.0118, 0.2551, 0.0873, 0.2098, 0.0488, 0.3499, 0.0187, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000],
        [0.0185, 0.0118, 0.2551, 0.0873, 0.2098, 0.0488, 0.3499, 0.0187, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000]]
a = np.array(a)[0:13, 0:13]
x = '<s>', 'The', 'equipment', 'in', 'front', 'of', 'the', 'blond', 'dancing', 'girl', 'is', 'sound', '</s>' 
y= ['<s>', 'A', 'girl', 'in', 'white', 'is', 'dancing', '</s>']
print(a.shape)
#y = ['<s>', 'Four', 'girls', 'are', 'doing', 'backbends', 'and', 'playing', 'outdoors', '</s>']





def read_plot_alignment_matrices(f, start=0):
    global a
    global x
    global y
    attentions = json.load(f, encoding="utf-8")

    for idx, att in attentions.items():

        if int(idx) < int(start): continue
        source_labels = att["source"].split() + ["SEQUENCE_END"]
        target_labels = att["translation"].split()
        att_list = att["attentions"]
        assert att_list[0]["type"] == "simple", "Do not use this tool for multihead attention."
        mma = numpy.array(att_list[0]["value"])
        print(mma.shape, len(target_labels), len(source_labels), target_labels, source_labels)
        #asasas
        if mma.shape[0] == len(target_labels) + 1:
            target_labels += ["SEQUENCE_END"]

        print(mma)
        plot_head_map(a, x, y)
        asas
        plot_head_map(mma, target_labels, source_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=argparse.FileType("rb"),
                        default="trans.att",
                        metavar='PATH',
                        help="Input file (default: standard input)")
    parser.add_argument('--start', type=int, default=0)

    args = parser.parse_args()

    read_plot_alignment_matrices(args.input, args.start)
