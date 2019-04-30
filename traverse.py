#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 22:28:52 2018

@author: ahmed
"""

#data = {'rating': 3, 'depth': 1, 'children': [{'depth': 3, 'children': [{'depth': 4, 'word': 'Dozens', 'size': 1}, {'depth': 4, 'children': [{'depth': 5, 'word': 'of', 'size': 1}, {'depth': 5, 'children': [{'depth': 6, 'word': 'government', 'size': 1}, {'depth': 6, 'word': 'proposals', 'size': 1}], 'size': 2}], 'size': 3}], 'size': 4}, {'depth': 3, 'children': [{'depth': 4, 'children': [{'depth': 4, 'word': 'arise', 'size': 1}, {'depth': 4, 'children': [{'depth': 4, 'children': [{'depth': 5, 'word': ',', 'size': 1}, {'depth': 5, 'word': 'but', 'size': 1}], 'size': 2}, {'depth': 4, 'children': [{'depth': 4, 'word': 'never', 'size': 1}, {'depth': 4, 'children': [{'depth': 5, 'word': 'get', 'size': 1}, {'depth': 5, 'children': [{'depth': 5, 'word': 'off', 'size': 1}, {'depth': 5, 'children': [{'depth': 6, 'word': 'the', 'size': 1}, {'depth': 6, 'word': 'ground', 'size': 1}], 'size': 2}], 'size': 3}], 'size': 4}], 'size': 5}], 'size': 7}], 'size': 8}, {'depth': 4, 'word': '.', 'size': 1}], 'size': 9}], 'size': 13}

#def traverse(data, side, node_depth):
#    if 'children' in data:
#        print("node depth:\t", data['depth'])
#        print("going left")
#        traverse(data['children'][0], side = "left", node_depth = data['depth'])
#        print("left done\n")
#        print("node depth:\t", data['depth'])
#        print("going right")
#        traverse(data['children'][1], side = "right", node_depth = data['depth'])
#        print("right done\n")
#    else:
#        print(data['word'].lower(), end = '\t')
#        print("depth:\t", data['depth'], "\tside: ", side, end = '\n')
#    
#traverse(data, side = None, node_depth = None)

#data = {'rating': 3, 'depth': 1, 'children': [{'depth': 3, 'children': [{'depth': 4, 'word': 'Dozens', 'size': 1}, {'depth': 4, 'children': [{'depth': 5, 'word': 'of', 'size': 1}, {'depth': 5, 'children': [{'depth': 6, 'word': 'government', 'size': 1}, {'depth': 6, 'word': 'proposals', 'size': 1}], 'size': 2}], 'size': 3}], 'size': 4}, {'depth': 3, 'children': [{'depth': 4, 'children': [{'depth': 4, 'word': 'arise', 'size': 1}, {'depth': 4, 'children': [{'depth': 4, 'children': [{'depth': 5, 'word': ',', 'size': 1}, {'depth': 5, 'word': 'but', 'size': 1}], 'size': 2}, {'depth': 4, 'children': [{'depth': 4, 'word': 'never', 'size': 1}, {'depth': 4, 'children': [{'depth': 5, 'word': 'get', 'size': 1}, {'depth': 5, 'children': [{'depth': 5, 'word': 'off', 'size': 1}, {'depth': 5, 'children': [{'depth': 6, 'word': 'the', 'size': 1}, {'depth': 6, 'word': 'ground', 'size': 1}], 'size': 2}], 'size': 3}], 'size': 4}], 'size': 5}], 'size': 7}], 'size': 8}, {'depth': 4, 'word': '.', 'size': 1}], 'size': 9}], 'size': 13}
#
#import numpy as np
#def traverse(data):
#    if 'children' not in data:
#        return np.random.random(2), data['word'].lower()
#    else:
#        idd = 0
#        ids = []
#        word = []
#        vectors = []
#        for i in range(len(data['children'])):
#            x,y = traverse(data['children'][i])
#            word.append(y)
#            vectors.append(x)
#            idd += 1
#            ids.append(idd)
#        print(word, "\t", ids)
#        return sum(vectors), "demo"
#        
#            
#a,b = traverse(data)
#print(a)


#q.remove('DT')
#q.remove('JJ')
#q.remove('NN')
#q.remove('VBZ')
#q.remove('NN')
#q.remove('JJ')
#q.remove('DT')
#q.remove('IN')




q = ['(',
 'ROOT',
 '(',
 'S',
 '(',
 'NP',
 '(',
 'The',
 ')',
 '(',
 'quick',
 ')',
 '(',
 'brown',
 ')',
 '(',
 'fox',
 ')',
 ')',
 '(',
 'VP',
 '(',
 'jumps',
 ')',
 '(',
 'PP',
 '(',
 'over',
 ')',
 '(',
 'NP',
 '(',
 'the',
 ')',
 '(',
 'JJ',
 'lazy',
 ')',
 '(',
 'dog',
 ')',
 ')',
 ')',
 ')',
 ')',
 ')']


node = dict()
brackets = []
phrase = []
for i in range(len(q)):
    if q[i] == "(":
        brackets.append("(")
        continue
    if q[i] == ")":
        brackets.pop()
        continue

import numpy as np
from nltk import Tree
e = "".join(q)

dic = dict()
dic['i'] = np.array([1,2,3])
dic['eat'] = np.array([10,20,300])
dic['rice'] = np.array([100,200,300])

import nltk       
def traverse_tree(tree):
    if type(tree[0]) != Tree:
        tree[0] = [1,2]
        print("leaf encountered: ", tree[0], end = " \n")
        return  np.array([1,1]), tree[0]
    else:
        idd = 0
        ids = []
        word = []
        vectors = []
        summ = 0
        for subtree in tree:
            vec, w = traverse_tree(subtree)
#            print(w)
            word.append(w)
            summ += vec
            idd += 1
            ids.append(idd)
        print(summ, word, ids)
        return summ, word
#t = Tree.fromstring('(ROOT\n  (S\n    (NP (PRP I))\n    (VP (VBP eat)\n      (NP (NN rice)))))')
pos ="CC CD DT EX FW IN JJ JJR JJS LS MD NN NNS NNP NNP PDT POS PRP PRP$ RB RBR RBS RP TO UH VB VBD VBG VBN VBP VBZ WDT WP WP WRB".split()
import pickle
#with open("sick_tree_data.pkl", "rb") as f:
#    train, valid, test = pickle.load(f)
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('/home/bob/Downloads/stanford-corenlp-full-2018-10-05/')
t = nlp.parse(train['s1'][0])
#for x in pos:
#    if x in t:
#        print("removing: ", x)
#        t= t.replace(x, "")
t = Tree.fromstring(t)
a= traverse_tree(t)
#print(a)

         
 