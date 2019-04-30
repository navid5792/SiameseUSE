import numpy as np
import string
import pickle
from numpy import random
from nltk import Tree
from stanfordcorenlp import StanfordCoreNLP
import sys
import argparse
import numpy as np
import torch
from data import build_vocab
from copy import deepcopy
from torch.autograd import Variable
import spacy
from data import get_nli, get_batch, build_vocab

table = str.maketrans({key: None for key in string.punctuation})
def get_MSRP_data():
    with open("dataset/MS_train.txt", "r") as f:
        data = f.readlines()
    train = dict()
    test = dict()
    valid = dict ()
    
    s1_temp = []
    s2_temp = []
    label_temp = []
    for i in range(1,4001):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[3].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[4].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(x[0]))

    train['s1'] = s1_temp
    train['s2'] = s2_temp
    train['label'] = np.array(label_temp)

    s1_temp = []
    s2_temp = []
    label_temp = []
    for i in range(4001,len(data)):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[3].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[4].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(x[0]))

    test['s1'] = s1_temp
    test['s2'] = s2_temp
    test['label'] = np.array(label_temp)

    valid['s1'] = s1_temp
    valid['s2'] = s2_temp
    valid['label'] = np.array(label_temp)
    return train, valid, test

def get_SICK_binary_data():
    with open("dataset/SICK_en-en_train.txt", "r") as f:
        data = f.readlines()
    train = dict()
    test = dict()
    valid = dict ()
    
    s1_temp = []
    s2_temp = []
    label_temp = []
    for i in range(1, 3364):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[3].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[4].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(x[0]))

    train['s1'] = s1_temp
    train['s2'] = s2_temp
    train['label'] = np.array(label_temp)

    s1_temp = []
    s2_temp = []
    label_temp = []
    for i in range(3364, 3364+366):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[3].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[4].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(x[0]))
    
    valid['s1'] = s1_temp
    valid['s2'] = s2_temp
    valid['label'] = np.array(label_temp)
    
    s1_temp = []
    s2_temp = []
    label_temp = []
    for i in range(3364+366, len(data)):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[3].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[4].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(x[0]))
       
    
    
    test['s1'] = s1_temp
    test['s2'] = s2_temp
    test['label'] = np.array(label_temp)

    
    return train, test, valid

def get_SICK_data():
    with open("dataset/SICK_train.txt", "r") as f:
        data = f.readlines()
    train = dict()
    test = dict()
    valid = dict ()
    judgement = dict()
    judgement['ENTAILMENT'] = 0
    judgement['NEUTRAL'] = 1
    judgement['CONTRADICTION'] = 2
    
    s1_temp = []
    s2_temp = []
    label_temp=[]
    for i in range(1,4501):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[1].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[2].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(judgement[x[4]]))
    train['s1'] = s1_temp
    train['s2'] = s2_temp
    train['label'] = np.array(label_temp)

    s1_temp = []
    s2_temp = []
    label_temp=[]
    for i in range(4501,5001):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[1].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[2].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(judgement[x[4]]))       
    valid['s1'] = s1_temp
    valid['s2'] = s2_temp
    valid['label'] = np.array(label_temp)
    
    s1_temp = []
    s2_temp = []
    label_temp=[]
    for i in range(5001,len(data)):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[1].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[2].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(judgement[x[4]]))    
    test['s1'] = s1_temp
    test['s2'] = s2_temp
    test['label'] = np.array(label_temp)
    return train, valid, test

def get_AI_data():
    with open("dataset/AI_train.txt", "r") as f:
        data = f.readlines()
    train = dict()
    test = dict()
    valid = dict ()
    
    s1_temp = []
    s2_temp = []
    label_temp = []
    for i in range(1,12690):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[1].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[2].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(x[0]))

    train['s1'] = s1_temp
    train['s2'] = s2_temp
    train['label'] = np.array(label_temp)

    s1_temp = []
    s2_temp = []
    label_temp = []
    for i in range(12690, 15174):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[1].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[2].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(x[0]))

    valid['s1'] = s1_temp
    valid['s2'] = s2_temp
    valid['label'] = np.array(label_temp)

    s1_temp = []
    s2_temp = []
    label_temp = []
    for i in range(15174, len(data)):
        x = data[i].rstrip().split("\t")
        s1_ = []
        for j in x[1].translate(table).split():
            if j.isdigit():
                s1_.append("<number>")
            else:
                s1_.append(j)
        s2_ = []
        for j in x[2].translate(table).split():
            if j.isdigit():
                s2_.append("<number>")
            else:
                s2_.append(j)
        s1_temp.append(" ".join(s1_))
        s2_temp.append(" ".join(s2_))
        label_temp.append(int(x[0]))

    test['s1'] = s1_temp
    test['s2'] = s2_temp
    test['label'] = np.array(label_temp)
    return train, valid, test

def write_AI():
    train, valid, test = get_AI_data()
    with open("demo.txt", "w") as f:
        for i in range(len(train['s1'])):
            new_s1 = []
            for j in train['s1'][i].split():
                if j.isdigit():
                    new_s1.append("<number>")
                else:
                    new_s1.append(j)
            new_s1 = " ".join(new_s1)
            new_s2 = []
            for j in train['s2'][i].split():
                if j.isdigit():
                    new_s2.append("<number>")
                else:
                    new_s2.append(j)
            new_s2 = " ".join(new_s2)
            f.write(str(train['label'][i]) + "\t" + new_s1 + "\t" + new_s2 + "\n")
        for i in range(len(test['s1'])):
            new_s1 = []
            for j in test['s1'][i].split():
                if j.isdigit():
                    new_s1.append("<number>")
                else:
                    new_s1.append(j)
            new_s1 = " ".join(new_s1)
            new_s2 = []
            for j in test['s2'][i].split():
                if j.isdigit():
                    new_s2.append("<number>")
                else:
                    new_s2.append(j)
            new_s2 = " ".join(new_s2)
            f.write(str(test['label'][i]) + "\t" + new_s1 + "\t" + new_s2 + "\n")
maxlen = 0      
def get_max_len(): 
#    train, valid, test = get_SICK_data()
#    train, valid, test = get_nli('dataset/SNLI/')       
    global maxlen
    for i in range(len(train['s1'])):
        if len(train['s1'][i].split()) >= maxlen:
            maxlen = len(train['s1'][i].split())
    for i in range(len(test['s1'])):
        if len(test['s1'][i].split()) >= maxlen:
            maxlen = len(test['s1'][i].split())
    for i in range(len(valid['s1'])):
        if len(valid['s1'][i].split()) >= maxlen:
            maxlen = len(valid['s1'][i].split())
    for i in range(len(train['s2'])):
        if len(train['s2'][i].split()) >= maxlen:
            maxlen = len(train['s2'][i].split())
    for i in range(len(test['s2'])):
        if len(test['s2'][i].split()) >= maxlen:
            maxlen = len(test['s2'][i].split())
    for i in range(len(valid['s2'])):
        if len(valid['s2'][i].split()) >= maxlen:
            maxlen = len(valid['s2'][i].split())
    print(maxlen)
        
def read_tree_data():
    train, valid, test = get_SICK_data()
    nlp = StanfordCoreNLP('/home/bob/Downloads/stanford-corenlp-full-2018-10-05')
    
    train_tree = dict()
    test_tree = dict()
    valid_tree = dict()
    
    train_tree['s1'] = []
    train_tree['s2'] = []
    for i in range(len(train['s1'])):
        print(i)
        x = nlp.parse(train['s1'][i])
        x = Tree.fromstring(x)
        train_tree['s1'].append(x)
        x = nlp.parse(train['s2'][i])
        x = Tree.fromstring(x)
        train_tree['s2'].append(x)
    train_tree['label'] = train['label']
    
    test_tree['s1'] = []
    test_tree['s2'] = []
    for i in range(len(test['s1'])):
        print(i)
        x = nlp.parse(test['s1'][i])
        x = Tree.fromstring(x)
        test_tree['s1'].append(x)
        x = nlp.parse(test['s2'][i])
        x = Tree.fromstring(x)
        test_tree['s2'].append(x)
    test_tree['label'] = test['label']
    
    valid_tree['s1'] = []
    valid_tree['s2'] = []
    for i in range(len(valid['s1'])):
        print(i)
        x = nlp.parse(valid['s1'][i])
        x = Tree.fromstring(x)
        valid_tree['s1'].append(x)
        x = nlp.parse(valid['s2'][i])
        x = Tree.fromstring(x)
        valid_tree['s2'].append(x)
    valid_tree['label'] = valid['label'] 
    return train_tree, valid_tree, test_tree

def get_SICK_tree_data():
    with open("sick_tree_data.pkl", "rb") as f:
        train, valid, test = pickle.load(f)
    return train, valid, test

def assign_vectors(tree, w2v):
    if type(tree[0]) != Tree:
        if tree[0] in w2v:
            tree[0] = Variable(torch.from_numpy(w2v[tree[0]]).float())
        else:
            tree[0] = Variable(torch.from_numpy(np.array([random.uniform(-0.5, 0.5) for _ in range(300)])).float())
        return tree
    else:
        for subtree in tree:
            assign_vectors(subtree, w2v)
        return tree
maxximum = 0
def traverse_tree(tree):
    global maxximum
    if type(tree[0]) != Tree:
#        print("leaf encountered: ", tree[0].size(), end = " \n")
        return  np.array([1,1]), tree[0]
    else:
        idd = 0
        ids = []
        word = []
        vectors = []
        summ = 0
        for subtree in tree:
            vec, _ = traverse_tree(subtree)
#            print(w)
#            word.append(w)
            summ += vec
            idd += 1
            if idd > maxximum:
                maxximum = idd
            ids.append(idd)
#        print(summ, word, ids)
        return summ, word  

visited = []
def print_dep_tree(tree):
    global visited
    if type(tree) != Tree:
        return tree
    else:
        c = []
        for i in range(len(tree)):
            c.append(print_dep_tree(tree[i]))
        print(c)
        print(tree.label())
    return tree.label()


def fill_tre_with_vectors():
    train_tree , valid_tree, test_tree = get_SICK_tree_data()
    filename = "transformer_SICk"
    print(filename)
    parser = argparse.ArgumentParser(description='NLI training')
    # paths
    parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
    parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
    parser.add_argument("--outputmodelname", type=str, default='model.pickle')
    parser.add_argument("--word_emb_path", type=str, default="glove.840B.300d.txt", help="word embedding file path")
    
    # training
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dpout_model", type=float, default=0.1, help="encoder dropout")
    parser.add_argument("--dpout_fc", type=float, default=0.1, help="classifier dropout")
    parser.add_argument("--nonlinear_fc", type=float, default=5, help="use nonlinearity in fc")
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
    parser.add_argument("--lrshrink", type=float, default=1, help="shrink factor for sgd")
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
    
    # model
    parser.add_argument("--encoder_type", type=str, default='LSTMEncoder', help="see list of encoders")
    parser.add_argument("--enc_lstm_dim", type=int, default=600, help="encoder nhid dimension")
    parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
    parser.add_argument("--fc_dim", type=int, default=150, help="nhid of fc layers")
    parser.add_argument("--n_classes", type=int, default=2, help="entailment/neutral/contradiction")
    parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
    
    # gpu
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    
    # data
    parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")
    
    params, _ = parser.parse_known_args()
    
    # set gpu device
    torch.cuda.set_device(params.gpu_id)
    
    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(params)
    
    """
    SEED
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    
    """
    DATA
    """
    #train, valid, test = get_nli(params.nlipath)
    
    train_tree , valid_tree, test_tree = get_SICK_tree_data()
    train , valid, test = get_SICK_data()
    
    
    word_vec = build_vocab(train['s1'] + train['s2'] +
                           valid['s1'] + valid['s2'] +
                           test['s1'] + test['s2'], params.word_emb_path)
    
    for i in range(len(train_tree['s1'])):
        x = deepcopy(assign_vectors(train_tree['s1'][i], word_vec))
        train_tree['s1'][i] = deepcopy(x)
        x = deepcopy(assign_vectors(train_tree['s2'][i], word_vec))
        train_tree['s1'][i] = deepcopy(x)
    
    for i in range(len(test_tree['s1'])):
        x = deepcopy(assign_vectors(test_tree['s1'][i], word_vec))
        test_tree['s1'][i] = deepcopy(x)
        x = deepcopy(assign_vectors(test_tree['s2'][i], word_vec))
        test_tree['s1'][i] = deepcopy(x)
        
    for i in range(len(valid_tree['s1'])):
        x = deepcopy(assign_vectors(valid_tree['s1'][i], word_vec))
        valid_tree['s1'][i] = deepcopy(x)
        x = deepcopy(assign_vectors(valid_tree['s2'][i], word_vec))
        valid_tree['s1'][i] = deepcopy(x)
    
    with open("sick_tree_data_tensor.pkl", "wb") as f:
        pickle.dump([train_tree, valid_tree, test_tree], f)
        
    return train_tree, valid_tree, test_tree

def get_sick_tree_data_tensor():
    train_tree, valid_tree, test_tree = fill_tre_with_vectors()
    with open("sick_tree_data_tensor.pkl", "wb") as f:
        pickle.dump([train_tree, valid_tree, test_tree], f)

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_
    
def traverse_dependency_tree():
    en_nlp = spacy.load('en')
    doc = en_nlp("The quick brown fox jumps over the lazy dog.")
    for x in doc.sents:
        t=(to_nltk_tree(x.root))
#    print(t)
    print_dep_tree(t)

traverse_dependency_tree()

def create_tensorflow_data():
    #id","qid1","qid2","question1","question2","is_duplicate
    train, valid, test = get_AI_data()
    with open("msrp_train.txt", "a") as f:
        for i in range(len(train['s1'])):   
            f.write(train['s1'][i] + "\t" + train['s2'][i] + "\t" + str(train['label'][i]) + "\n" )
    with open("msrp_train.txt", "a") as f:
        for i in range(len(valid['s1'])):   
            f.write(valid['s1'][i] + "\t" + valid['s2'][i] + "\t" + str(valid['label'][i]) + "\n" )
    with open("msrp_train.txt", "a") as f:
        for i in range(len(test['s1'])):   
            f.write(test['s1'][i] + "\t" + test['s2'][i] + "\t" + str(train['label'][i]) + "\n" )

#    with open("msrp_train.txt", "a") as f:
#        for i in range(len(train['s1'])):   
#            f.write(str(i) + "\",\"" + str(i) + "\",\"" + str(i) + "\",\"" + train['s1'][i] + "\",\"" + train['s2'][i] + "\",\"" + str(train['label'][i]) + "\n" )
#    with open("msrp_test.txt", "a") as f:
#        for i in range(len(test['s1'])):   
#            f.write(str(i) + "\",\"" + test['s1'][i] + "\",\"" + test['s2'][i] + "\",\"" + "\n" )

create_tensorflow_data()       
#train_tree, valid_tree, test_tree = fill_tre_with_vectors()
#traverse_tree(train_tree['s1'][0])
import pandas
def get_QQP_data():
    data = pandas.read_csv("dataset/qqp.csv")
    train = dict()
    test = dict()
    valid = dict ()
    train['s1'] = []
    train['s2'] = []
    train['label'] = []
    question1 = list(data['question1'])
    question2 = list(data['question2'])
    label = list(data['is_duplicate'])
    q1 = question1[0:362466]
    q2 = question2[0:362466]
    cls = label[0:362466]
    for i in range (len(q1)):
        if type(q1[i]) is not str or type(q2[i]) is not str:
            continue
        train['s1'].append(q1[i])
        train['s2'].append(q2[i])
        train['label'].append(cls[i])
    train['label'] = np.array(train['label'])   
    q1 = question1[362466:362466 + 1213]
    q2 = question2[362466:362466 + 1213]
    cls = label[362466:362466 + 1213]
    valid['s1'] = []
    valid['s2'] = []
    valid['label'] = []
    for i in range (len(q1)):
        if type(q1[i]) is not str or type(q2[i]) is not str:
            continue
        valid['s1'].append(q1[i])
        valid['s2'].append(q2[i])
        valid['label'].append(cls[i])
    valid['label'] = np.array(valid['label'])
    q1 = question1[362466 + 1213:]
    q2 = question2[362466 + 1213:]
    cls = label[362466 + 1213:]
    test['s1'] = []
    test['s2'] = []
    test['label'] = []
    for i in range (len(q1)):
        if type(q1[i]) is not str or type(q2[i]) is not str:
            continue
        test['s1'].append(q1[i])
        test['s2'].append(q2[i])
        test['label'].append(cls[i])
    test['label'] = np.array(test['label'])
    return train, test, valid