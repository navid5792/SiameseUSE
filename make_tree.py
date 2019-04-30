from nltk import Tree
import spacy

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_



def assign_vectors_d(tree):
    if type(tree) != Tree:
        tree = Tree(1,[])
        return tree
    else:
        c = []
        x = 0
        for i in range(len(tree)):
            ch = assign_vectors_d(tree[i])
            x = x + ch.label()
            #c.append(assign_vectors_d(tree[i]))
        
#        for i in range(len(tree)):
#            if type(tree[i]) != Tree:
#                tree[i] = 1
#        
        tree.set_label(x)
        return tree
    
    
    
en_nlp = spacy.load('en')
doc = en_nlp("The quick brown fox jumps over the lazy dog.")
for x in doc.sents:
    t=(to_nltk_tree(x.root))
t = assign_vectors_d(t)
