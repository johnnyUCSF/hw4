import numpy as np
import copy
import random

def onehot(basepair):
    ###define dictionary
    encode_dict = {
    'A':'1000',
    'T':'0100',
    'C':'0010',
    'G':'0001'
    }
    ###return
    return encode_dict[basepair]

def encode_pos_neg(neg,pos,N):
    ####generate one hot encoded versions of training examples
    neg_encoded = []
    for seq in neg:
        i = 0
        tmp = ''
        for basepair in seq:
            tmp += onehot(basepair)
            ###take only first N basepairs
            if i >= N-1:
                break
            i += 1
        neg_encoded.append(tmp)
    ###
    pos_encoded = []
    for seq in pos:
        tmp = ''
        for basepair in seq:
            tmp += onehot(basepair)
        pos_encoded.append(tmp)
    ###filter by size
    for item in neg_encoded:
        if len(item) != N*4:
            neg_encoded.pop(neg_encoded.index(item))
    for item in pos_encoded:
        if len(item) != N*4:
            pos_encoded.pop(pos_encoded.index(item))
    #####
    return neg_encoded,pos_encoded
    
    
    
