from hw4 import neuralnetwork as NN
from hw4 import encoding as EN
import numpy as np
import pytest
import os

def test_nn():
    ############
    ######this test all major neural network functions directly or indirectly by function calls within functions
    ############
    #####create identity matrix
    inputs = ['10000000','01000000','00100000','00010000','00001000','00000100','00000010','00000001']
    for item in inputs:
        #####run NN
        model = NN.NeuralNet(8,3,8,2.0,0.01)
        iterations = 800
        for i in range(iterations):
            model.clear_deltas()
            for item in inputs:
                model.prop_forward(NN.transform_string(item),NN.transform_string(item))
                model.prop_backward()
    #####test
    model.prop_forward(NN.transform_string(inputs[0]),NN.transform_string(inputs[0]))
    result_nn = np.round(model.layer3_activation)
    print(result_nn)
    assert list(result_nn) == [1,0,0,0,0,0,0,0]

def test_encoding():
    ############
    ######this test all major encoding functions directly or indirectly by function calls within functions
    ############
    neg = ['ATCG']
    pos = ['TCGA']
    ####
    neg_encoded,pos_encoded = EN.encode_pos_neg(neg,pos,4)
    print(neg_encoded)
    ####
    assert neg_encoded[0]=='1000010000100001'
    
    
    
    
    
    