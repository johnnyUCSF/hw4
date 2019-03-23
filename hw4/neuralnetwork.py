import numpy as np
import copy
import random

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def transform_string(outputs):
    ##split up string inputs for identity matrix testing
    outputs_np = [0.0]*len(outputs)
    for i in range(len(outputs)):
        outputs_np[i] = float(outputs[i])
    outputs_np = np.array(outputs_np)
    return(outputs_np)

class NeuralNet:
    #####creates a one hidden layer NN with variable input and output size
    def __init__(self, size_input, size_hidden, size_output, learnrate, lambdarate): 
        ##define parameters for NN
        self.learnrate = float(learnrate)
        self.lambdarate = float(lambdarate)
        self.size_input = size_input
        self.size_hidden = size_hidden
        self.size_output = size_output
        init_factor = 0.1
        ##initialize weights 
        self.weights1 = np.random.rand(size_input,size_hidden)*init_factor
        self.weights2 = np.random.rand(size_hidden,size_output)*init_factor
        ##initialize biases
        self.bias1 = np.random.rand(size_hidden)*init_factor
        self.bias2 = np.random.rand(size_output)*init_factor
        ##clear delta matrices
        self.clear_deltas()
    #####clear delta matrices    
    def clear_deltas(self):
        ##initialize delta matrices which hold delta sum across all examples
        self.delta_w1 = np.zeros((self.size_input,self.size_hidden))
        self.delta_w2 = np.zeros((self.size_hidden,self.size_output))
        #
        self.delta_b1 = np.zeros(self.size_hidden)
        self.delta_b2 = np.zeros(self.size_output)
        ##counter for number of training examples
        self.trainset_count = 0
    #####propagate forward based on saved input data; note layer 1 = input, layer 2 = hidden layer, layer 3 = output layer
    def prop_forward(self,inputs, outputs):
        ##read in inputs and for this training example
        self.input_train = inputs
        self.output_train = outputs
        ##propagate activation functions forward
        self.layer2_activation = sigmoid(np.dot(self.input_train,self.weights1)+self.bias1)
        self.layer3_activation = sigmoid(np.dot(self.layer2_activation,self.weights2)+self.bias2)
        ##update training data counter
        self.trainset_count += 1
    #####propagate backward
    def prop_backward(self):
        ##calc derivatives of the weights
        self.d_weights2 = np.outer(self.layer2_activation, ((self.output_train - self.layer3_activation) * sigmoid(self.layer3_activation,derivative=True)))
        self.d_weights1 = np.outer(self.input_train, (np.dot(self.weights2,(self.output_train - self.layer3_activation) * sigmoid(self.layer3_activation,derivative=True)) * sigmoid(self.layer2_activation,derivative=True)))
        ##calc derivatives of the biases
        self.d_bias2 = ((self.output_train - self.layer3_activation) * sigmoid(self.layer3_activation,derivative=True))
        self.d_bias1 = (np.dot(self.weights2,(self.output_train - self.layer3_activation) * sigmoid(self.layer3_activation,derivative=True)) * sigmoid(self.layer2_activation,derivative=True))
        ##add error to delta matrices
        self.weights2 += self.learnrate*self.d_weights2
        self.weights1 += self.learnrate*self.d_weights1
        #
        self.bias2 += self.learnrate*self.d_bias2
        self.bias1 += self.learnrate*self.d_bias1

        
def evaluate_accuracy(pos_predicted,pos_true,neg_predicted,neg_true): 
    ####evaluate accuracy
    TP = 0
    TN = 0 
    P = len(pos_predicted)
    N = len(neg_predicted)
    ###
    for item in pos_predicted:
        if item in pos_true:
            TP += 1
    ##
    for item in neg_predicted:
        if item in neg_true:
            TN += 1
    ##
    accuracy = float((TP+TN))/float((P+N))
    return accuracy

def call_results(train_inputs,train_outputs,model):
    pos_predicted = []
    neg_predicted = []
    pos_true = []
    neg_true = []
    #####evaluate trues
    for i in range(len(train_inputs)):
        if train_outputs[i] == '1':
            pos_true.append(train_inputs[i])
        if train_outputs[i] == '0':
            neg_true.append(train_inputs[i])
    #####evaluate calls from model
    for i in range(len(train_inputs)):
        model.prop_forward(transform_string(train_inputs[i]),transform_string(train_outputs[i]))
        call = np.round(model.layer3_activation)
        if call == 1:
            pos_predicted.append(train_inputs[i])
        if call == 0:
            neg_predicted.append(train_inputs[i])
    return pos_predicted,pos_true,neg_predicted,neg_true

def chunk_kfold(num_parts,neg_encoded,pos_encoded):
    #####chunk data for k-fold cross validation
    neg_chunks = list(chunks(neg_encoded,int(len(neg_encoded)/num_parts)+1))
    pos_chunks = list(chunks(pos_encoded,int(len(pos_encoded)/num_parts)+1))
    return neg_chunks,pos_chunks

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def format_trainingdata(neg_chunks,pos_chunks):
    #####create output data based on 1=positive, 0=negative
    #####concatenate pos and negative chunks and results
    training_inputs = []
    training_outputs = []
    for i in range(len(pos_chunks)):
        tmp_input = []
        tmp_output = []
        for item in pos_chunks[i]:
            tmp_input.append(item)
            tmp_output.append('1')
        for item in neg_chunks[i]:
            tmp_input.append(item)
            tmp_output.append('0')
        ##
        training_inputs.append(tmp_input)
        training_outputs.append(tmp_output)
    #####
    return training_inputs,training_outputs


def train_model(iterations,train_inputs,train_outputs,model):
    for i in range(iterations):
        model.clear_deltas()
        for i in range(len(train_inputs)):
            model.prop_forward(transform_string(train_inputs[i]),transform_string(train_outputs[i]))
            model.prop_backward()
    return model


