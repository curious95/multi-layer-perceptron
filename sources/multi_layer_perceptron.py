import numpy as np
import math


# tanh activation function
def tanH_act(val):
    num = math.exp(val)-math.exp(-val)
    den = math.exp(val)+math.exp(-val)
    return num/den;

#sigmoid activation function
def sigmoid_act(val):
    num = 1
    den = 1+math.exp(-val)
    return num/den;

#initial model hyper parameters
def init_model_hyper_params(train, labels, no_hidden_neurons):

    NI=train.shape[1]
    NH=no_hidden_neurons
    NO=labels.shape[1]

    #xaviers initialization for tanh
    W1 = np.random.rand(NI,NH)*(1/math.sqrt(NI))
    W2 = np.random.rand(NH,NO) * (1 / math.sqrt(NH))

    return {"NI": NI,"NH": NH,"NO": NO,"W1": W1, "W2": W2}


def forward(model_params, train, label):
    n_rows = train.shape[0]

    W1 = model_params["W1"]
    W2 = model_params["W2"]

    Z1 = train @ W1
    if model_params["hidden_layer_func"] == "tanh":
        A1 = tanH_act(Z1)
    else:
        A1 = sigmoid_act(Z1)
    Z2 = A1 @ W2
    if model_params["output_layer_func"] == "sigmoid":
        A2 = sigmoid_act(Z2)
    else:
        A2 = Z2
    cost = float('nan')
    if label is not None:
        predicted_label = A2
        cost = 1 / (2 * n_rows) * np.sum(np.square(Y=label - predicted_label))
        cost = np.squeeze(cost)

    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}, cost



