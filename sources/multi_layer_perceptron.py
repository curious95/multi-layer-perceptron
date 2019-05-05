import numpy as np
import math

np.random.seed(60001)

# tanh activation function
def tanH_act(val):
    num = np.exp(val)-np.exp(-val)
    den = np.exp(val)+np.exp(-val)
    return num/den;

#sigmoid activation function
def sigmoid_act(val):
    num = 1
    den = 1+np.exp(-val)
    return num/den;

#initial model hyper parameters
def init_model_hyper_params(train, labels, no_hidden_neurons):

    NI=train.shape[1]
    NH=no_hidden_neurons
    NO=labels.shape[1]

    #xaviers initialization for tanh
    W1 = np.random.rand(NI,NH)*(1/np.sqrt(NI))
    W2 = np.random.rand(NH,NO) * (1 / np.sqrt(NH))

    return {"NI": NI,"NH": NH,"NO": NO,"W1": W1, "W2": W2}


# forward propagation module
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
        cost = 1 / (2 * n_rows) * np.sum(np.square(label - predicted_label))
        cost = np.squeeze(cost)

    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}, cost


# backward propagation module
def back_prop(learning_rate, params, X, Y, cache, hidden="sigmoid", output="sigmoid"):
    n = X.shape[0]

    W1 = params["W1"]
    W2 = params["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    if hidden == "sigmoid":
        f_dash = sigmoid_act(Z1) * (1 - sigmoid_act(Z1))
    else:
        f_dash = 1 - np.square(tanH_act(Z1))

    if output == "sigmoid":
        delta_o = (A2 - Y) * (sigmoid_act(Z2) * (1 - sigmoid_act(Z2)))
    else:
        delta_o = A2 - Y

    dW2 = 1 / n * (A1.T @ delta_o)
    W2 -= learning_rate * dW2
    delta_h = (delta_o @ W2.T) * f_dash
    dW1 = 1 / n * (A1.T @ delta_h)

    params["W1"] = W1
    params["W2"] = W2

    return params

# fitting
def model_fit(params, X, Y, hidden, output, epochs=2000, learning_rate=0.8, verbose=False):
    loss_log = []
    for i in range(epochs):
        cache, loss = forward(params, X, Y)
        params = back_prop(learning_rate, params, X, Y, cache, hidden, output)

        # logs
        if i % 1000 == 0:
            loss_log.append(np.asscalar(loss))
            if verbose:
                print("Loss after {} iterations: {:.3f}".format(i, loss))

    return params, loss_log


# prediction
def model_predict(params, X, hidden, output):
    cache, _ = forward(params, X, None)
    if output == "sigmoid":
        Y_hat = (cache["A2"] > 0.5).astype(int)
    else:
        return cache["A2"]
    return Y_hat


#model
def mlp(input_train, label_train, input_test, label_test, hidden, output, no_hidden_neurons, epochs, learning_rate):

    hyper_params = init_model_hyper_params(input_train, label_train, no_hidden_neurons)
    hyper_params["hidden_layer_func"] = hidden
    hyper_params["output_layer_func"] = output

    hyper_params, loss = model_fit(hyper_params, input_train, label_train, hidden, output, epochs, learning_rate, verbose=True)
    Y_hat_train = model_predict(hyper_params, input_train, hidden, output)
    Y_hat_test = model_predict(hyper_params, input_test, hidden, output)
    train_acc = (100 * (1 - np.mean(np.abs(input_test - Y_hat_train))))
    test_acc = (100 * (1 - np.mean(np.abs(label_test - Y_hat_test))))

    print("{:.1f}% training acc.".format(train_acc))
    print("{:.1f}% test acc.".format(test_acc))

    return {"PARAMS": hyper_params, "LOSS": loss, "ACC": [train_acc, test_acc], "LR": learning_rate, "Y_hat": Y_hat_test,
            "Y_test": label_test}

# data init
input=np.array([[0,0],[0,1],[1,0],[1,1]])
label=np.array([[0],[1],[1],[0]])

#model
mlp(input,label,input,label,"tanh","sigmoid",3,20000,0.8)