import numpy as np
np.random.seed(6700)

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
def init_model_hyper_params(input, labels, no_hidden_neurons):

    NI=input.shape[1]
    NH=no_hidden_neurons
    NO=labels.shape[1]

    #xaviers initialization for tanh
    W1 = np.random.randn(NI,NH)*(1/np.sqrt(NI))
    W2 = np.random.randn(NH,NO) * (1 / np.sqrt(NH))

    return {"NI": NI,"NH": NH,"NO": NO,"W1": W1, "W2": W2}


# forward propagation module
def forward(hyper_params, input, label):
    n_rows = input.shape[0]

    W1 = hyper_params["W1"]
    W2 = hyper_params["W2"]

    Z1 = input @ W1
    if hyper_params["hidden_layer_func"] == "tanh":
        A1 = tanH_act(Z1)
    else:
        A1 = sigmoid_act(Z1)
    Z2 = A1 @ W2
    if hyper_params["output_layer_func"] == "sigmoid":
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
def backward(learning_rate, hyper_params, input, label, cache):
    n = input.shape[0]

    W1 = hyper_params["W1"]
    W2 = hyper_params["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    if hyper_params["hidden_layer_func"] == "sigmoid":
        hidden_act_val = sigmoid_act(Z1) * (1 - sigmoid_act(Z1))
    else:
        hidden_act_val = 1 - np.square(tanH_act(Z1))

    if hyper_params["output_layer_func"] == "sigmoid":
        dZ2 = (A2 - label) * (sigmoid_act(Z2) * (1 - sigmoid_act(Z2)))
    else:
        dZ2 = A2 - label

    dW2 = 1 / n * (A1.T @ dZ2)
    W2 -= learning_rate * dW2
    dZ1 = (dZ2 @ W2.T) * hidden_act_val
    dW1 = 1 / n * (input.T @ dZ1)
    W1 -= learning_rate * dW1

    hyper_params["W1"] = W1
    hyper_params["W2"] = W2

    return hyper_params

# fitting
def mlp_fit(hyper_params, input, label):
    loss_log = []
    epochs = hyper_params["epochs"]
    learning_rate = hyper_params["learning_rate"]
    for i in range(epochs):
        cache, cost = forward(hyper_params, input, label)
        hyper_params = backward(learning_rate, hyper_params, input, label, cache)

        # logs
        if i % 10 == 0:
            loss_log.append(np.asscalar(cost))
            print("Loss after {} iterations: {:.3f}".format(i, cost))

    return hyper_params, loss_log


# prediction
def predict(hyper_params, input):
    cache, _ = forward(hyper_params, input, None)
    if hyper_params["output_layer_func"] == "sigmoid":
        predicted = (cache["A2"] > 0.5).astype(int)
    else:
        return cache["A2"]
    return predicted


#model
def mlp(input_train, label_train, input_test, label_test, hidden, output, no_hidden_neurons, epochs, learning_rate):

    hyper_params = init_model_hyper_params(input_train, label_train, no_hidden_neurons)
    hyper_params["hidden_layer_func"] = hidden
    hyper_params["output_layer_func"] = output
    hyper_params["epochs"] = epochs
    hyper_params["learning_rate"] = learning_rate

    hyper_params, cost = mlp_fit(hyper_params, input_train, label_train)
    predicted_label_train = predict(hyper_params, input_train)
    predicted_label_test = predict(hyper_params, input_test)
    train_acc = (100 * (1 - np.mean(np.abs(label_train - predicted_label_train))))
    test_acc = (100 * (1 - np.mean(np.abs(label_test - predicted_label_test))))

    print("{:.1f}% training acc.".format(train_acc))
    print("{:.1f}% test acc.".format(test_acc))

    return {"PARAMS": hyper_params, "COST": cost, "ACC": [train_acc, test_acc], "LR": learning_rate, "Predicted": predicted_label_test,
            "Y_test": label_test}

# data init
input=np.array([[0,0],[0,1],[1,0],[1,1]])
label=np.array([[0],[1],[1],[0]])

#model
mlp(input,label,input,label,"tanh","sigmoid",3,400,0.3)