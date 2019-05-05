from sources import multi_layer_perceptron as model
import numpy as np
np.random.seed(6700)


# Task 1 Part 1
input=np.array([[0,0],[0,1],[1,0],[1,1]])
label=np.array([[0],[1],[1],[0]])

# Training the model
#model.train_mlp(input,label,input,label,"tanh","sigmoid",3,400,0.3)


# Task 1 Part 2


# Task 1 Part 3
#print(np.random.rand(200,4))
input = np.random.uniform(low=-1, high=1, size=(200, 4))
label = np.sin(np.sum(input,axis=1).reshape(200,1))

input_train = input[:150,:]
label_train = label [:150,:]
input_test = input[150:,:]
label_test = label [150:,:]

model.train_mlp(input_train,label_train,input_test,label_test,"tanh","sigmoid",10,10000,0.02)


