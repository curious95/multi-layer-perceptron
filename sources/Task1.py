from sources import multi_layer_perceptron as model
import numpy as np
np.random.seed(6700)


# Task 1 Part 1
input=np.array([[0,0],[0,1],[1,0],[1,1]])
label=np.array([[0],[1],[1],[0]])

# Training the model
print("Executing Task 1")
result = model.train_mlp(input,label,input,label,"tanh","sigmoid",2,400,0.3,False)
print("Training Accuracy = {:.1f}%".format(result["Accuracies"][0]))

# Task 1 Part 2
print("Testing Accuracy = {:.1f}%".format(result["Accuracies"][1]))



print("\n\n\nTask 1 Part3 : For random numbers and sin function")

# Task 1 Part 3
#Taking random numbers
input = np.random.uniform(low=-1, high=1, size=(200, 4))

#computing sin function
label = np.sin(np.sum(input,axis=1).reshape(200,1))

# splitting data into training and test set
input_train = input[:150,:]
label_train = label [:150,:]
input_test = input[150:,:]
label_test = label [150:,:]

result = model.train_mlp(input_train,label_train,input_test,label_test,"tanh","linear",5,10000,0.05,False)
print("Training Accuracy = {:.1f}%".format(result["Accuracies"][0]))
print("Testing Accuracy = {:.1f}%".format(result["Accuracies"][1]))

