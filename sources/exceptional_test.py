from sources import multi_layer_perceptron as model
import numpy as np
import pandas as pd
np.random.seed(6700)

# Reading the letter recognition data file
data = pd.read_csv("letter-recognition.data")

# Input columns
input = np.array(data.iloc[:,1:])

# normalization of data
input = input - input.mean()
input = input / input.max()

# one hot encoded target columns
label = np.array(pd.get_dummies(data.iloc[:,0]))

# Dividing test and train set in 1:5 ration
input_train = input[:16000,:]
label_train = label [:16000,:]
input_test = input[16000:,:]
label_test = label [16000:,:]


# for 10 hidden units
result = model.train_mlp(input_train,label_train,input_test,label_test,"tanh","sigmoid",10,1000,0.5,False)
predicted = ["".join([str(i) for i in elem]) for elem in result["Predicted"]]
labels = ["".join([str(i) for i in elem]) for elem in result["Actual"]]
correctlyPredicted = [i for i in range(3999) if predicted[i]==labels[i]]
print("Accuracy = {:.1f}%".format(len(correctlyPredicted)*100/3999))


# for 20 hidden units
result=model.train_mlp(input_train,label_train,input_test,label_test,"tanh","sigmoid",50,5000,0.8,False)
predicted = ["".join([str(i) for i in elem]) for elem in result["Predicted"]]
labels = ["".join([str(i) for i in elem]) for elem in result["Actual"]]
correctlyPredicted = [i for i in range(3999) if predicted[i]==labels[i]]
print("Accuracy = {:.1f}%".format(len(correctlyPredicted)*100/3999))



# for 30 hidden units
result = model.train_mlp(input_train,label_train,input_test,label_test,"tanh","sigmoid",80,10000,1,False)
predicted = ["".join([str(i) for i in elem]) for elem in result["Predicted"]]
labels = ["".join([str(i) for i in elem]) for elem in result["Actual"]]
correctlyPredicted = [i for i in range(3999) if predicted[i]==labels[i]]
print("Accuracy = {:.1f}%".format(len(correctlyPredicted)*100/3999))

