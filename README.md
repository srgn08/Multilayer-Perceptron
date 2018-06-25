# Compile

Running the tests:

- The program needs 4 arguments for its operation. These are the names of the positives,negatives,vector filename and training size.


 Sample run code:

$ python3 assignment4.py positive.txt negative.txt vectors.txt 75



 Explanation of functions:

 main():
 
- The main function read arguments. Call the appropriate functions and calculate the accuracy.


 readfile(filename):
 
- This function reads the desired file.



 create_weight_bias(input,hidden_one,hidden_two,number_of_classes):
 
- This function generates weight and bias according to hidden and input numbers.


 split_vector(array_vector):
 
- This function parse the features of the vector file and keeps the words in array.


 multilayer_perceptron(x,weight,bias):
 
- This function creates a neural network.



 create_data(data,train_length):

-This function generates train and test data.



 Average duration of the program:
 
- It takes about 3-4 seconds to do all the operations.
 
Authors: Sergen Topcu



