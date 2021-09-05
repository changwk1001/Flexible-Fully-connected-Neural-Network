# Flexible-Fully-connected-Neural-Network
This is a flexible neural network for everyone to easily create a fully-connected-neural-network to have a peak about the deep-learning. The followings are information about the network and  what you can do about your network.

## 0. Language: Python
## 1. Training data set: Mnist
## 2. Parameters and their defult value:  
          Learning_rate: 0.001  
          Iteration: 10  
          Loss_function: square -> choice: square, logistic  
          Activation_function: Relu -> choice: Relu, logistic  
          Number_of_layer: 4  
          Sequence_of_layer: [784,40,20,10]   
          Regulation: None -> choice: None, L2 regulation  
          Batch_size: 1000  
          Regulation_strength: 0.01  
## 3. An additional function to find out the wrongly classified images.
## 4. Example:  
   If you use the default value for your network, you will get the following network:
![image](https://user-images.githubusercontent.com/90007478/132129154-a7693a9d-535c-458d-be03-5cac79471c83.png)
   And these are examples of wrongly classified images:
![image](https://user-images.githubusercontent.com/90007478/132129202-6dbfc453-8108-43cb-8587-91ca4686699a.png)
## 5. Note:
1. I did not use any pytorch module since that by reading the codes for forward and backward propogations can help to formulate better understanding of the neural-network.
2. The first and last layers will be for your input and output, and thus you will need to adjust them to fit your own data dimensions.
3. Hope everyone has fun learning deep learning!!
