# Counting sequences using LSTM's
This is a short problem statement I found [Monik's blog](http://monik.in/), which I wanted to try out. The aim is to create an RNN model which would simply output the number of 1's it had seen in a binary sequence containing ```input_sample_size``` number of elements. The main purpose of this exercise was to observe the effectiveness of a RNN at retaining information which it had encountered earlier in the sequence.

## Dependencies
* Tensorflow
* Numpy
* tqdm

## Easy to visualize now?
![rnn](https://user-images.githubusercontent.com/34591573/34469281-7c7ccb58-ef41-11e7-945e-7fc2e7e56675.png)
This is an image I got from [Christopher Olah's blog](http://colah.github.io/). Notice how the information is being accumulated at each proceeding time step. Let's say that the A's are our hidden layer at each time step. So now, we are interested in collecting the hidden layer output produced at the last time step.

## Simplified Approach
* Firstly, data having a sequence length of ```input_sample_size``` is generated using the ```generate_data()``` function. Note that this data is evenly distributed over the whole sequence length. Therefore the probablity of encountering a "1" varies from 0 to 1(it is not centralized).
* For example in a case where ```input_sample_size = 10``` the probablity of encountering a particular number of 1's is evenly distributed distributed over the whole dataset. Also note that the number of outputs will be 11 i.e. ```n_classes = 11``` as the possible outputs are 0, 1, 2, 3 ...10.
* The data is split into training and testing set. I'm using 25% of the total data as my test set.
* As there is a need for retaining long term sequences, I've used the LSTM cell to capture the sequence information.
* LSTMs are well suited for time series data or sequential data, where the temporal dynamics of the system matters.
* The input sequence of length ```time_steps = input_sample_size``` is fed into the network. One element of the sequence, at a time(at a single time step).
* Here's where the real magic happens!
	* When an element of the sequence is fed into the hidden layer at every time step, this hidden layer passes the information into the next time step. This way temporal information keeps on accumulating in the hidden layer at every time step.
	* The same hidden layer, at the last time step accumulates enough information within itself, and the vector which represents this hidden state is called the "thought vector"(cool name). Now the network is ready to make a prediction(make sure you understood every bit of this point).
* As we only care about the final output only, we discard all the outputs(of all the other time steps) except the one which we get after the whole sequence has been seen by the network.
* This final output is passed through the softmax function to give a probablity distribution over the number of 1's the network "thinks" it has seen in the sequence.
* A prediction is made, the loss is calculated(cross-entropy) with respect to the weights, the loss is backpropagated over the ```truncated_backprop_length```(I haven't included this in the code), and finally the weights are updated.

# Results
## Training
![trainresult](https://user-images.githubusercontent.com/34591573/34521217-9eaa84b4-f0b2-11e7-9e6a-a5231ed724c9.png)

I got a training error of 0.6% after 3000 epochs, which is quite impressive!
The hyperparameters were set to the following:
```
batch_size = 1000
time_steps = input_sample_size #List of 20 binary elements
input_size = 1 
num_units = 24 
n_classes = input_sample_size + 1 # +1 is because number possible outputs = 21
learning_rate = 0.001
epochs = 3000
keep_prob = 0.7 
```
* Having limited compute power, I used 0.08% i.e. 10000 out of 125000 of my data, as the training set.
* The model could give better results, for example, if the epochs were increased to ```epochs = 5000``` as mentioned in [Monik's blog](http://monik.in/).
## Testing
![testresult](https://user-images.githubusercontent.com/34591573/34521225-b201bf78-f0b2-11e7-9082-b72191221573.png)

* Looking at the test accuracy of 99.595%, we could quite confidently conclude that the model did not overfit, and it properly generalized over the dataset.
* The model could be improved by adding a validation set, which would allow us to have a better understanding of when to stop training.


## Basic Usage
For Running, type in terminal
```
python lstmCounter.py
```
For the beautiful visualization, type in terminal
```
tensorboard --logdir="visualize"
```



