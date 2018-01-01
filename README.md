# Counting sequences using LSTM's
This is a short problem statement I found [monik's blog](http://monik.in/), which I wanted to try out. The aim is to create an RNN model which would simply output the number of 1's it had seen in a binary sequence containing ```input_sample_size``` number of elements. The main purpose of this exercise was to observe the effectiveness of a RNN at retaining information which it had encountered earlier in the sequence.

## Dependencies
* Tensorflow
* Numpy
* tqdm

## Easy to visualize now?
![rnn](https://user-images.githubusercontent.com/34591573/34469281-7c7ccb58-ef41-11e7-945e-7fc2e7e56675.png)
This is an image I got from [Christopher Olah's blog](http://colah.github.io/).Notice how the information is being accumulated at each proceeding time step. Let's say that the A's are our hidden layer at each time step. So now, we are interested in collecting the hidden layer output produced at the last time step.

## Simplified Approach
* Firstly, data having a sequence length of ```input_sample_size``` is generated using the ```generate_data()``` function. Note that this data is evenly distributed over the whole sequence length. Therefore the probablity of encountering a "1" varies from 0 to 1(it is not centralized).
* For example in a case where ```input_sample_size = 10``` the probablity of encountering a particular number of 1's is evenly distributed distributed over the whole dataset. Also note that the number of output will be 11 i.e ```n_classes = 11``` as the possible outputs are 0, 1, 2, 3 ...10.
* The data is split into train-test set. I'm using 25% of the total data as my test set.
* As there is a need for retaining long term sequences, I've used the LSTM cell to capture the sequence information.
* LSTMs are well suited for time series data or sequential data, where the temporal dynamics of the system matters.
* The input sequence of length ```time_steps``` is fed into the network. 
* Here's where the real magic happens. When a sequence is fed into the hidden layer at every time step, this hidden layer passes the information into the next time step. This way temporal information accumulates in the hidden layer at every time step. The same hidden layer, at the last time step accumulates enough information inside what is called the "thought vector", to make a prediction.(make sure you understood every bit of this sentence)
* As we only care about the final output only, we discard all the outputs except the one which we get after the whole sequence has been seen.
* This final output is passed through the softmax function to give a probablity distribution over the number of 1's the network "thinks" it has seen in the sequence.
* The prediction is made, the loss is calculated(cross-entropy) with respect to the weights, the loss is backpropagated over the ```truncated_backprop_length```(I haven't included this in the code), and finally the weights are updated.
 

## Basic Usage
For Running, type in terminal
```
python lstmCounter.py
```
For the beautiful visualization, type in terminal
```
tensorboard --logdir="visualize"
```



