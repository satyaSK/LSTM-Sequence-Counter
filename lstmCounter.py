#problem statement on monik's page: http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np 
from tqdm import tqdm
input_sample_size = 20 #Use LSTM to count the number of 1's in a list of input_sample_size binary elements
# fn to generate 100000 random lists(input_sample_size elements each) of binary elements

def generate_data(input_sample_size):
	x=[]
	size = 100000
	for i in range(125000):
		a = np.random.randint(input_sample_size)/input_sample_size
		b = 1 - a
		sample = list(np.random.choice(2, input_sample_size, p=[a,b]))
		x.append(sample)
	
	y=[]
	for i in x:
		count = int(np.sum(i))
		y.append(count)
	
	one_hot = np.zeros((len(x),input_sample_size + 1), dtype=np.int8)
	j = 0 #input_sample_size + 1 because there could be 21 outputs- from 0,1,2...20
	
	for i in one_hot:
		i[y[j]] = 1
		j += 1 

	ti = np.array(x[:size])
	to = np.array(one_hot[:size])
	xi = np.array(x[size:len(x)])
	xo = np.array(one_hot[size:len(x)])

	return ti, to, xi, xo

#get the generated data
ti, to, xi, xo = generate_data(input_sample_size)

print("Shape of training input :",ti.shape)
print("Shape of training labels :",to.shape)

#define hyperparameters
keep_prob = 0.7 
batch_size = 100
time_steps = input_sample_size# list of 20 binary elements
input_size = 1 
num_units = 128 
n_classes = input_sample_size + 1#again +1 is because number possible outputs = 21
learning_rate = 0.001
epochs = 4


X = tf.placeholder(tf.float32, [None, time_steps, input_size], name='X')
Y = tf.placeholder(tf.float32, [None, n_classes], name='Y')

W2 = tf.Variable(tf.random_normal([num_units, n_classes]))
b2 = tf.Variable(tf.random_normal([n_classes]))

shapedInputs = tf.unstack(X, time_steps,1)# convert it into a list of tensor of shape [batch_size, input_size] of length time_steps which is the input taken by static_rnn


cell = rnn.BasicLSTMCell(num_units, forget_bias = 1)# set the forgrt gate bias of the LSTM to 1(It is a default value but Im explicitly mentioning it)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)#this is the LSTM layer wrapped with dropout
all_outputs, all_hidden_states = rnn.static_rnn(cell, shapedInputs, dtype='float32')

prediction = tf.matmul(all_outputs[-1],W2) + b2

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels = Y)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

on_point=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
training_accuracy=tf.reduce_mean(tf.cast(on_point,tf.float32))

with tf.Session() as sess:
    # Train the Model
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter("./visualize", sess.graph) 
	num_batches = int(len(ti)/batch_size)
	print("\nGood To Go - Training Starts\n")
	for i in tqdm(range(epochs)):
	 	epoch_loss = 0
	 	s = 0
	 	for j in range(num_batches):
	 		X_batch, Y_batch = ti[s : s+batch_size], to[s : s+batch_size]
	 		X_batch = X_batch.reshape((batch_size,time_steps, input_size))
	 		e, _ = sess.run([loss, optimizer], feed_dict={X:X_batch,Y:Y_batch})
	 		epoch_loss += e
	 		s += batch_size
	 		if j%100==0:
	 			a = sess.run(training_accuracy, feed_dict={X:X_batch,Y:Y_batch})
	 			print("Training accuracy percent now: ",a*100)
	 	print("The total loss for EPOCH {0} is {1}".format(i+1,epoch_loss))
	
	#Testing the model

	num_batches = int(len(xi)/batch_size)
	overall_correct_preds = 0
	s = 0
	for k in range(num_batches):# you gotta create your own batches
		X_batch, Y_batch = xi[s : s + batch_size], xo[s : s + batch_size]
		X_batch = X_batch.reshape((batch_size,time_steps, input_size))
		a = sess.run(training_accuracy, feed_dict={X:X_batch, Y:Y_batch}) 		
		overall_correct_preds += a
		s += batch_size
		print("Test {0} Batch Accuracy:{1}".format(k,a*100))

	print("Accuracy of Model = {0}".format((overall_correct_preds/num_batches)*100))
	writer.close()
	testIt = [[[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]] #shape(?,20,1) aka (?,time_steps,input_size)
	print(sess.run(prediction,feed_dict={X: testIt}))
	print("Sum of 1's in sequence is: ",sess.run(tf.argmax(prediction,1),feed_dict={X: testIt}))	
