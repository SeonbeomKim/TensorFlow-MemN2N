import numpy as np
import tensorflow as tf #1.4
import babi_process as bp
import MemN2N as mn
import os

path = './tasks_1-20_v1-2.tar/tasks_1-20_v1-2/en-10k/'
data_savepath = "./preprocess_data/"
test_each_tasks_path = 'test_set_each_tasks/'

weight_savepath = "./saver/"
hop = 3
memory_capacity = 50 #논문 4.2 Training Details. The capacity of memory is restricted to the most recent 50 sentences.
maximum_word_in_sentence = 11#11
vali_ratio = 0.1 # 10% 논문 4.2
embedding_size = 50 # joint training은 50, independent training은 20

def data_read():
	data = bp.babi()
	data.store_testset_each_tasks(path, memory_capacity, maximum_word_in_sentence, dataset='test') 
	
	test_set = {}
	for i in range(1,21):
		test = np.load(test_each_tasks_path+str(i)+'/'+str(i)+'.npy')
		test_set[i] = test
	
	word_dict = np.load(data_savepath+'word_dict.npy')
	rev_word_dict = np.load(data_savepath+'rev_word_dict.npy')

	return test_set, word_dict.item(), rev_word_dict.item()


def toNumpy(data, dtype=np.int32):
	return np.array(data.tolist(), dtype)


def test(model, data):
	batch_size = 128
	correct = 0

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		sentence = toNumpy(batch[:, :memory_capacity], np.int32)
		question = toNumpy(batch[:, -2], np.int32)
		y = toNumpy(batch[:, -1], np.int32).flatten()

		check = sess.run(model.correct_check, {model.sentence:sentence, model.question:question, model.y:y})
		correct += check

	print(len(data), correct)
	return correct/len(data)


def run(model, test_set, restore):
	model.saver.restore(sess, weight_savepath+str(restore)+".ckpt")

	#for i in range(1, 21):
	accuracy = test(model, test_set)
	#accuracy = test(model, test_set[i])
	print("data set:", i, "\taccuracy:", accuracy)
		
	

test_set, word_dict, rev_word_dict = data_read()
word_len = len(word_dict) #165  -1 ~ 163  -1은 pad

sess = tf.Session()

#model
model = mn.MemN2N(sess, hop, maximum_word_in_sentence, word_len, embedding_size, memory_capacity, lr=0.01)

#print(test_set)

testdata = []
for i in test_set:
	print(len(test_set[i]))
	testdata.extend(test_set[i])
testdata = np.array(testdata)

run(model, testdata, 138)
#run(model, test_set, 138)
