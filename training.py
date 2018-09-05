#https://arxiv.org/abs/1503.08895 MemN2N
import MemN2N as mn
import babi_process as bp
import numpy as np
import tensorflow as tf # 1.4
import os

data_path = './tasks_1-20_v1-2.tar/tasks_1-20_v1-2/en-10k/' # version 1.2
#data_path = './tasks_1-20_v1.tar/en/'  # version 1.0
print("data_path", data_path)

saver_path = "./saver/"

hop = 3
memory_capacity = 50 #논문 4.2 Training Details. The capacity of memory is restricted to the most recent 50 sentences.
vali_ratio = 0.1 # 10% 논문 4.2
embedding_size = 50 # joint training은 50, independent training은 20
lr = 0.01

def data_read_and_preprocess(data_path, memory_capacity, vali_ratio):
	
	# data_read
	train, test = [], []
	for i in range(1, 21):
		train.append(bp.data_get(data_path, data_num=i, dataset='train', memory_capacity=memory_capacity))
		test.append(bp.data_get(data_path, data_num=i, dataset='test', memory_capacity=memory_capacity))

	# data_split
	train, vali = bp.train_vali_split(train, vali_ratio)
	
	# get_information
	word_dict, rev_word_dict, maximum_word_in_sentence = bp.get_word_dict_and_maximum_word_in_sentence(train+vali+test)

	# preprocess (vectorize)
	train = bp.data_to_vector(train, word_dict, maximum_word_in_sentence, memory_capacity)
	vali = bp.data_to_vector(vali, word_dict, maximum_word_in_sentence, memory_capacity)
	test = bp.data_to_vector(test, word_dict, maximum_word_in_sentence, memory_capacity)

	return train, vali, test, word_dict, rev_word_dict, maximum_word_in_sentence


def merge_tasks(data):
	return np.array(bp.merge_tasks(data))


def toNumpy(data, dtype=np.int32):
	return np.array(data.tolist(), dtype)


def train(model, data):
	batch_size = 32
	loss = 0
	
	np.random.shuffle(data)

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		#print(i+1, '/',  int(np.ceil(len(data)/batch_size)) )
		batch = data[batch_size * i: batch_size * (i + 1)]
		story = toNumpy(batch[:, 0], np.int32)
		question = toNumpy(batch[:, 1], np.int32)
		answer = toNumpy(batch[:, 2], np.int64).flatten()

		train_loss, _ = sess.run([model.cost, model.minimize], {model.story:story, model.question:question, model.answer:answer})
		loss += train_loss

	return loss/len(data)



def validation(model, data):
	batch_size = 128
	loss = 0

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		story = toNumpy(batch[:, 0], np.int32)
		question = toNumpy(batch[:, 1], np.int32)
		answer = toNumpy(batch[:, 2], np.int64).flatten()

		train_loss, _ = sess.run([model.cost, model.minimize], {model.story:story, model.question:question, model.answer:answer})
		loss += train_loss


	return loss/len(data)




def test(model, data):
	batch_size = 128
	correct = 0

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		story = toNumpy(batch[:, 0], np.int32)
		question = toNumpy(batch[:, 1], np.int32)
		answer = toNumpy(batch[:, 2], np.int64).flatten()

		check = sess.run(model.correct_check, {model.story:story, model.question:question, model.answer:answer})
		correct += check
		
		#check, pred = sess.run([model.correct_check, tf.argmax(model.pred, axis=1)], {model.sentence:sentence, model.question:question, model.y:y})
		#correct += check
		#print('target', y,rev_word_dict[y[0]], '\tpred', pred, rev_word_dict[pred[0]], '\tcorrect', correct, '\tbatch_epoch', i+1)
	return correct/len(data)




def run(model, merge_train, merge_vali, merge_test, task_test, restore=0):
	if not os.path.exists(saver_path):
		os.makedirs(saver_path)
	
	if restore != 0:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")

	for epoch in range(restore+1, 2000+1):
		# lr annealing
		if epoch <= 20 and epoch % 5 == 0:
			model.lr /= 2

		# train, vali, test
		train_loss = train(model, merge_train)
		vali_loss = validation(model, merge_vali)
		accuracy = test(model, merge_test)
		print("epoch:", epoch, "\ttrain_loss:", train_loss, "\tvali_loss:", vali_loss, "\taccuracy:", accuracy)
		
		# task test
		for index, task in enumerate(task_test):
			accuracy = test(model, np.array(task))
			print(index+1, accuracy)

		#weight save
		model.saver.save(sess, saver_path+str(epoch)+'.ckpt')


(task_train, task_vali, task_test,
	word_dict, rev_word_dict, maximum_word_in_sentence) = data_read_and_preprocess(data_path, memory_capacity, vali_ratio)
merge_train = merge_tasks(task_train)
del task_train
merge_vali = merge_tasks(task_vali)
del task_vali
merge_test = merge_tasks(task_test)


sess = tf.Session()
model = mn.MemN2N(sess, hop, maximum_word_in_sentence, len(word_dict), embedding_size, memory_capacity, lr=lr)

run(model, merge_train, merge_vali, merge_test, task_test)
'''
np.random.shuffle(merge_test)
batch = merge_test[0:2]
story = toNumpy(batch[:, 0], np.int32)
question = toNumpy(batch[:, 1], np.int32)
answer = toNumpy(batch[:, 2], np.int64).flatten()

a = sess.run(model.pred, {model.story:story, model.question:question, model.answer:answer})
print(a,'\n')
#print(b,'\n')
#print(c,'\n')
#print(d,'\n')

'''