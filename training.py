import numpy as np
import tensorflow as tf #1.4
import babi_process as bp
import MemN2N as mn
import os

path = './tasks_1-20_v1-2.tar/tasks_1-20_v1-2/en-10k/'
data_savepath = "./preprocess_data/"

weight_savepath = "./saver/"
hop = 3
memory_capacity = 50 #논문 4.2 Training Details. The capacity of memory is restricted to the most recent 50 sentences.
maximum_word_in_sentence = 11#11
vali_ratio = 0.1 # 10% 논문 4.2
embedding_size = 50 # joint training은 50, independent training은 20

def data_read():
	data = bp.babi()
	data.preprocess_and_save(path, data_savepath, memory_capacity, maximum_word_in_sentence, vali_ratio)
	train, vali, test, word_dict, rev_word_dict = data.npload(data_savepath)
	return train, vali, test, word_dict.item(), rev_word_dict.item()   # numpy load 했기 때문에 dictionary 쓰려면 item() 해줘야함.

def toNumpy(data, dtype=np.int32):
	return np.array(data.tolist(), dtype)

def train(model, data):
	batch_size = 32
	loss = 0
	np.random.shuffle(data)

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		#print(i+1, '/',  int(np.ceil(len(data)/batch_size)) )
		batch = data[batch_size * i: batch_size * (i + 1)]
		sentence = toNumpy(batch[:, :memory_capacity], np.int32) # [N, memory_capacity, maximum_word_in_sentence]
		question = toNumpy(batch[:, -2], np.int32) # [N, maximum_word_in_sentence]
		y = toNumpy(batch[:, -1], np.int32).flatten() # [N]

		train_loss, _ = sess.run([model.cost, model.minimize], {model.sentence:sentence, model.question:question, model.y:y})
		loss += train_loss

	return loss/len(data)


def validation(model, data):
	batch_size = 128
	loss = 0
	
	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		sentence = toNumpy(batch[:, :memory_capacity], np.int32)
		question = toNumpy(batch[:, -2], np.int32)
		y = toNumpy(batch[:, -1], np.int32).flatten()
	
		vali_loss = sess.run(model.cost, {model.sentence:sentence, model.question:question, model.y:y})
		loss += vali_loss

	return loss/len(data)


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

	return correct/len(data)


def run(model, train_set, vali_set, test_set):
	if not os.path.exists(weight_savepath):
		os.makedirs(weight_savepath)

	for epoch in range(1, 20+1):
		if epoch <= 20 and epoch % 5 == 0:
			model.lr /= 2

		train_loss = train(model, train_set)
		vali_loss = validation(model, vali_set)
		accuracy = test(model, test_set)

		print("epoch:", epoch, "\ttrain_loss:", train_loss, "\tvali_loss:", vali_loss, "\taccuracy:", accuracy)
		
		#weight save
		save_path = model.saver.save(sess, weight_savepath+str(epoch)+'.ckpt')


train_set, vali_set, test_set, word_dict, rev_word_dict = data_read()
word_len = len(word_dict) #165  -1 ~ 163  -1은 pad

sess = tf.Session()

#model
model = mn.MemN2N(sess, hop, maximum_word_in_sentence, word_len, embedding_size, memory_capacity, lr=0.01)

run(model, train_set, vali_set, test_set)

'''
test = train[0:2, :-2]
test = toNumpy(test, np.int32)

question = train[0:2, -2]
question = toNumpy(question, np.int32)

print(train[0:2, -1])
#x = sess.run(model.x, {model.x:np.array([test])})


#emb = (sess.run(model.embedding_lookup_A, {model.x:np.array([test])}))
#emb = (sess.run(model.test, {model.x:test}))
#print(emb.shape)
#print(sess.run(model.embedding_lookup_A, {model.x:test}))

#print(train[0][:-1])

#run(model, train_set, vali_set, test_set)
'''