#https://arxiv.org/abs/1503.08895 MemN2N
import tensorflow as tf
import numpy as np
import os

class MemN2N:
	def __init__(self, sess, hop, maximum_word_in_sentence, word_len, embedding_size, memory_capacity, sentence_numbering=True, lr=0.01):
		self.hop = hop
		self.maximum_word_in_sentence = maximum_word_in_sentence # 11
		self.word_len = word_len #padding 포함 # 165    -1:pad
		self.embedding_size = embedding_size # 50
		self.memory_capacity = memory_capacity #50
		self.sentence_numbering = sentence_numbering
		self.lr = lr
		self.clip_norm = 40.0

		with tf.name_scope("placeholder"):
			if sentence_numbering == True:								# sentence numbering 1칸 필요.
				self.story = tf.placeholder(tf.int32, [None, memory_capacity, maximum_word_in_sentence+1]) # [N, memory_capacity, maximum_word_in_sentence+1]
			else:
				self.story = tf.placeholder(tf.int32, [None, memory_capacity, maximum_word_in_sentence]) # [N, memory_capacity, maximum_word_in_sentence]
			self.question = tf.placeholder(tf.int32, [None, maximum_word_in_sentence]) # [N, maximum_word_in_sentence]
			self.answer = tf.placeholder(tf.int64, [None]) # [N]
			self.one_hot_answer = tf.one_hot(self.answer, depth=word_len-1) #[N, word_len-1]
		

		with tf.name_scope('predict'):
			self.pred = self.predict_using_adjacent_layer()

		with tf.name_scope('cost'): # cost is not averaged over a batch! in paper
			self.cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_answer, logits=self.pred))

		with tf.name_scope('optimizer'): #10k dataset은 5epoch마다 lr/2
			#optimizer = tf.train.AdamOptimizer(self.lr) #4.2 Training Details. Momentum or weight decay 안씀.
			optimizer = tf.train.GradientDescentOptimizer(self.lr) #4.2 Training Details. Momentum or weight decay 안씀.
			#https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
			grads_and_vars = optimizer.compute_gradients(self.cost)
			#https://www.tensorflow.org/api_docs/python/tf/clip_by_norm
			clip_grads_and_vars = [(tf.clip_by_norm(gv[0], self.clip_norm), gv[1]) for gv in grads_and_vars]
			self.minimize = optimizer.apply_gradients(clip_grads_and_vars)
			#self.minimize = optimizer.minimize(self.cost)

		with tf.name_scope('correct_check'):
			self.correct_check = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.pred, axis=1), self.answer), tf.int32))

		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)

		sess.run(tf.global_variables_initializer())



	def predict_using_adjacent_layer(self):
		if self.sentence_numbering == True:
			embedding_numbering = tf.Variable(tf.random_normal([self.memory_capacity, self.embedding_size], mean=0, stddev=0.1))

		for epoch in range(self.hop):
			if epoch == 0:
				# u(question), m(story in)
				embedding_A_B = tf.Variable(tf.random_normal([self.word_len-1, self.embedding_size], mean=0, stddev=0.1))
				u = tf.nn.embedding_lookup(embedding_A_B, self.question) # [N, maximum_word_in_sentence, embedding_size]
				u = tf.reduce_sum(u, axis=-2) # [N, embedding_size]

				if self.sentence_numbering == True:
					m_numbering = tf.nn.embedding_lookup(embedding_numbering, self.story[:, :, 0:1]) # [N, memory_capacity, 1, embedding_size]
					m_words = tf.nn.embedding_lookup(embedding_A_B, self.story[:, :, 1:]) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
					m = tf.concat((m_numbering, m_words), axis=2) # [N, memory_capacity, maximum_word_in_sentence+1, embedding_size]
				else:
					m = tf.nn.embedding_lookup(embedding_A_B, self.story) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				m = tf.reduce_sum(m, axis=-2) # [N, memory_capacity, embedding_size]
				
				# c(story out)
				embedding_C_A = tf.Variable(tf.random_normal([self.word_len-1, self.embedding_size], mean=0, stddev=0.1))
				if self.sentence_numbering == True:
					c_numbering = tf.nn.embedding_lookup(embedding_numbering, self.story[:, :, 0:1]) # [N, memory_capacity, 1, embedding_size]
					c_words = tf.nn.embedding_lookup(embedding_C_A, self.story[:, :, 1:]) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
					c = tf.concat((c_numbering, c_words), axis=2) # [N, memory_capacity, maximum_word_in_sentence+1, embedding_size]
				else:
					c = tf.nn.embedding_lookup(embedding_C_A, self.story) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				c = tf.reduce_sum(c, axis=-2) # [N, memory_capacity, embedding_size]
			
				# p(attention)
				reshaped_u = tf.reshape(u, (-1, self.embedding_size, 1)) # [N, embedding_size, 1]
				p = tf.matmul(m, reshaped_u) # [N, memory_capacity, 1]
				p = tf.nn.softmax(p, dim=1) # [N, memory_capacity, 1]

				# o (weighted sum p,c)
				o = tf.reduce_sum(p*c, axis=1) # [N, embedding_size]

				# new u
				u = u + o # [N, embedding_size]
				#u = tf.nn.relu(u)

			else:
				# m(story in)
				if self.sentence_numbering == True:
					m_numbering = tf.nn.embedding_lookup(embedding_numbering, self.story[:, :, 0:1]) # [N, memory_capacity, 1, embedding_size]
					m_words = tf.nn.embedding_lookup(embedding_C_A, self.story[:, :, 1:]) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
					m = tf.concat((m_numbering, m_words), axis=2) # [N, memory_capacity, maximum_word_in_sentence+1, embedding_size]
				else:
					m = tf.nn.embedding_lookup(embedding_C_A, self.story) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				m = tf.reduce_sum(m, axis=-2) # [N, memory_capacity, embedding_size]
				
				# c(story out)
				embedding_C_A = tf.Variable(tf.random_normal([self.word_len-1, self.embedding_size], mean=0, stddev=0.1))
				if self.sentence_numbering == True:
					c_numbering = tf.nn.embedding_lookup(embedding_numbering, self.story[:, :, 0:1]) # [N, memory_capacity, 1, embedding_size]
					c_words = tf.nn.embedding_lookup(embedding_C_A, self.story[:, :, 1:]) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
					c = tf.concat((c_numbering, c_words), axis=2) # [N, memory_capacity, maximum_word_in_sentence+1, embedding_size]					
				else:
					c = tf.nn.embedding_lookup(embedding_C_A, self.story) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				c = tf.reduce_sum(c, axis=-2) # [N, memory_capacity, embedding_size]
								
				# p(attention)
				reshaped_u = tf.reshape(u, (-1, self.embedding_size, 1)) # [N, embedding_size, 1]
				p = tf.matmul(m, reshaped_u) # [N, memory_capacity, 1]
				p = tf.nn.softmax(p, dim=1) # [N, memory_capacity, 1]
				
				# o (weighted sum p,c)
				o = tf.reduce_sum(p*c, axis=1) # [N, embedding_size]

				# new u
				u = u + o # [N, embedding_size]
				#u = tf.nn.relu(u)
				
		predict = tf.matmul(u, tf.transpose(embedding_C_A)) # [N, word_len-1]
		return predict

	
		
	def position_encoding(self, l_value, embedding):
		pe = embedding * l_value # element wise product [N, memory_capacity, maximum_word_in_sentence, embedding_size]
		return pe



	def position_encoding_l_value(self, mode='story'): #PE
		#embedding: [N, memory_capacity, maximum_word_in_sentence, embedding_size]
	
		
		J = self.maximum_word_in_sentence #11
		d = self.embedding_size #50

		if mode == 'story':
			print("changed PE")
			K = self.memory_capacity #50
			l = np.zeros([K, J, d]) # [memory_capacity, maximum_word_in_sentence, embedding_size]
			
			for j in range(1, J+1): # 1 ~ 11. 1문장당 11단어.
				l[:, j-1, :] = ((1-j)/J) - (K/d)*((1-2*j)/J)  
			'''
			for k in range(1, K+1): # 1 ~ 50. 총 50문장
				for j in range(1, J+1): # 1 ~ 11. 1문장당 11단어.
					l[k-1, j-1, :] = ((1-j)/J) - (k/d)*((1-2*j)/J)  
			'''
			return l #element wise product
		

		elif mode == 'question':
			#K = 1
			K = self.memory_capacity
			l = np.zeros([J, d]) # [maximum_word_in_sentence, embedding_size]

			for j in range(1, J+1): # 1 ~ 11. 1문장당 11단어.
				l[j-1, :] = ((1-j)/J) - (K/d)*((1-2*j)/J)  

			return l #element wise product
		