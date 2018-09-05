#https://arxiv.org/abs/1503.08895 MemN2N
import tensorflow as tf
import numpy as np
import os

class MemN2N:
	def __init__(self, sess, hop, maximum_word_in_sentence, word_len, embedding_size, memory_capacity, lr=0.01):
		self.hop = hop
		self.maximum_word_in_sentence = maximum_word_in_sentence # 11
		self.word_len = word_len #padding 포함 # 165    -1:pad
		self.embedding_size = embedding_size # 50
		self.memory_capacity = memory_capacity #50
		self.lr = lr
		self.clip_norm = 40.0

		with tf.name_scope("placeholder"):	
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
		activation = None#tf.nn.relu 

		#Position Encoding for word ordering
		PE =  self.position_encoding_l_value() # [maximum_word_in_sentence, embedding_size]
		#PE = tf.Variable(tf.random_normal([self.maximum_word_in_sentence, self.embedding_size], mean=0, stddev=0.1))
		
		#Temporal Encoding for sentence ordering	
		#TE_variable = tf.Variable(tf.random_normal([self.memory_capacity, self.embedding_size], mean=0, stddev=0.1)) # [memory_capacity, embedding_size]
		TE_variable_m = tf.Variable(tf.random_normal([self.memory_capacity, self.embedding_size], mean=0, stddev=0.1)) # [memory_capacity, embedding_size]
		TE_variable_c = tf.Variable(tf.random_normal([self.memory_capacity, self.embedding_size], mean=0, stddev=0.1)) # [memory_capacity, embedding_size]
		TE_mask = tf.cast(tf.equal(self.story, -1), tf.float32) # -1 is pad value, if nopad:0.0, pad:1.0  # [N, memory_capacity, maximum_word_in_sentence]
		TE_mask = tf.reduce_mean(TE_mask, axis=-1) # if 문장의 모든 단어가 패딩이라면: 1.0, else: [0.0,1.0) # [N, memory_capacity]
		TE_mask = tf.cast(TE_mask < 1.0, tf.float32) # if 문장의 모든 단어가 패딩이라면: 0.0, else 1.0 # [N, memory_capacity]
		TE_mask = tf.expand_dims(TE_mask, dim=-1) # [N, memory_capacity, 1]
		#TE = TE_variable * TE_mask # [N, memory_capacity, embedding_size]
		TE_m = TE_variable_m * TE_mask # [N, memory_capacity, embedding_size]
		TE_c = TE_variable_c * TE_mask # [N, memory_capacity, embedding_size]
		
		for epoch in range(self.hop):			
		
			if epoch == 0:
				# u(question)
				embedding_A_B = tf.Variable(tf.random_normal([self.word_len-1, self.embedding_size], mean=0, stddev=0.1))
				u = tf.nn.embedding_lookup(embedding_A_B, self.question) # [N, maximum_word_in_sentence, embedding_size]
				u = u * PE # [N, maximum_word_in_sentence, embedding_size]
				u = tf.reduce_sum(u, axis=-2) # [N, embedding_size]
				
				# m(story in)
				m = tf.nn.embedding_lookup(embedding_A_B, self.story) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				m = m * PE # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				m = tf.reduce_sum(m, axis=-2) # [N, memory_capacity, embedding_size]
				m += TE_m # [N, memory_capacity, embedding_size]
				
				# c(story out)
				embedding_C_A = tf.Variable(tf.random_normal([self.word_len-1, self.embedding_size], mean=0, stddev=0.1))
				c = tf.nn.embedding_lookup(embedding_C_A, self.story) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				c = c * PE # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				c = tf.reduce_sum(c, axis=-2) # [N, memory_capacity, embedding_size]
				c += TE_c # [N, memory_capacity, embedding_size]
				
				# p(attention)
				reshaped_u = tf.reshape(u, (-1, self.embedding_size, 1)) # [N, embedding_size, 1]
				p = tf.matmul(m, reshaped_u) # [N, memory_capacity, 1]
				p = tf.nn.softmax(p, dim=1) # [N, memory_capacity, 1]

				# o (weighted sum p,c)
				o = tf.reduce_sum(p*c, axis=1) # [N, embedding_size]

				# new u
				u = u + o # [N, embedding_size]
				if activation is not None:
					u = activation(u)

			else:
				# m(story in)
				m = tf.nn.embedding_lookup(embedding_C_A, self.story) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				m = m * PE # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				m = tf.reduce_sum(m, axis=-2) # [N, memory_capacity, embedding_size]
				m += TE_m # [N, memory_capacity, embedding_size]
			
				# c(story out)
				embedding_C_A = tf.Variable(tf.random_normal([self.word_len-1, self.embedding_size], mean=0, stddev=0.1))
				c = tf.nn.embedding_lookup(embedding_C_A, self.story) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				c = c * PE # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
				c = tf.reduce_sum(c, axis=-2) # [N, memory_capacity, embedding_size]
				c += TE_c # [N, memory_capacity, embedding_size]
								
				# p(attention)
				reshaped_u = tf.reshape(u, (-1, self.embedding_size, 1)) # [N, embedding_size, 1]
				p = tf.matmul(m, reshaped_u) # [N, memory_capacity, 1]
				p = tf.nn.softmax(p, dim=1) # [N, memory_capacity, 1]
				
				# o (weighted sum p,c)
				o = tf.reduce_sum(p*c, axis=1) # [N, embedding_size]

				# new u
				u = u + o # [N, embedding_size]
				if activation is not None:
					u = activation(u)

		predict = tf.matmul(u, tf.transpose(embedding_C_A)) # [N, word_len-1]
		return predict

	
	def position_encoding_l_value(self): #PE
		#embedding: [N, memory_capacity, maximum_word_in_sentence, embedding_size]		
		J = self.maximum_word_in_sentence #11
		d = self.embedding_size #50
		
		l = np.zeros([J, d]) # [maximum_word_in_sentence, embedding_size]

		for j in range(1, J+1): # 1 ~ 11 sentence
			for k in range(1, d+1):
				l[j-1, k-1] = (1-j/J) - (k/d)*(1-2*j/J) 
	
		return l #element wise product
		