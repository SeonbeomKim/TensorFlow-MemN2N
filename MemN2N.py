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

		with tf.name_scope("placeholder"):
			self.sentence = tf.placeholder(tf.int32, [None, memory_capacity, maximum_word_in_sentence]) # [N, memory_capacity, maximum_word_in_sentence]
			self.question = tf.placeholder(tf.int32, [None, maximum_word_in_sentence]) # [N, maximum_word_in_sentence]
			self.y = tf.placeholder(tf.int32, [None]) # [N]
			self.one_hot_y = tf.one_hot(self.y, depth=word_len-1) #[N, word_len-1]
		
		with tf.name_scope('predict'):
			self.pred = self.predict_using_adjacent_layer(hop)

		with tf.name_scope('cost'): # cost is not averaged over a batch! in paper
			self.cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.pred))

		with tf.name_scope('optimizer'): #10k dataset은 5epoch마다 lr/2
			optimizer = tf.train.GradientDescentOptimizer(self.lr) #4.2 Training Details. Momentum or weight decay 안씀.
			self.minimize = optimizer.minimize(self.cost)

		with tf.name_scope('correct_check'):
			self.correct_check = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.pred, axis=1, output_type=tf.int32), self.y), tf.int32))

		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)

		sess.run(tf.global_variables_initializer())



	def predict_using_adjacent_layer(self, hop):
		l_value_sentence = self.position_encoding_l_value(mode='sentence') # [memory_capacity, maximum_word_in_sentence, embedding_size]
		l_value_question = self.position_encoding_l_value(mode='question') # [memory_capacity, maximum_word_in_sentence, embedding_size]

		embedding_A_B = tf.Variable(tf.random_uniform([self.word_len-1, self.embedding_size], -1., 1.))
		u_question = tf.nn.embedding_lookup(embedding_A_B, self.question) # [N, maximum_word_in_sentence, embedding_size] 
		u_question = self.position_encoding(l_value_question, u_question) # [N, embedding_size]

		m_sentence = tf.nn.embedding_lookup(embedding_A_B, self.sentence) # [N, memory_capacity, maximum_word_in_sentence, embedding_size]
		m_sentence = self.position_encoding(l_value_sentence, m_sentence) # [N, memory_capacity, embedding_size]


		for i in range(hop):
			#p	
			p_score = tf.expand_dims(u_question, dim=1) * m_sentence # [N, memory_capacity, embedding_size] broadcast 때문에 동작.
			p_dot = tf.reduce_sum(p_score, axis=-1) # [N, memory_capacity]
			p_align = tf.nn.softmax(p_dot, dim=1) # [N, memory_capacity]
			p_align = tf.expand_dims(p_align, dim=-1) # [N, memory_capacity, 1]
		
			#embedding for c_i_minus_1 and a_i and W.T
			embedding_C_A = tf.Variable(tf.random_uniform([self.word_len-1, self.embedding_size], -1., 1.))
			c_sentence_i_minus_1 = tf.nn.embedding_lookup(embedding_C_A, self.sentence)
			c_sentence_i_minus_1 = self.position_encoding(l_value_sentence, c_sentence_i_minus_1) # [N, memory_capacity, embedding_size]

			#o 
			o = p_align * c_sentence_i_minus_1 # [N, memory_capacity, embedding_size]
			o = tf.reduce_sum(o, axis=1) # [N, embedding_size]
			
			#new u_question
			u_question += o # [N, embedding_size]

			#new a_i
			if i != hop-1:
				m_sentence = tf.nn.embedding_lookup(embedding_C_A, self.sentence)
				m_sentence = self.position_encoding(l_value_sentence, m_sentence) # [N, memory_capacity, embedding_size]

		#a_hat
		predict = tf.matmul(u_question, tf.transpose(embedding_C_A))
		return predict # [N, word_len-1]


		
	def position_encoding(self, l_value, embedding):
		pe = embedding * l_value # element wise product [N, memory_capacity, maximum_word_in_sentence, embedding_size]
		pe = tf.reduce_sum(pe, axis=-2) # [N, memory_capacity, embedding_size]
		# question embedding의 경우 [N, maximum_word_in_sentence, embedding_size] shape을 갖기 때문에, 이것도 axis=-2 함.
		# 즉 sentence, question 따로 구현할 필요 없음.
		return pe



	def position_encoding_l_value(self, mode='sentence'): #PE
		#embedding: [N, memory_capacity, maximum_word_in_sentence, embedding_size]
		J = self.maximum_word_in_sentence #11
		d = self.embedding_size #50

		if mode == 'sentence':
			k = self.memory_capacity #50
			l = np.zeros([k, J, d]) # [memory_capacity, maximum_word_in_sentence, embedding_size]

			for k in range(1, k+1): # 1 ~ 50. 총 50문장
				for j in range(1, J+1): # 1 ~ 11. 1문장당 11단어.
					#이렇게 하면 j값에 의해 문장 내의 단어별로 순서 차이를 구별할 수 있고,
					#11단어의 집합인 1문장들은 k에 의해 차이가 생기니까 문제 순서를 구별할 수 있음.
					l[k-1, j-1, :] = ((1-j)/J) - (k/d)*((1-2*j)/J)  

			return l #element wise product
		

		elif mode == 'question':
			k = 1
			l = np.zeros([J, d]) # [maximum_word_in_sentence, embedding_size]

			for j in range(1, J+1): # 1 ~ 11. 1문장당 11단어.
				#이렇게 하면 j값에 의해 문장 내의 단어별로 순서 차이를 구별할 수 있고,
				#11단어의 집합인 1문장들은 k에 의해 차이가 생기니까 문제 순서를 구별할 수 있음.
				l[j-1, :] = ((1-j)/J) - (k/d)*((1-2*j)/J)  

			return l #element wise product
		


		


#with tf.device('/gpu:0'):





