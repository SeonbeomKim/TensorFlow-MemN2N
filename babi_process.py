#https://arxiv.org/abs/1503.08895 MemN2N

import numpy as np 
import re
import os
from collections import deque


class babi:

	def make_word_dict(self, path, filename):
		word_dict = {}
		rev_word_dict = {}
		count = 0

		for files in filename:
			with open(path+files, 'r') as o:
				for i in o:
					i = re.sub("[^a-zA-Z?, ]+", '', i).lower()[1:] #alphabet, '?', ',', ' '만 남기고 나머지 제거, 소문자화, 첫공백 제거(slicing)
					i = re.sub("[?]+", ' ', i).split() # '?'를 ' '로 치환
					for word in i:
						if ',' in word:
							word = ','.join(sorted(word.split(',')))
						
						if word not in word_dict:
							word_dict[word] = count
							rev_word_dict[count] = word
							count += 1

		word_dict['pad'] = -1
		rev_word_dict[-1] = 'pad'

		return word_dict, rev_word_dict



	def data_read_pad_numbering(self, path, filename, word_dict, memory_capacity, maximum_word_in_sentence, dataset='train'): #데이터 읽어서 패딩, 단어->숫자화 진행
		data = []

		for files in filename:
			if dataset in files:
			
				with open(path+files, 'r') as o:	
					for i in o:
						if i.split()[0] == '1':
							query = deque(maxlen=memory_capacity) # maximum 50개 문장

						i = re.sub("[^a-zA-Z?, ]+", '', i).lower()[1:] #alphabet, '?', ',', ' '만 남기고 나머지 제거, 소문자화, 첫공백 제거(slicing)


						if '?' in i: #question 문장 만나면.	
							i = i.split("?") 
							question = i[0].split() # ex) ['is','bill','in','the','room']
							question = [word_dict[k] for k in question] # ex) [1, 5, 2, 3, 7]		
							question = np.pad(question, (0, maximum_word_in_sentence-len(question)), 
										'constant', constant_values=word_dict['pad']).tolist()
							# ex) [1, 5, 2, 3, 7, -1, -1, -1, -1, -1, -1]


							answer = i[1].strip() # ex) 'no'
							# ',' 기준으로 분할하고 정렬해서 다시 합치는 이유는 n,e e,n 처럼 같은 의미를 다르게 표현하는 경우가 있기 때문임.
							answer = ','.join(sorted(answer.split(','))) # ex) 'no'
							answer = [word_dict[answer]] # [8]

				
							query_pad = (list(query) + ( [[-1]*maximum_word_in_sentence]* (memory_capacity-len(query)) )
										 + [question] + [answer] )
							
							#print(query_pad)
							data.append(query_pad[:]) #query[:] 말고 query로 하면 주소가 넘어가서 그 후에 query가 바뀌는경우 data내부의 query도 바뀌어버림.

			
						else:
							i = i.split() # ex) ['mary', 'is', 'in', 'the', 'school']
							i = [word_dict[k] for k in i] # ex) [0, 1, 2, 3, 4]
							i = np.pad(i, (0, maximum_word_in_sentence-len(i)), 'constant', constant_values=word_dict['pad']).tolist()
							# ex) [0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1]   ==> -1 is 'pad'
							query.append(i)
						

		return np.array(data)



	def split_dataset(self, train, vali_ratio):
		np.random.shuffle(train)

		vali = train[:int(len(train)*vali_ratio)]
		train = train[int(len(train)*vali_ratio):]
		
		return train, vali



	def preprocess_and_save(self, path, savepath, memory_capacity, maximum_word_in_sentence, vali_ratio):
		fileinfo = list(os.walk(path))[0]
		filename = fileinfo[2]

		# numbering(word->number) 을 위해 dictionary 생성
		word_dict, rev_word_dict = self.make_word_dict(path, filename)

		# 데이터 read, pad, numbering 처리
		train = self.data_read_pad_numbering(path, filename, word_dict, memory_capacity, maximum_word_in_sentence, dataset='train')
		test = self.data_read_pad_numbering(path, filename, word_dict, memory_capacity, maximum_word_in_sentence, dataset='test')
		
		# train의 10%는 vali로 분리
		train, vali = self.split_dataset(train, vali_ratio)

		# save
		np.save(savepath+'train', train)
		np.save(savepath+'vali', vali)
		np.save(savepath+'test', test)


	def npload(self, path):
		train = np.load(path+'train.npy')
		vali = np.load(path+'vali.npy')
		test = np.load(path+'test.npy')
		return train, vali, test













'''

#path = './tasks_1-20_v1-2.tar/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt'
path = './tasks_1-20_v1-2.tar/tasks_1-20_v1-2/en-10k/'

savepath = "./"

fileinfo = list(os.walk(path))[0]
filename = fileinfo[2]

memory_capacity = 50 #논문 4.2 Training Details. The capacity of memory is restricted to the most recent 50 sentences.
maximum_word_in_sentence = 11#11
vali_ratio = 0.1 # 10% 논문 4.2



def read_data(path, filename):
	data = []
	query = []
	for files in filename:
		with open(path+files, 'r') as o:
			for i in o:
				if i.split()[0] == '1':
					query = []
				
				i = re.sub("[^a-zA-Z?, ]+", '', i).lower()[1:] #alphabet, '?', ',', ' '만 남기고 나머지 제거, 소문자화, 첫공백 제거(slicing)

				if '?' in i: #question 문장 만나면.	
					i = i.split("?")
					question = i[0].split()
					query.append(question)

					answer = i[1].strip()
					answer = ','.join(sorted(answer.split(',')))
					query.append([answer])

					data.append(query[:]) #query[:] 말고 query로 하면 주소가 넘어가서 그 후에 query가 바뀌는경우 data내부의 query도 바뀌어버림.
					
					query.pop(-1) # pop answer
					query.pop(-1) # pop question

				else:
					query.append(i.split())
				

	return data





def read_data_one_txt(path):
	data = []
	query = []
	with open(path, 'r') as o:
		for i in o:
			if i.split()[0] == '1':
				query = []
			
			i = re.sub("[^a-zA-Z?, ]+", '', i).lower()[1:] #alphabet, '?', ',', ' '만 남기고 나머지 제거, 소문자화, 첫공백 제거(slicing)

			if '?' in i: #question 문장 만나면.	
				i = i.split("?")
				question = i[0].split()
				query.append(question)

				answer = i[1].strip()
				answer = ','.join(sorted(answer.split(',')))
				query.append([answer])

				data.append(query[:]) #query[:] 말고 query로 하면 주소가 넘어가서 그 후에 query가 바뀌는경우 data내부의 query도 바뀌어버림.
				
				query.pop(-1) # pop answer
				query.pop(-1) # pop question

			else:
				query.append(i.split())
			

	return data


def embedding_test():
	import tensorflow as tf

	va = tf.Variable([[1.,1., 1.],[2.,2.,2.],[3.,3.,3.]])
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	a = [[-1,3], [1,2]]
	em = tf.nn.embedding_lookup(va, a) 
	print(sess.run(em)) # [ [[0,0,0], [0,0,0]], [[2,2,2], [3,3,3]] ]


def make_word_dict(path, filename):
	word_dict = {}
	rev_word_dict = {}
	count = 0

	for files in filename:
		with open(path+files, 'r') as o:
			for i in o:
				i = re.sub("[^a-zA-Z?, ]+", '', i).lower()[1:] #alphabet, '?', ',', ' '만 남기고 나머지 제거, 소문자화, 첫공백 제거(slicing)
				i = re.sub("[?]+", ' ', i).split() # '?'를 ' '로 치환
				for word in i:
					if ',' in word:
						word = ','.join(sorted(word.split(',')))
					
					if word not in word_dict:
						word_dict[word] = count
						rev_word_dict[count] = word
						count += 1

	word_dict['pad'] = -1
	rev_word_dict[-1] = 'pad'

	return word_dict, rev_word_dict



def data_read_pad_numbering(path, filename, word_dict, dataset='train'): #데이터 읽어서 패딩, 단어->숫자화 진행
	data = []

	for files in filename:
		if dataset in files:
		
			with open(path+files, 'r') as o:	
				for i in o:
					if i.split()[0] == '1':
						query = deque(maxlen=memory_capacity) # maximum 50개 문장

					i = re.sub("[^a-zA-Z?, ]+", '', i).lower()[1:] #alphabet, '?', ',', ' '만 남기고 나머지 제거, 소문자화, 첫공백 제거(slicing)


					if '?' in i: #question 문장 만나면.	
						i = i.split("?") 
						question = i[0].split() # ex) ['is','bill','in','the','room']
						question = [word_dict[k] for k in question] # ex) [1, 5, 2, 3, 7]		
						question = np.pad(question, (0, maximum_word_in_sentence-len(question)), 
									'constant', constant_values=word_dict['pad']).tolist()
						# ex) [1, 5, 2, 3, 7, -1, -1, -1, -1, -1, -1]


						answer = i[1].strip() # ex) 'no'
						# ',' 기준으로 분할하고 정렬해서 다시 합치는 이유는 n,e e,n 처럼 같은 의미를 다르게 표현하는 경우가 있기 때문임.
						answer = ','.join(sorted(answer.split(','))) # ex) 'no'
						answer = [word_dict[answer]] # [8]

			
						query_pad = (list(query) + ( [[-1]*maximum_word_in_sentence]* (memory_capacity-len(query)) )
									 + [question] + [answer] )
						
						#print(query_pad)
						data.append(query_pad[:]) #query[:] 말고 query로 하면 주소가 넘어가서 그 후에 query가 바뀌는경우 data내부의 query도 바뀌어버림.

		
					else:
						i = i.split() # ex) ['mary', 'is', 'in', 'the', 'school']
						i = [word_dict[k] for k in i] # ex) [0, 1, 2, 3, 4]
						i = np.pad(i, (0, maximum_word_in_sentence-len(i)), 'constant', constant_values=word_dict['pad']).tolist()
						# ex) [0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1]   ==> -1 is 'pad'
						query.append(i)
					

	return np.array(data)



def split_dataset(train):
	np.random.shuffle(train)

	vali = train[:int(len(train)*vali_ratio)]
	train = train[int(len(train)*vali_ratio):]
	
	return train, vali



def preprocess(path, filename):
	# numbering(word->number) 을 위해 dictionary 생성
	word_dict, rev_word_dict = make_word_dict(path, filename)

	# 데이터 read, pad, numbering 처리
	train = data_read_pad_numbering(path, filename, word_dict, dataset='train')
	test = data_read_pad_numbering(path, filename, word_dict, dataset='test')
	
	# train의 10%는 vali로 분리
	train, vali = split_dataset(train)

	# save
	np.save(savepath+'train', train)
	np.save(savepath+'vali', vali)
	np.save(savepath+'test', test)


def npload(path):
	train = np.load(path+'train.npy')
	vali = np.load(path+'vali.npy')
	test = np.load(path+'test.npy')
	return train, vali, test

#preprocess(path, filename)
train, vali, test = npload(savepath)
print(train[0])

'''

