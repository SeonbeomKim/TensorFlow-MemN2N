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
		if not os.path.exists(savepath):
			os.makedirs(savepath)

		elif os.path.exists(savepath+'train.npy') and os.path.exists(savepath+'vali.npy') \
			and os.path.exists(savepath+'test.npy') and os.path.exists(savepath+'word_dict.npy') \
			and	os.path.exists(savepath+'rev_word_dict.npy'):
			return None

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
		np.save(savepath+'word_dict', word_dict)
		np.save(savepath+'rev_word_dict', rev_word_dict)

	def npload(self, path):
		train = np.load(path+'train.npy')
		vali = np.load(path+'vali.npy')
		test = np.load(path+'test.npy')
		word_dict = np.load(path+'word_dict.npy')
		rev_word_dict = np.load(path+'rev_word_dict.npy')
		return train, vali, test, word_dict, rev_word_dict



	# test셋별로 따로 저장하는 코드
	def store_testset_each_tasks(self, path, memory_capacity, maximum_word_in_sentence, dataset='test'): #데이터 읽어서 패딩, 단어->숫자화 진행
		fileinfo = list(os.walk(path))[0]
		filename = fileinfo[2]

		# numbering(word->number) 을 위해 dictionary 생성
		word_dict, rev_word_dict = self.make_word_dict(path, filename)

		for files in filename:
			if dataset in files:
				data_num = files[2:].split('_')[0]
				data = []

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
				
				if not os.path.exists('test_set_each_tasks/'+data_num):
					os.makedirs('test_set_each_tasks/'+data_num)		
				np.save('test_set_each_tasks/'+data_num+'/'+data_num, data)

		return np.array(data)








