#https://arxiv.org/abs/1503.08895 MemN2N

import numpy as np 
import re
import os
from collections import deque


def data_get(data_path, data_num=1, dataset='train', memory_capacity=50): #데이터 읽어서 패딩, 단어->숫자화 진행
	filename = list(os.walk(data_path))[0][2]
	
	result = []
	for files in filename:
		if (dataset in files) and (data_num == int(files.split('_')[0][2:])): # train or test
			
			with open(data_path+files, 'r') as o:	
				for i in o:
					if i.split()[0] == '1':
						story = deque(maxlen=memory_capacity) # maximum 50개 문장

					i = re.sub("[^a-zA-Z?, ]+", '', i).lower()[1:] #alphabet, '?', ',', ' '만 남기고 나머지 제거, 소문자화, 첫공백 제거(slicing)

					if '?' in i: #question 문장 만나면.	
						i = i.split("?") 
						question = i[0].split() # ex) ['is','bill','in','the','room']
						answer = i[1].strip() # ex) 'no'
						# ',' 기준으로 분할하고 정렬해서 다시 합치는 이유는 n,e e,n 처럼 같은 의미를 다르게 표현하는 경우가 있기 때문임.
						answer = [','.join(sorted(answer.split(',')))] # ex) ['no']

						sqa = [list(story.copy()), question.copy(), answer.copy()] #Story Question Answer
						result.append(sqa)

					else:
						i = i.split() # ex) ['mary', 'is', 'in', 'the', 'school']
						story.append(i)
	
	return result



def get_word_dict_and_maximum_word_in_sentence(dataset):
	word_dict = {}
	rev_word_dict = {}
	maximum_word_in_sentence = 0

	count = 0

	for data in dataset: # (1번 파일 데이터 ~ 20번 파일 데이터) * 3  : train20개, valid20개, test20개.	
		for story, question, answer in data:
			
			### story ###
			for s_sentence in story:
				maximum_word_in_sentence = max(maximum_word_in_sentence, len(s_sentence))
				
				for word in s_sentence:
					if word not in word_dict:
						word_dict[word] = count
						rev_word_dict[count] = word
						count += 1
					

			### question ###
			maximum_word_in_sentence = max(maximum_word_in_sentence, len(question))			
			for word in question:
				if word not in word_dict:
					word_dict[word] = count
					rev_word_dict[count] = word
					count += 1	
				

			### answer ###
			word = answer[0]
			if word not in word_dict:
				word_dict[word] = count
				rev_word_dict[count] = word
				count += 1

	word_dict['pad'] = -1
	rev_word_dict[-1] = 'pad'

	return word_dict, rev_word_dict, maximum_word_in_sentence
		

def train_vali_split(data, vali_ratio):
	train = []
	vali = []

	for task_data in data:
		vali.append(task_data[:int(len(task_data)*vali_ratio)])
		train.append(task_data[int(len(task_data)*vali_ratio):])

	return train, vali


def data_to_vector(data, word_dict, maximum_word_in_sentence, memory_capacity=50, sentence_numbering=True):
	# sentence는 sentence number + maximum_word_in_sentence 만큼 패딩하자
	# 첫자리에 sentence number 붙이자.
	# 붙이는건 유효한 부분까지만.
	# 패딩되는 memory_capacity에는 sentence number 붙이지 말자.
	# embedding 할때는 sentence number부분만 따로 임베딩생성해서 하고, 임베딩 결과는 concat해서 쓰자.

	result = []
	for task_data in data:
		task = []
		for story, question, answer in task_data:
			sentence_number = np.arange(1, len(story)+1)

			### story ###
			s_vector = []
			if sentence_numbering == True:
				for number, s_sentence in enumerate(story):
					temp = [number]
					temp.extend( [word_dict[word] for word in s_sentence] )
					temp.extend([-1] * ((maximum_word_in_sentence+1)-len(temp)))
					s_vector.append(temp)
				s_vector.extend([[-1] * (maximum_word_in_sentence+1)] * (memory_capacity-len(s_vector)) )
			
			else:
				for s_sentence in story:
					temp = []
					temp.extend( [word_dict[word] for word in s_sentence] )
					temp.extend([-1] * ((maximum_word_in_sentence)-len(temp)))
					s_vector.append(temp)
				s_vector.extend([[-1] * (maximum_word_in_sentence)] * (memory_capacity-len(s_vector)) )	



			### question ###
			q_vector = [word_dict[word] for word in question]
			q_vector.extend([-1]* ((maximum_word_in_sentence)-len(q_vector)))
	


			### answer ###
			a_vector = [word_dict[answer[0]]]
			
			
			### task ###
			task.append([s_vector, q_vector, a_vector])

		### result ###
		result.append(task)
	
	return result



def merge_tasks(data):
	merge = []
	for task_data in data:
		merge.extend(task_data)
	return merge