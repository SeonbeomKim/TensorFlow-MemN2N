#https://arxiv.org/abs/1503.08895 MemN2N

import babi_process as bp

path = './tasks_1-20_v1-2.tar/tasks_1-20_v1-2/en-10k/'
savepath = "./"
memory_capacity = 50 #논문 4.2 Training Details. The capacity of memory is restricted to the most recent 50 sentences.
maximum_word_in_sentence = 11#11
vali_ratio = 0.1 # 10% 논문 4.2

data = bp.babi()
data.preprocess_and_save(path, savepath, memory_capacity, maximum_word_in_sentence, vali_ratio)
train, vali, test = data.npload(savepath)
print(test[0])
