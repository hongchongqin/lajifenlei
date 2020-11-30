#功能为对导入的数据进行处理
import numpy as np
import re
import word2vec
import jieba

class w2v_wrapper:
     def __init__(self, file_path):
        # w2v_file = os.path.join(base_path,  "vectors_poem.bin")
        self.model = word2vec.load(file_path)
        if 'unknown' not  in self.model.vocab_hash:
            unknown_vec = np.random.uniform(-0.1, 0.1, size=128)
            self.model.vocab_hash['unknown'] = len(self.model.vocab)
            self.model.vectors = np.row_stack((self.model.vectors, unknown_vec))


def clean_str(string):
	#除SST外的所有数据集的标记化/字符串清理。
	#提取汉字、数字、字母
    string = re.sub(r"[^A-Za-z0-9(), !?\'\`]",  " ",  string)
    string = re.sub(r"\'s",  " \'s",  string)
    string = re.sub(r"\'ve",  " \'ve",  string)
    string = re.sub(r"n\'t",  " n\'t",  string)
    string = re.sub(r"\'re",  " \'re",  string)
    string = re.sub(r"\'d",  " \'d",  string)
    string = re.sub(r"\'ll",  " \'ll",  string)
    string = re.sub(r", ",  " ,  ",  string)
    string = re.sub(r"!",  " ! ",  string)
    string = re.sub(r"\(",  " \( ",  string)
    string = re.sub(r"\)",  " \) ",  string)
    string = re.sub(r"\?",  " \? ",  string)
    string = re.sub(r"\s{2, }",  " ",  string)
    return string.strip().lower()

#去除空值
def removezero( x,  y):
    nozero = np.nonzero(y)
    print('removezero', np.shape(nozero)[-1], len(y))

    if(np.shape(nozero)[-1] == len(y)):
        return np.array(x), np.array(y)

    y = np.array(y)[nozero]
    x = np.array(x)
    x = x[nozero]
    return x,  y

#获取文件的行数
def read_file_lines(filename, from_size, line_num):
    i = 0
    text = []
    end_num = from_size + line_num
    for line in open(filename):
        if(i >= from_size):
            text.append(line.strip())

        i += 1
        if i >= end_num:
            return text

    return text

def load_data_and_labels(filepath, max_size = -1):
	#从文件加载MR极性数据，将数据拆分为单词并生成标签。
	#返回拆分语句和标签。
    train_datas = []
    with open(filepath,  'r',  encoding='utf-8', errors='ignore') as f:
        train_datas = f.readlines()

    one_hot_labels = []
    x_datas = []
    for line in train_datas:
        line = line.strip()
        parts = line.split('\t', 1)
        if(len(parts[1].strip()) == 0):
            continue
		#jieba分词
        words = jieba.cut(parts[1])
        x_datas.append(' '.join(words))
        
        one_hot = [0, 0, 0, 0]#垃圾共4类
        one_hot[int(parts[0])] = 1
        one_hot_labels.append(one_hot)
        
    print (' data size = ' , len(train_datas))
    # Split by words
    # x_text = [clean_str(sent) for sent in x_text]
    return [x_datas,  np.array(one_hot_labels)]#返回拆分语句和标签。


def batch_iter(data,  batch_size,  num_epochs,  shuffle=True):
    
    #Generates a batch iterator for a dataset.
	#为数据集生成批处理迭代器
    
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
		#在每个epoch处洗牌数据
		#shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))#随机排序
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size,  data_size)

            # print('epoch = %d, batch_num = %d, start = %d, end_idx = %d' % (epoch, batch_num, start_index, end_index))
            yield shuffled_data[start_index:end_index]


def get_text_idx(text, vocab, max_document_length):
    text_array = np.zeros([len(text),  max_document_length], dtype=np.int32)

    for i, x in  enumerate(text):
        words = x.split(" ")
        for j,  w in enumerate(words):
            if w in vocab:
                text_array[i,  j] = vocab[w]
            else :
                text_array[i,  j] = vocab['unknown']

    return text_array


if __name__ == "__main__":
    x_text,  y = load_data_and_labels('./data/data.txt')
    print (len(x_text))
