# coding=utf-8
from __future__ import unicode_literals
from defend import Defend
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import tokenize
from keras.utils.np_utils import to_categorical
import operator

DATA_PATH = 'data'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'Defend_model.h5'
EMBEDDINGS_PATH = 'saved_models/glove.6B.100d.txt'


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


if __name__ == '__main__':
    # dataset used for training
    platform = 'goss_fake'
    # platform = 'politifact'
    data_train = pd.read_csv( platform + '.tsv', sep='\t')
    VALIDATION_SPLIT = 0.25
    contents = []
    labels = []
    texts = []
    ids = []

    for idx in range(data_train.content.shape[0]): #request中有content
        text = BeautifulSoup(data_train.content[idx], features="html5lib") #构造BS，解析对象是data_train.cotent[idx],解析器是html5lib
        text = clean_str(text.get_text().encode('ascii', 'ignore')) # 用ascii来解码，忽略UnicodeEncodeError错误
        texts.append(text) #加入texts
        sentences = tokenize.sent_tokenize(text) #按句子分割 英文短句符+空格
        '''
        import nltk
        nltk.download('punkt')
        '''
        contents.append(sentences) #加入contents
        ids.append(data_train.id[idx]) #加入已经遍历过的id

        labels.append(data_train.label[idx]) #加入标签

    labels = np.asarray(labels)  #不会占用新的内存来存储labels，但可能会发生改变，将labels变为ndarray
    labels = to_categorical(labels) #将数据转化为二进制矩阵,数据里面是数字

    # load user comments
    comments = []
    comments_text = []
    comments_train = pd.read_csv('data/' + platform + '_comment_no_ignore.tsv', sep='\t') 
    print (comments_train.shape) #矩阵的长款

    content_ids = set(ids) #形成集合

    for idx in range(comments_train.comment.shape[0]):
        if comments_train.id[idx] in  content_ids:
            com_text = BeautifulSoup(comments_train.comment[idx], features="html5lib")
            com_text = clean_str(com_text.get_text().encode('ascii', 'ignore')) 
            '''
            html5解析，ascii编码
            '''
            tmp_comments = []
            for ct in com_text.split('::'): #以：：为分隔符
                tmp_comments.append(ct)   #评论内部连接
            comments.append(tmp_comments) #各个评论连接
            comments_text.extend(tmp_comments) 

    id_train, id_test, x_train, x_val, y_train, y_val, c_train, c_val = train_test_split(ids,contents, labels, comments,
                                                                      test_size=VALIDATION_SPLIT, random_state=42,
                                                                      stratify=labels)

    # Train and save the model
    SAVED_MODEL_FILENAME = platform + '_Defend_model.h5'
    h = Defend(platform)
    h.train(x_train, y_train, c_train, c_val, x_val, y_val,
            batch_size=20,
            epochs=30,
            embeddings_path='./glove.6B.100d.txt',
            saved_model_dir=SAVED_MODEL_DIR,
            saved_model_filename=SAVED_MODEL_FILENAME)

    h.load_weights(saved_model_dir = SAVED_MODEL_DIR, saved_model_filename = SAVED_MODEL_FILENAME)

    # Get the attention weights for sentences in the news contents as well as comments
    activation_maps = h.activation_maps(x_val, c_val)
