from bs4 import BeautifulSoup


from bs4 import BeautifulSoup
from bs4 import element

import pandas as pd
import codecs


class Parser(object):
    def __init__(self):
        self.contents = list()
    def parse(self,str):
        soup = BeautifulSoup(str)
        root = soup.contents
        self.findIter(root)

        return soup

    def findIter(self,tags):
        for tag in tags:
            if(type(tag) is element.Tag):
                self.findIter(tag)
            else:
                if(tag!=' ' and tag!= ' '):
                    self.contents.append(tag)

if __name__ == "__main__":


    f= open('E:\\BaiduYunDownload\\News_info_train.txt',encoding='utf-8',errors='ignore')
    # f= open('E:\\BaiduYunDownload\\News_info_train_example100.txt',encoding='utf-8',errors='ignore')
    # f2= open('E:\\BaiduYunDownload\\News_pic_label_train.txt',encoding='utf-8',errors='ignore')
    # htmls= list()



    htmls= dict()
    for line in f.readlines():
        temp =line.replace(line.split("\t")[0],"")
        htmls[line.split("\t")[0]]=temp[1:]
    f.close

    header = ['id', 'label', 'pic', 'content']
    datas = pd.read_csv("E:\\BaiduYunDownload\\News_pic_label_train.txt", sep='\t', names=header,index_col=None)
    # datas = pd.read_csv("E:\\BaiduYunDownload\\News_pic_label_train_example100.txt", sep='\t', names=header,index_col=None)
    label2 = datas.ix[datas.label == 2]
    label0 = datas.ix[datas.label == 0]

    print(1)
    '''
    data preprocess.1
    '''
    comments=dict()
    for index in htmls.keys():
        parser = Parser()
        parser.parse(htmls[index].split("\t")[0])
        # todo data clean
        line =str()
        for value in parser.contents:
            line+=value
        comments[index] = line
    '''
    data preprocess.2
    '''

        # datas[index]=line+"__label__"+ datas.ix[datas.id ==index]
    print(2)

    train_raw = list()
    for row in label2.id:
        try:
            line = comments[row] + '__label__' + str(2)
            train_raw.append(line)
        except:
            print()
    for row in label0.id:
        try:
            line = comments[row] + '__label__' + str(0)
            train_raw.append(line)
        except:
            print()
            # vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label
    target_file_path = 'e:/grossAd/src/data/comments2.txt'
    target_object = codecs.open(target_file_path, mode='a', encoding='utf-8')

    for i, line in enumerate(train_raw):
        target_object.write(line)
        target_object.write("\n")
    target_object.close()
    print(3)