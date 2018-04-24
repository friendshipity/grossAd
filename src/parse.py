from bs4 import BeautifulSoup


from bs4 import BeautifulSoup
from bs4 import element

import pandas as pd

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
    # f= open('E:\\BaiduYunDownload\\News_info_train.txt',encoding='utf-8',errors='ignore')
    f= open('E:\\BaiduYunDownload\\News_info_train_example100.txt',encoding='utf-8',errors='ignore')
    f2= open('E:\\BaiduYunDownload\\News_pic_label_train.txt',encoding='utf-8',errors='ignore')
    # htmls= list()

    htmls= dict()
    for line in f.readlines():
        temp =line.replace(line.split("\t")[0],"")
        htmls[line.split("\t")[0]]=temp[1:]
    f.close

    header = ['id', 'label', 'pic', 'content']
    datas = pd.read_csv("E:\\BaiduYunDownload\\News_pic_label_train.txt", sep='\t', names=header,index_col=None)
    lable2 = datas.ix[datas.label == 2]
    print(1)

    for index in htmls.keys():
        parser = Parser()
        parser.parse(htmls[index].split("\t")[0])
        # todo data clean
        line =str()
        for value in parser.contents:
            line+=value
        datas[index]=line
    print()

    '''
    data preprocess
    '''
    # for value in


    print(1)
