from bs4 import BeautifulSoup
from bs4 import element


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
                if(tag.contents[0]!=''):
                    self.contents.append(tag.contents[0])

