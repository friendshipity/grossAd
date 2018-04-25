def get_vocabulary():
    PAD_ID = 0
    UNK_ID=1
    _PAD="_PAD"
    _UNK="UNK"

    f = open('e:\grossAd\src\data\comments2.txt', encoding='utf-8', errors='ignore')
    # f = open('src\data\comments2.txt', encoding='utf-8', errors='ignore')


    vacabulary = set()
    labels = set()
    for line in f.readlines():
        comment = line.split("__label__")[0]
        labels.add(line.split("__label__")[1].strip())
        for j in range(0, len(comment)):
            vacabulary.add(str(comment[j]))

    f.close()

    vocabulary_word2index = dict()
    vocabulary_index2word = dict()
    vocabulary_label2index = dict()
    vocabulary_index2label = dict()


    vocabulary_word2index[_PAD]=PAD_ID
    vocabulary_index2word[PAD_ID]=_PAD
    vocabulary_word2index[_UNK]=UNK_ID
    vocabulary_index2word[UNK_ID]=_UNK



    for index,word in enumerate(vacabulary):
        vocabulary_word2index[word] = index+2
        vocabulary_index2word[index+2] = word

    for index,word in enumerate(labels):
        vocabulary_label2index[word]= index
        vocabulary_index2label[index]= word




    return vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label

    # f.close
# vocabulary_word2index, vocabulary_index2word