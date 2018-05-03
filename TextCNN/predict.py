# -*- coding: utf-8 -*-
#prediction using model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.predict
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import codecs
import os

import numpy as np
import tensorflow as tf
# from p5_fastTextB_model import fastTextB as fastText
# from a02_TextCNN.other_experiement.data_util_zhihu import load_data_predict,load_final_test_data,create_voabulary,create_voabulary_label
from tflearn.data_utils import pad_sequences  # to_categorical

from data_preprocess import get_vocabulary
from TextCNN.data_util import create_vocabulary1, load_data_multilabel2
from TextCNN.p7_TextCNN_model import TextCNN

#configuration
FLAGS=tf.app.flags.FLAGS
# tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
# tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.") #批处理的大小 32-->128
# tf.app.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate.") #批处理的大小 32-->128
# tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少
# tf.app.flags.DEFINE_string("ckpt_dir","text_cnn_title_desc_checkpoint/","checkpoint location for the model")
# tf.app.flags.DEFINE_integer("sentence_len",100,"max sentence length")
# tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
# tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
# tf.app.flags.DEFINE_integer("num_epochs",15,"number of epochs.")
# tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
#
# tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
# tf.app.flags.DEFINE_string("predict_target_file","text_cnn_title_desc_checkpoint/zhihu_result_cnn_multilabel_v6_e14.csv","target file path for final prediction")
# tf.app.flags.DEFINE_string("predict_source_file",'test-zhihu-forpredict-title-desc-v6.txt',"target file path for final prediction") #test-zhihu-forpredict-v4only-title.txt
# tf.app.flags.DEFINE_string("word2vec_model_path","zhihu-word2vec-title-desc.bin-100","word2vec's vocabulary and vectors") #zhihu-word2vec.bin-100
# tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") #128
# tf.app.flags.DEFINE_string("traning_data_path","../data/comments2.txt","path of traning data.") #sample_multiple_label.txt-->train_label_single100_merge
# tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_string("traning_data_path","../data/comments2.txt","path of traning data.") #sample_multiple_label.txt-->train_label_single100_merge
tf.app.flags.DEFINE_integer("vocab_size",100000,"maximum vocab size.")

tf.app.flags.DEFINE_float("learning_rate",0.0003,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","text_cnn_title_desc_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",100,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",50,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") #256--->512
tf.app.flags.DEFINE_string("word2vec_model_path","word2vec-title-desc.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
##############################################################################################################################################
# filter_sizes=[1,2,3,4,5,6,7]#[1,2,3,4,5,6,7]

filter_sizes=[6,7,8]

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
# 1.load data with vocabulary of words and labels
# vocabulary_word2index, vocabulary_index2word = create_voabulary(simple='simple',
#                                                                 word2vec_model_path=FLAGS.word2vec_model_path,
#                                                                 name_scope="cnn2")
# vocab_size = len(vocabulary_word2index)
# vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope="cnn2")
# questionid_question_lists = load_final_test_data(FLAGS.predict_source_file)
# test = load_data_predict(vocabulary_word2index, vocabulary_word2index_label, questionid_question_lists)




vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label = create_vocabulary1(FLAGS.traning_data_path,name_scope=FLAGS.name_scope)

vocab_size = len(vocabulary_word2index);print("cnn_model.vocab_size:",vocab_size);num_classes=len(vocabulary_index2label);print("num_classes:",num_classes)
train, test= load_data_multilabel2(FLAGS.traning_data_path,vocabulary_word2index, vocabulary_label2index,FLAGS.sentence_len)
testX = []

# 3.create session.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph=tf.Graph().as_default()
num_classes = 2
global sess
global textCNN
with graph:
    sess=tf.Session(config=config)
# 4.Instantiate Model
    textCNN = TextCNN(filter_sizes, FLAGS.num_filters, num_classes, FLAGS.learning_rate, FLAGS.batch_size,
                  FLAGS.decay_steps, FLAGS.decay_rate,
                  FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training,multi_label_flag=FLAGS.multi_label_flag)
    # ,multi_label_flag=FLAGS.multi_label_flag)
    saver = tf.train.Saver()
    if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
        print("Restoring Variables from Checkpoint")
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
    else:
        print("Can't find the checkpoint.going to stop")
    #return
# 5.feed data, to get logits
number_of_training_data = len(test);
print("number_of_training_data:", number_of_training_data)
#index = 0
#predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
#############################################################################################################################################
def get_logits_with_value_by_input(start,end):
    x=test[0][start:end]
    global sess
    global textCNN
    logits = sess.run(textCNN.logits, feed_dict={textCNN.input_x: x, textCNN.dropout_keep_prob: 1,textCNN.tst: True})
    predicted_labels,value_labels = get_label_using_logits_with_value(logits[0], vocabulary_index2label)
    value_labels_exp= np.exp(value_labels)
    p_labels=value_labels_exp/np.sum(value_labels_exp)
    return predicted_labels,p_labels

def predict_test(start,end):
    x=test[0][start:end]
    y=test[1][start:end]
    global sess
    global textCNN
    sentence = get_input_words(x)
    # print(sentence)
    logits = sess.run(textCNN.logits, feed_dict={textCNN.input_x: x, textCNN.dropout_keep_prob: 1,textCNN.tst: True})
    predicted_labels,value_labels = get_label_using_logits_with_value(logits[0], vocabulary_index2label)
    value_labels_exp= np.exp(value_labels)
    p_labels=value_labels_exp/np.sum(value_labels_exp)
    print(sentence)
    print('predict:'+predicted_labels[0])
    # print('prob'+p_labels)
    print(y[0])
    return predicted_labels,p_labels


def get_input_words(x):
    sentence = str()
    for value in x[0]:
        try:
            sentence+=(vocabulary_index2word[value])
        except:
            print(value)
    return sentence

def main(_):
    # 1.load data with vocabulary of words and labels
    # vocabulary_word2index, vocabulary_index2word = create_voabulary(simple='simple',word2vec_model_path=FLAGS.word2vec_model_path,name_scope="cnn2")
    # vocab_size = len(vocabulary_word2index)
    # vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope="cnn2")
    # questionid_question_lists=load_final_test_data(FLAGS.predict_source_file)
    # test= load_data_predict(vocabulary_word2index,vocabulary_word2index_label,questionid_question_lists)


    vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label = get_vocabulary()

    vocab_size = len(vocabulary_word2index);
    print("cnn_model.vocab_size:", vocab_size);
    num_classes = len(vocabulary_index2label);
    print("num_classes:", num_classes)
    train, test = load_data_multilabel2(FLAGS.traning_data_path, vocabulary_word2index, vocabulary_label2index,
                                        FLAGS.sentence_len)

    testX=[]
    question_id_list=[]
    for tuple in test:
        question_id,question_string_list=tuple
        question_id_list.append(question_id)
        testX.append(question_string_list)
    # 2.Data preprocessing: Sequence padding
    print("start padding....")
    testX2 = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    print("end padding...")
   # 3.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # 4.Instantiate Model
        textCNN=TextCNN(filter_sizes,FLAGS.num_filters,FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,FLAGS.decay_rate,
                        FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        # 5.feed data, to get logits
        number_of_training_data=len(testX2);print("number_of_training_data:",number_of_training_data)
        index=0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
            logits=sess.run(textCNN.logits,feed_dict={textCNN.input_x:testX2[start:end],textCNN.dropout_keep_prob:1}) #'shape of logits:', ( 1, 1999)
            # 6. get lable using logtis
            # predicted_labels=get_label_using_logits(logits[0],vocabulary_index2word_label)
            predicted_labels=get_label_using_logits(logits[0],vocabulary_index2label)
            # 7. write question id and labels to file system.
            write_question_id_with_labels(question_id_list[index],predicted_labels,predict_target_file_f)
            index=index+1
        predict_target_file_f.close()

# get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=5):
    index_list=np.argsort(logits)[-top_number:] #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list=index_list[::-1]
    label_list=[]
    for index in index_list:
        label=vocabulary_index2word_label[index]
        label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return label_list

# get label using logits
def get_label_using_logits_with_value(logits,vocabulary_index2word_label,top_number=5):
    index_list=np.argsort(logits)[-top_number:] #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list=index_list[::-1]
    value_list=[]
    label_list=[]
    for index in index_list:
        label=vocabulary_index2word_label[index]
        label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
        value_list.append(logits[index])
    return label_list,value_list

# write question id and labels to file system.
def write_question_id_with_labels(question_id,labels_list,f):
    labels_string=",".join(labels_list)
    f.write(question_id+","+labels_string+"\n")

if __name__ == "__main__":
    #tf.app.run()
    for i in range(len(test[0])):
        labels,list_value=predict_test(i, i+1)
    # print("labels:",labels)
    # print("list_value:", list_value)