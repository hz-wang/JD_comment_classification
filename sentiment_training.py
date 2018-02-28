# -*- coding:utf-8 -*-
import numpy as np
import codecs
import time
import tools
from keras.models import load_model

# load training sentiment data
# Input: @path of the sentiment data
# Output:@sample list
#        @label list
#        @ID2label transition table
def load_training_data_sentiment(path):
    samples = []
    labels = []
    file = codecs.open(path, "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        samples.append(terms[1])
        labels.append(int(terms[0]))
    file_not = codecs.open('./data/produce_not.txt', "r", encoding='utf-8', errors='ignore')
    for line in file_not.readlines():
        line = line.strip()
        terms = line.split('\t')
        samples.append(terms[1])
        labels.append(int(terms[0]))
    file_0sentiment = codecs.open('./data/produce_0sentiment.txt', "r", encoding='utf-8', errors='ignore')
    for line in file_0sentiment.readlines():
        line = line.strip()
        terms = line.split('\t')
        for i in range(3):
            samples.append(terms[1])
            labels.append(int(terms[0]))
    ID2label = {}
    ID2label[0] = '中性'
    ID2label[1] = '正面'
    ID2label[2] = '负面'
    return samples, labels, ID2label

# load validation sentiment data
# Input: @path of the sentiment data
# Output:@sample list
#        @label list
def load_val_data_sentiment(path):
    samples = []
    labels = []
    file = codecs.open(path, "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        samples.append(terms[1])
        labels.append(int(terms[0]))
    return samples, labels

# load prediction sentiment data
# Output:@sample list
#        @label list
#        @ID2label transition table
def load_predict_data_sentiment():
    samples = []
    labels = []
    file = codecs.open('./data/sentiment_test_data.txt', "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        samples.append(terms[1])
        labels.append(int(terms[0]))
    ID2label = {}
    ID2label[0] = 0
    ID2label[1] = 1
    ID2label[2] = 2
    return samples, labels, ID2label

# load sentiment data
# Output:@accuracy of the model
def train_sentiment():
    #load data
    samples, labels, ID2label = load_training_data_sentiment('./data/manually_labeled_data_sentiment.txt')
    samples_val, labels_val = load_val_data_sentiment('./data/sentiment_test_data.txt')

    dict = tools.build_dict(samples, tools.MAX_NB_WORDS)    #bulid dict
    tools.save_dict(dict)   #save the dict to local

    #calculate weight for different to improve balance
    sentiment_weight = {}
    for i in range(2):
        sentiment_weight[i]=len(labels) / labels.count(i)

    print(len(dict))
    embedding_matrix, nb_words, EMBEDDING_DIM = tools.load_embedding(dict)  #load embedding
    N_label = len(ID2label)
    X, y = tools.normalize_training_data(samples, labels, N_label, dict, 100)   #normalize the input data
    X_val, y_val = tools.normalize_training_data(samples_val, labels_val, N_label, dict, 100)

    print(len(X))
    print(len(y))

    NUM = len(X)
    indices = np.arange(NUM)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    samples = np.asarray(samples)
    samples = samples[indices]
    labels = np.asarray(labels)
    labels = labels[indices]
    training_ratio = 1    #setting the training data percentage
    N_train = int(NUM * training_ratio)
    X_train = X[:N_train]
    y_train = y[:N_train]
    #X_val = X[N_train:]
    #y_val = y[N_train:]
    #samples_val = samples[N_train:]
    #labels_val = labels[N_train:]
    sample_weights = np.ones(len(y_train))  #initialize the sample weight as all 1

    model = tools.define_model(tools.MAX_SEQUENCE_LENGTH, embedding_matrix, nb_words, EMBEDDING_DIM, N_label)
    model_save_path = 'code\model_sentiment' #save the best model
    model = tools.train_model(model, X_train, y_train, X_val, y_val, sample_weights, model_save_path, sentiment_weight)

    score, acc = model.evaluate(X_val, y_val, batch_size=2000)  #get the score and acc for the model

    print('Test score:', score)
    print('Test accuracy:', acc)

    pred = model.predict(X_val, batch_size=2000) #get the concrete predicted value for each text
    labels_pred = tools.probs2label(pred)      #change the predicted value to labels
    #save the wrong result
    writer_sentiment = codecs.open('./data/wrong_analysis/sentiment_wrong_result.txt', "w", encoding='utf-8', errors='ignore')
    for i in range(len(labels_val)):
        if labels_val[i]!=labels_pred[i]:
            writer_sentiment.write(samples_val[i] +'\t'+ ID2label[labels_val[i]] +'\t'+ ID2label[labels_pred[i]] + '\n')
    writer_sentiment.flush()
    writer_sentiment.close()
    return acc

# predict sentiment type
# Output File:@accuracy of the model
#             @wrong prediction of the model
def predict():
    samples_sentiment, labels_sentiment, ID2label_sentiment = load_predict_data_sentiment()
    dict_sentiment = tools.load_dict('./code/sentiment_model/sentiment_weight_no_reduce_produce.dict')
    # dict_sentiment_att = load_dict('D:/MSRA/JD_comments_analysis/code/sentiment_weight_no_reduce_produce.dict')
    print(len(dict_sentiment))
    model_sentiment1 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce1.h5')
    model_sentiment2 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce2.h5')
    model_sentiment3 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce3.h5')
    model_sentiment4 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce4.h5')
    model_sentiment5 = load_model('./code/sentiment_model/model_sentiment_weight_no_reduce_produce5.h5')

    worng_sentiment_list = ["ori\tp1\tp2\tp3\tp4\tp5\tp"]
    count = 0
    for i in range(len(samples_sentiment)):
        predict_label_sentiment1 = tools.predict(samples_sentiment[i], dict_sentiment, model_sentiment1, ID2label_sentiment)
        predict_label_sentiment2 = tools.predict(samples_sentiment[i], dict_sentiment, model_sentiment2, ID2label_sentiment)
        predict_label_sentiment3 = tools.predict(samples_sentiment[i], dict_sentiment, model_sentiment3, ID2label_sentiment)
        predict_label_sentiment4 = tools.predict(samples_sentiment[i], dict_sentiment, model_sentiment4, ID2label_sentiment)
        predict_label_sentiment5 = tools.predict(samples_sentiment[i], dict_sentiment, model_sentiment5, ID2label_sentiment)
        predict_list = [predict_label_sentiment1, predict_label_sentiment2, predict_label_sentiment3, predict_label_sentiment4, predict_label_sentiment5]
        predict_label_sentiment = max(predict_list, key=predict_list.count)

        if int(predict_label_sentiment) != int(labels_sentiment[i]):
            # worng_sentiment_list.append(str(labels_sentiment[i])+"\t"+str(predict_label_sentiment4)+"\t"+samples_sentiment[i])
            worng_sentiment_list.append(str(labels_sentiment[i])+"\t"+str(predict_label_sentiment1)+"\t"+str(predict_label_sentiment2)+"\t"+str(predict_label_sentiment3)+"\t"+str(predict_label_sentiment4)+"\t"+str(predict_label_sentiment5)+"\t"+str(predict_label_sentiment)+"\t"+samples_sentiment[i])
            count = count + 1
    writer = codecs.open('./data/wrong_analysis/sentiment_test_data_predict_wrong_5.txt', "w", encoding='utf-8',
                         errors='ignore')
    print(1-count/len(samples_sentiment))
    writer.write("Acc:"+str(1-count/len(samples_sentiment))+'\n')
    for text in worng_sentiment_list:
        writer.write(text+'\n')
    writer.flush()
    writer.close()


if __name__ == '__main__':
    # start = time.clock()
    # train_sentiment()
    # print(time.clock() - start) #show the running time of the training process

    predict()