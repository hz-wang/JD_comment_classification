# -*- coding:utf-8 -*-
import numpy as np
import codecs
import time
import tools
from keras.models import load_model

# load sentiment data
# Input: @path of the sentiment data
# Output:@merge the class2 to class1
def load_class2_to_class1(class_transition_path):
    file = codecs.open(class_transition_path, "r", encoding='utf-8', errors='ignore')
    class2_class1 = {}
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        class2_class1[terms[0]] = terms[1]
    return class2_class1

# load training class2 data
# Input: @path of the class2 label
#        @path of the class2 data
# Output:@sample list
#        @label list
#        @ID2label transition table
def load_training_data_class2(label_path, data_path):
    classname2ID = {}
    ID2label = {}
    file = codecs.open(label_path, "r", encoding='utf-8', errors='ignore')
    count = 0
    for line in file.readlines():
        classname2ID[line.strip()] = count
        ID2label[count] = line.strip()
        count += 1

    samples = []
    labels = []
    file = codecs.open(data_path, "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        samples.append(terms[1])
        labels.append(classname2ID[terms[0]])
    return samples, labels, ID2label

# load validationclass2 data
# Input: @path of the class2 label
#        @path of the class2 data
# Output:@sample list
#        @label list
def load_val_data_class2(label_path, data_path):
    classname2ID = {}
    ID2label = {}
    file = codecs.open(label_path, "r", encoding='utf-8', errors='ignore')
    count = 0
    for line in file.readlines():
        classname2ID[line.strip()] = count
        ID2label[count] = line.strip()
        count += 1

    samples = []
    labels = []
    file = codecs.open(data_path, "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        samples.append(terms[1])
        labels.append(classname2ID[terms[0]])

    return samples, labels

# load prediction class2 data
# Output:@sample list
#        @label list
#        @ID2label transition table
def load_prediction_data_class2():
    classname2ID = {}
    ID2label = {}
    file = codecs.open('./data/class2_labels.txt', "r", encoding='utf-8', errors='ignore')
    count = 0
    for line in file.readlines():
        classname2ID[line.strip()] = count
        ID2label[count] = line.strip()
        count += 1

    samples = []
    labels = []
    file = codecs.open('./data/class_test_data.txt', "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        samples.append(terms[1])
        labels.append(classname2ID[terms[0]])
    return samples, labels, ID2label

# load sentiment data
# Output:@accuracy of the model class2
#        @accuracy of the model class1
def train_class():
    #load class data
    samples, labels, ID2label = load_training_data_class2('./data/class2_labels.txt', './data/manually_labeled_data_class2.txt')
    samples_val, labels_val = load_val_data_class2('./data/class2_labels.txt', './data/class_test_data.txt')
    dict = tools.build_dict(samples, tools.MAX_NB_WORDS)    #bulid dict

    #Calculate weight for class to improve balance with a bias 50
    class2_weight = {}
    for i in range(36):
        class2_weight[i]=len(labels) / (labels.count(i) + 50)

    print(len(dict))
    tools.save_dict(dict)    #save the dict to local
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
    model_save_path = 'code\model_class2'   #save the best model
    model = tools.train_model(model, X_train, y_train, X_val, y_val, sample_weights, model_save_path, class2_weight)

    score, accuracy_class2 = model.evaluate(X_val, y_val, batch_size=2000)   #get the score and acc for the model
    print('Test score:', score)
    print('Test accuracy:', accuracy_class2)

    pred = model.predict(X_val, batch_size=2000)    #get the concrete predicted value for each text
    labels_pred = tools.probs2label(pred)   #change the predicted value to labels

    #save the wrong result for class2
    writer_class2 = codecs.open('./data/wrong_analysis/class2_wrong_result.txt', "w", encoding='utf-8', errors='ignore')
    for i in range(len(labels_val)):
        if labels_val[i]!=labels_pred[i]:
            writer_class2.write(samples_val[i] +'\t'+ ID2label[labels_val[i]] +'\t'+ ID2label[labels_pred[i]] + '\n')
    writer_class2.flush()
    writer_class2.close()

    class2_class1 = load_class2_to_class1('./data/class2_class1.txt') #merge the class2 to class1
    N_class1_true = 0
    worng_class = []
    for i in range(len(labels_val)):
        if class2_class1[ID2label[labels_val[i]]]==class2_class1[ID2label[labels_pred[i]]]:
            N_class1_true += 1
        else:
            worng_class.append(class2_class1[ID2label[labels_val[i]]]+"\t"+class2_class1[ID2label[labels_pred[i]]]+"\t"+samples_val[i])

    #save the wrong result for class1
    writer = codecs.open('./data/wrong_analysis/class1_wrong_result.txt', "w", encoding='utf-8', errors='ignore')
    writer.write("original_label"+"\t"+"predict_label"+"\t"+"sample"+"\n")
    for item in worng_class:
        writer.write(item + '\n')
    writer.flush()
    writer.close()

    accuracy_class1 = N_class1_true/len(labels_val)
    print(accuracy_class1)
    return accuracy_class2, accuracy_class1

# predict sentiment type
# Output File:@class2 accuracy of the model
#             @class1 accurary of the model
#             @wrong prediction of the model
def predict():
    samples_class2, labels_class2, ID2label_class = load_prediction_data_class2()

    dict_class = tools.load_dict('./code/class_model/class2_weight_new.dict')
    # dict_sentiment_att = load_dict('./code/sentiment_weight_no_reduce_produce.dict')
    print(len(dict_class))
    model_class1 = load_model('./code/class_model/model_class2_weight_new1.h5')
    model_class2 = load_model('./code/class_model/model_class2_weight_new2.h5')
    model_class3 = load_model('./code/class_model/model_class2_weight_new3.h5')
    model_class4 = load_model('./code/class_model/model_class2_weight_new4.h5')
    model_class5 = load_model('./code/class_model/model_class2_weight_new5.h5')
    # model_class6 = load_model('./code/class_model/model_class2_weight_new6.h5')
    # model_class7 = load_model('./code/class_model/model_class2_weight_new7.h5')
    # model_class8 = load_model('./code/class_model/model_class2_weight_new8.h5')
    # model_class9 = load_model('./code/class_model/model_class2_weight_new9.h5')
    # model_class10 = load_model('./code/class_model/model_class2_weight_new10.h5')
    worng_class_list = ["ori\tp1\tp2\tp3\tp4\tp5\tp"]
    count1 = 0
    count2 = 0
    class_label = []
    for i in range(len(samples_class2)):
        predict_label_class2_1 = tools.predict(samples_class2[i], dict_class, model_class1, ID2label_class)
        predict_label_class2_2 = tools.predict(samples_class2[i], dict_class, model_class2, ID2label_class)
        predict_label_class2_3 = tools.predict(samples_class2[i], dict_class, model_class3, ID2label_class)
        predict_label_class2_4 = tools.predict(samples_class2[i], dict_class, model_class4, ID2label_class)
        predict_label_class2_5 = tools.predict(samples_class2[i], dict_class, model_class5, ID2label_class)
        # predict_label_class2_6 = tools.predict(samples_class2[i], dict_class, model_class1, ID2label_class)
        # predict_label_class2_7 = tools.predict(samples_class2[i], dict_class, model_class2, ID2label_class)
        # predict_label_class2_8 = tools.predict(samples_class2[i], dict_class, model_class3, ID2label_class)
        # predict_label_class2_9 = tools.predict(samples_class2[i], dict_class, model_class4, ID2label_class)
        # predict_label_class2_10 = tools.predict(samples_class2[i], dict_class, model_class5, ID2label_class)
        predict_list = [predict_label_class2_1, predict_label_class2_2, predict_label_class2_3, predict_label_class2_4, predict_label_class2_5]
        # predict_list = [predict_label_class2_1, predict_label_class2_2, predict_label_class2_3, predict_label_class2_4, predict_label_class2_5,
        #                 predict_label_class2_6, predict_label_class2_7, predict_label_class2_8, predict_label_class2_9, predict_label_class2_10]
        predict_label_class2 = max(predict_list, key=predict_list.count)
        class2_class1 = load_class2_to_class1('./data/class2_class1.txt')

        if class2_class1[predict_label_class2]!=class2_class1[ID2label_class[labels_class2[i]]]:
            count1 += 1
            # worng_class_list.append(ID2label_class[labels_class2[i]]+"\t"+str(predict_label_class2_1)+"\t"+samples_class2[i])
            worng_class_list.append(class2_class1[ID2label_class[labels_class2[i]]]+"\t"+str(predict_label_class2_1)+"\t"+str(predict_label_class2_2)+"\t"+str(predict_label_class2_3)+"\t"+str(predict_label_class2_4)+"\t"+str(predict_label_class2_5)+"\t"+str(class2_class1[predict_label_class2])+"\t"+samples_class2[i])
        if predict_label_class2!=ID2label_class[labels_class2[i]]:
            count2 += 1
    writer = codecs.open('./data/wrong_analysis/class_test_data_predict_wrong_5.txt', "w", encoding='utf-8',
                         errors='ignore')
    print("class1:"+str(1-count1/len(samples_class2)))
    print("class2:"+str(1-count2/len(samples_class2)))

    writer.write("Acc:"+str(1-count1/len(samples_class2))+'\n')
    for text in worng_class_list:
        writer.write(text+'\n')
    writer.flush()
    writer.close()

if __name__ == '__main__':
    # start = time.clock()
    # train_class()
    # print(time.clock() - start) #show the running time of the training process

    predict()