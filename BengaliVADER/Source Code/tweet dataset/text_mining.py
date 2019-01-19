import re,math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
f1 = open('absolute_bangla_tweets_positive.txt', 'r', encoding='utf-8-sig')
f2 = open('absolute_bangla_tweets_negative.txt', 'r', encoding='utf-8-sig')
# f1 = open('Restaurant_pos.txt', 'r', encoding='utf-8-sig')
# f2 = open('Restaurant_neg.txt', 'r', encoding='utf-8-sig')
# data = open('toTest.txt', 'r', encoding='utf-8-sig')
st = open('stopwords.txt', 'r', encoding="utf-8-sig")

PUNC_LIST = ["।", "!", "?", ",", ";", "ঃ", "\"", "-", "(", ")", "[", "]","$","%"]

# file1 = []
# file2 = []
count = 0
file1 = [i.strip() for i in f1]
file2 = [i.strip() for i in f2]
s = [i.strip() for i in st]
# data = [i.strip() for i in data]
# for i in f1:
#     a, b = i.split("\t")
#     file1.append(a.strip())
#     # polarity.append(b.strip())
# for i in f2:
#     a, b = i.split("\t")
#     file2.append(a.strip())

# removing punctuation from data
def remove_punc(line):
    token_line = []
    l = filter(None, re.split("[, \-!?:\d+]+", line))
    for token in l:
        if token not in PUNC_LIST:
            token_line.append(token)
    return token_line

# removing stopwords from data
def remove_stopwords(token):
    token_af_rem_st = []
    for t in token:
        if t not in s:
            token_af_rem_st.append(t)
    return token_af_rem_st


Precision = []
Recall = []
f1_score = []
def cross_fold(x):
    i = 0
    train_data = []
    test_data = []
    test_labels = []
    train_labels = []

    for line in file1:
        if i >= x*1106 and i<(x+1)*1106: # i >= x*158 and i<(x+1)*158
            test_data.append(line)
            test_labels.append("pos")
        else:
            train_data.append(line)
            train_labels.append("pos")
        i += 1

    i = 0
    for line in file2:
            
        if i >= x*1106 and i<(x+1)*1106:
            test_data.append(line)
            test_labels.append("neg")
        else:
            train_data.append(line)
            train_labels.append("neg")
        i += 1
    print("Testing fold %d: %d" % (x + 1, len(test_labels)))

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)
    # print(test_vectors)
    # Perform Naive Bayes classifier for multinomial models
    # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

    classifier_rbf = MultinomialNB()
    classifier_rbf.fit(train_vectors, train_labels)
    prediction_rbf = classifier_rbf.predict(test_vectors)

    # classifier_rbf = svm.SVC(kernel='linear')
    # classifier_rbf.fit(train_vectors, train_labels)
    # prediction_rbf = classifier_rbf.predict(test_vectors)

    # classifier_rbf = tree.DecisionTreeClassifier()
    # classifier_rbf.fit(train_vectors, train_labels)
    # prediction_rbf = classifier_rbf.predict(test_vectors)

    # Print results in a nice table
    print("Results of Naive Bayes classifier")
    print(classification_report(test_labels, prediction_rbf))
    print("Accuracy : %.05f" % np.mean(test_labels == prediction_rbf))
    confusion = confusion_matrix(test_labels, prediction_rbf)
    print(confusion)
    TP = confusion[0,0]
    FN = confusion[0,1]
    FP = confusion[1,0]
    TN = confusion[1,1]
    PRECISION = TP/(TP+FP)
    RECALL = TP/(TP+FN)
    print("Precision:",PRECISION)
    print("Recall:",RECALL)
    print("F1:",(2*PRECISION*RECALL)/(PRECISION+RECALL))
    print(TP,TN,FN,FP)
    
    Precision.append(PRECISION)
    Recall.append(RECALL)
    f1_score.append((2*PRECISION*RECALL)/(PRECISION+RECALL))
    
    return np.mean(test_labels == prediction_rbf)


Acc = []
for f in range(0, 10):
    acc = cross_fold(f)
    Acc.append(acc)
print("Cross Validation Accuracies:  %.07f" % np.mean(Acc))
print("Average Precision:",sum(Precision)/len(Precision))
print("Average Recall:",sum(Recall)/len(Recall))
print("Average f1_score:",sum(f1_score)/len(f1_score))

# # Testing
# data = [i.strip() for i in data]
# for line in data:
#     token_line = remove_punc(line)
#     sum_of_tf_idf_pos = 0
#     sum_of_tf_idf_neg = 0
#     for term in token_line:
#         if term in tf_idf_pos:
#             sum_of_tf_idf_pos += tf_idf_pos[term]
#         else:
#             sum_of_tf_idf_pos += 0
#         if term in tf_idf_neg:
#             sum_of_tf_idf_neg += tf_idf_neg[term]
#         else:
#             sum_of_tf_idf_neg += 0
    
    
#     if sum_of_tf_idf_pos != 0 and sum_of_tf_idf_neg != 0:
#         positivity = (sum_of_tf_idf_pos/(sum_of_tf_idf_pos*sum_of_tf_idf_neg))*100
#         negativity = (sum_of_tf_idf_neg/(sum_of_tf_idf_pos*sum_of_tf_idf_neg))*100
#     else:
#         positivity = 0
#         negativity = 0

#     print(positivity,negativity)
#     if positivity > negativity:
#         print(line + " :positive")
#     elif positivity < negativity:
#         print(line + " :negative")
#     else:
        
#         print(line + " :neutral")
