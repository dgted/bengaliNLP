import os
from sklearn import svm
from sklearn import tree
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import MultinomialNB

Precision = []
Recall = []
f1_score = []
def cross_fold(fold=0):
    data_dir = "dataset_movie_reviews"
    classes = ['positive', 'negative']
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)  # data_dir\cur
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r',encoding="utf-8-sig") as f:
                content = f.read()
                if fname.startswith('cv' + str(fold)):
                    test_data.append(content)
                    test_labels.append(curr_class)
                else:
                    train_data.append(content)
                    train_labels.append(curr_class)

    print("Testing fold %d: %d" % (fold + 1, len(test_labels)))

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
