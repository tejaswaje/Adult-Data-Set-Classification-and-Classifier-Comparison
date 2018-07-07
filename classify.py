# Importing external library
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from threading import Thread
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#%matplotlib inline
from sklearn.neighbors import KNeighborsClassifier
from statistics import mean
from sklearn.metrics import roc_curve,auc
import time

# Creating initial time stamp for the entire program
start_mainprog = time.time()

# Creating initial time stamp for the preprocessing
start_pre = time.time()

# Importing the dataset
dataset = pd.read_csv('adultdata.csv', delimiter = ",",header= None)
X = dataset.iloc[:,0:13]
y = np.asarray(dataset.iloc[:,14])

# Filling the missing values by most occuring value(mode) of respective column.
X = X.fillna(X.mode().iloc[0])

# Label Encoding the cloumns from string values to int values
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
for i in [1,3,5,6,7,8,9]:
    X.iloc[:,i] = le.fit_transform(X.iloc[:,i])

# Standarsize large values   
sc = StandardScaler()
X.iloc[:,[2,10,11]] = sc.fit_transform(X.iloc[:,[2,10,11]]) 

# Dropping the redundant data column
X = X.drop(X.columns[3], axis =1)   

# Transforming data to normal distribution
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X = quantile_transformer.fit_transform(X)

# Claculating the execution time of the preprocessing
execution_pre = time.time()-start_pre
print("Execution time for preprocessing is {}s".format(execution_pre))

# Creating initial time stamp for the Fine Tuning classifiers
start_tune = time.time()

# Split the data in 80:20 ratio for trianing and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Creating graph for the classifier, which shows Accuracy vs value of hyperparameter
f = plt.figure(figsize=(15,5))
clf_1 = f.add_subplot(131)
clf_2 = f.add_subplot(132)
clf_3 = f.add_subplot(133)


# Function which tunes Random Forest by looping values of hyperparameter and returns hyperparameter which has max accuracy
def tune_rdm_forest():
    n_range = range(1, 50)
    clf_scores = []
    for n in n_range:
        clf = RandomForestClassifier(n_estimators = n)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        #print("For value of n_estimators = {} accuracy is {}".format(n,accuracy_score(y_test,pred)))
        clf_scores.append(accuracy_score(y_test,pred))
        
# Plotting Accuracy vs value of hyperparameter  
    clf_1.plot(n_range, clf_scores)
    clf_1.set_xlabel('Value of n_estimators for Random Forest')
    clf_1.set_ylabel('Accuracy')
    return (clf_scores.index(max(clf_scores)) + 1)

# Function which tunes SVM by looping values of hyperparameter and returns hyperparameter which has max accuracy
def tune_svm():
    c_range = range(1, 5)
    clf_scores = []
    for c in c_range:
        clf = SVC(kernel = 'rbf', C = c*10)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        #print("For value of C = {} accuracy is {}".format(c*10,accuracy_score(y_test,pred)))
        clf_scores.append(accuracy_score(y_test,pred))
        
# Plotting Accuracy vs value of hyperparameter     
    clf_2.plot(c_range, clf_scores)
    clf_2.set_xlabel('Value of C for SVM')
    clf_2.set_ylabel('Accuracy')
    return (clf_scores.index(max(clf_scores)) * 10)

# Function which tunes KNN by looping values of hyperparameter and returns hyperparameter which has max accuracy
def tune_knn():
    k_range = range(1, 30)
    clf_scores = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors = k)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        #print("For value of K = {} accuracy is {}".format(k,accuracy_score(y_test,pred)))
        clf_scores.append(accuracy_score(y_test,pred))
        
# Plotting Accuracy vs value of hyperparameter     
    clf_3.plot(k_range, clf_scores)
    clf_3.set_xlabel('Value of Neighbors for KNN')
    clf_3.set_ylabel('Accuracy')
    return (clf_scores.index(max(clf_scores)) + 1)

# Calling functions to obtain best value of Hyperparameter
best_value_SVM = tune_svm()
best_value_KNN = tune_knn()  
best_value_rdm_forest = tune_rdm_forest()

# Plot the graph to show accuracy vs value of hyperparameter
plt.show()

# Claculating the execution time of the Fine tuning
execution_tune = time.time()-start_tune
print("Execution time for fine tuning each classifier is {}s".format(execution_tune))

# Creating initial time stamp for classifier prediciton
start_pred = time.time()

# Initialise the K-fold cross-validation value to 10
cv = KFold(n_splits=10)

# Array of Lists to store Confusion Matrix and positive rates (fpr and tpr) of classifiers after each flold of K-fold Cross Validation
cnf_mat_clf_1, clf_auc_1 = [], []
cnf_mat_clf_2, clf_auc_2 = [], []
cnf_mat_clf_3, clf_auc_3 = [], []
cnf_mat_clf_4, clf_auc_4 = [], []

# Creating graph to display ROC curve of each classifier
f2 = plt.figure(figsize=(10,10))
clf_1_plot = f2.add_subplot(221)
clf_2_plot = f2.add_subplot(222)
clf_3_plot = f2.add_subplot(223)
clf_4_plot = f2.add_subplot(224)


# Function for SVM Classifier to fit data, predict data, obtain confusion matrix and positive rates (fpr and tpr)
def clf_1(X_train_index, y_train_index, X_test_index, y_test_index): 
    global cnf_mat_clf_1, clf_1_plot
    clf = SVC(kernel = 'rbf', C = best_value_SVM, probability=True)
    clf.fit(X_train_index,y_train_index)
    ypred = clf.predict(X_test_index)
    
    y_score = clf.predict_proba(X_test_index)
    fpr, tpr, t = roc_curve(y_test_index, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    pr = [fpr, tpr]
    clf_auc_1.append(pr)
    
    clf_1_plot.plot(fpr, tpr, lw=2, label='ROC fold for SVM (AUC = %0.4f)' % (roc_auc))
    cnf_mat_clf_1.append(confusion_matrix(y_test_index, ypred))
    #print(confusion_matrix(y_test_index, ypred))
    #print("Accuracy for SVM = {} ".format(accuracy_score(y_test_index, ypred)))

# Function for Random Forest Classifier to fit data, predict data, obtain confusion matrix and positive rates (fpr and tpr)
def clf_2(X_train_index,y_train_index, X_test_index, y_test_index):
    global cnf_mat_clf_2, clf_2_plot
    clf = RandomForestClassifier(n_estimators = best_value_rdm_forest)
    clf.fit(X_train_index,y_train_index)
    ypred = clf.predict(X_test_index)
    
    y_score = clf.predict_proba(X_test_index)
    fpr, tpr, t = roc_curve(y_test_index, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    pr = [fpr, tpr]
    clf_auc_2.append(pr)
    
    clf_2_plot.plot(fpr, tpr, lw=2, label='ROC fold for RF (AUC = %0.4f)' % (roc_auc))
    cnf_mat_clf_2.append(confusion_matrix(y_test_index, ypred))
    #print(confusion_matrix(y_test_index, ypred))
    #print("Accuracy for Random Forest = {} ".format(accuracy_score(y_test_index, ypred)))

# Function for KNN Classifier to fit data, predict data, obtain confusion matrix and positive rates (fpr and tpr)    
def clf_3(X_train_index,y_train_index, X_test_index, y_test_index):
    global cnf_mat_clf_3, clf_3_plot
    clf = KNeighborsClassifier(n_neighbors = best_value_KNN)
    clf.fit(X_train_index,y_train_index)
    ypred = clf.predict(X_test_index)
    
    y_score = clf.predict_proba(X_test_index)
    fpr, tpr, t = roc_curve(y_test_index, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    pr = [fpr, tpr]
    clf_auc_3.append(pr)
    
    clf_3_plot.plot(fpr, tpr, lw=2, label='ROC fold for KNN (AUC = %0.4f)' % (roc_auc))
    cnf_mat_clf_3.append(confusion_matrix(y_test_index, ypred))
    #print(confusion_matrix(y_test_index, ypred))
    #print("Accuracy for KNN = {} ".format(accuracy_score(y_test_index, ypred))) 
    
# Function for Naive Bayes Classifier to fit data, predict data, obtain confusion matrix and positive rates (fpr and tpr)
def clf_4(X_train_index,y_train_index, X_test_index, y_test_index):
    global cnf_mat_clf_4, clf_3_plot
    clf = GaussianNB()
    clf.fit(X_train_index,y_train_index)
    ypred = clf.predict(X_test_index)
    
    y_score = clf.predict_proba(X_test_index)
    fpr, tpr, t = roc_curve(y_test_index, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    pr = [fpr, tpr]
    clf_auc_4.append(pr)
    
    clf_4_plot.plot(fpr, tpr, lw=2, label = 'ROC fold for NB (AUC = %0.4f)' % (roc_auc))
    cnf_mat_clf_4.append( confusion_matrix(y_test_index, ypred))
    #print(confusion_matrix(y_test_index, ypred))
    #print("Accuracy for Naive Bayes = {} ".format(accuracy_score(y_test_index, ypred)))

# List to store all the threads    
threads = []

# Splitting the data into 10 folds
for train_index, test_index in cv.split(X):

# Running all the classifiers
    for clf in [clf_1, clf_2, clf_3, clf_4]:

# Creating Thread for each iteration of 10 fold cross validation for each classifier        
        p = Thread(target = clf , args=(X[train_index], y[train_index],X[test_index],y[test_index]))
        p.start()
        threads.append(p)

# Waiting for all threads to execute        
for i in threads:
    i.join()

# Claculating the execution time for the classifiers
execution_pred = time.time()-start_pred
print("Execution time for making predictions by each classifier is {}s".format(execution_pred))

# Creating initial time stamp for the Evaluation Metrices
start_eval = time.time()

# Initialization Variables for Creating Evaluation Metrices
classifiers = ["SVM", "Random Forest", "KNN", "GaussianNB"]
count = -1 
cnf_matrix_list = []
fpr_list = []
tpr_list = []

# Obtaining data from all the folds for every classifier and creating confusion matrix
for clf in [cnf_mat_clf_1, cnf_mat_clf_2, cnf_mat_clf_3, cnf_mat_clf_4]:
    count += 1
    accuracy_list = []
    precision_list = []
    recall_list = []
    cnf_matrix = np.zeros(shape= [2,2])
    for i in range(0,10):
        cnf_matrix += clf[i]
        tp = clf[i][0][0]
        fp = clf[i][0][1]
        fn = clf[i][1][0]
        tn = clf[i][1][1]
        fpr_list.append(clf[i][0])
        tpr_list.append(clf[i][1])
        
        accuracy_list.append((tp+tn)/(tp+tn+fp+fn))
        precision_list.append(tp/(tp+fp))
        recall_list.append(tp/(tp +fn))
        
# Obtainig mean accuracy, precision, recall for each classifier        
    accuracy = mean(accuracy_list)
    precision = mean(precision_list)
    recall = mean(recall_list)
    cnf_matrix_list.append(cnf_matrix)
    print("Accuracy for {} is {}".format(classifiers[count], accuracy))
    print("Precision for {} is {}".format(classifiers[count], precision))
    print("Recall for {} is {}".format(classifiers[count], recall))
    print("Confusion Matrix for {} is {}".format(classifiers[count], cnf_matrix))
    
# Initialixation and loop for prointing ROC Curve
count = -1
for i in [clf_1_plot,clf_2_plot,clf_3_plot,clf_4_plot]:
    count += 1
    i.legend(loc = "best")
    i.set_xlabel('False Positivr Rate')
    i.set_ylabel('True Positivr Rate')
    i.set_title(" ROC for {}".format(classifiers[count]))
plt.legend(loc = "best")
plt.show()     

# Claculating the execution time for the classification
execution_eval = time.time()-start_eval
print("Execution time for evaluating each classifier is {}s".format(execution_eval))  

# Claculating the execution time for the entire program
execution_mainprog = time.time()-start_mainprog
print("Execution time for the entire code is {}s".format(execution_mainprog))  