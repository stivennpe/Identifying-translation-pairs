import pandas as pd 
from nltk.util import ngrams
'''
Dataset Creation
Data should be parallel e.g. en: DNA	es: ADN
'''
with open ("/content/en_datasets.txt", "r", encoding="UTF-8") as f:
    doc= f.readlines()
    
final_list_en= []
for line in doc:
    final_list_en.append(line.replace("\n",""))

with open ("/content/es_datasets.txt", "r", encoding="UTF-8") as f:
    doc= f.readlines()
    
final_list_es= []
for line in doc:
    final_list_es.append(line.replace("\n",""))



df_para= pd.DataFrame({"en": final_list_en, "es": final_list_es})
#df_para.head(20)



## N character search now
#shuffle the list to create a new dataset with negative examples
import copy
import random


df_para_pos = df_para.copy()
df_para_pos["Label"] = 1
df_para_neg = df_para.copy()

df_para_neg["es"] = df_para_neg["es"].sample(frac=1).values
df_para_neg["Label"] = -1

training_dataset = pd.concat([df_para_pos,df_para_neg])
df_shuffled = training_dataset.sample(frac=1, random_state=42)
df_shuffled.head()

#After Shuffluing the positive examples to create negative examples. The resulting dataset will have, for instance:

#en	    es	      Label
#LOXL	poliA +	   -1
#HP1	HP1	        1

#Model Creation

from sklearn import linear_model
from sklearn.model_selection import train_test_split
X = df_shuffled[['en', 'es']]
y = df_shuffled['Label']
    
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = .20, random_state = 40)

#Vectorization
#Char_wb to obtain character n grams from 2 to 5 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

tf_en_vectorizer= TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5))
tf_es_vectorizer= TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5))

#concatenation of vector spaces

from scipy.sparse import hstack

#train
X_train_features_en= tf_en_vectorizer.fit_transform(X_train["en"])
X_train_features_es= tf_es_vectorizer.fit_transform(X_train["es"])
X_train_features = hstack([X_train_features_en,X_train_features_es])
X_train_features.shape

#test
X_test_features_en= tf_en_vectorizer.transform(X_test["en"])
X_test_features_es= tf_es_vectorizer.transform(X_test["es"])
X_test_features = hstack([X_test_features_en,X_test_features_es])


#create LR instance and call fit
from sklearn.linear_model import LinearRegression,RidgeClassifierCV,RidgeClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score 


#Other options, uncomment for three classifiers training.
#clfs = {
#    "RandomForest":RandomForestClassifier(),
#    "Logistic Regression": LogisticRegression(),
#    "RidgeClassifierCV": RidgeClassifierCV()
#}
#for clf_name in clfs:
#  clf = clfs[clf_name]
#  clf.fit(X_train_features, Y_train)
#  y_pred = clf.predict(X_test_features)
#  print(clf_name, accuracy_score(Y_test, y_pred))

#Logistic Regression
lr_model= LogisticRegression()
lr_model.fit(X_train_features, Y_train)
y_pred_lr = lr_model.predict(X_test_features)

print(accuracy_score(Y_test, y_pred_lr))

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(Y_test, y_pred_lr)

#Random Forest

model= RandomForestClassifier()
model.fit(X_train_features, Y_train)
y_pred = model.predict(X_test_features)

print(accuracy_score(Y_test, y_pred))

#coefficient of determination (𝑅²) 
model.score(X_train_features, Y_train)

#Visualization of metrics.

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
accuracy_score(Y_test, y_pred)
print(f"The accuracy of the model is {round(accuracy_score(Y_test,y_pred),3)*100} %")

confusion_matrix(Y_test, y_pred)

import itertools
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): 
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #Labels
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), 
             fontsize = 20,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

#Confusion Matrix
cm = confusion_matrix(Y_test, y_pred)
plot_confusion_matrix(cm, classes = ['-1 Wrong', '1 - Correct'],
                      title = 'Exit_status Confusion Matrix')


'''
Save model to disk and load it later
'''                      

# save the model to disk
import pickle
filename = 'DesicionTree.sav'
pickle.dump(model, open(filename, 'wb'))
 

#Save vicabulary

pickle.dump(tf_en_vectorizer.vocabulary_,open("vocab_en.pkl","wb"))
pickle.dump(tf_es_vectorizer.vocabulary_,open("vocab_es.pkl","wb"))



# Make predictions using the testing set

test_enes.shape

lr_model.predict(test_enes)

'''
Using a trained model to identify if a word is the translation of the other. Uncomment as needed
Example.

filename = '/content/drive/MyDrive/random_fore/DesicionTree.sav'
vocab_en= "/content/drive/MyDrive/random_fore/vocab_en.pkl"
vocab_es= "/content/drive/MyDrive/random_fore/vocab_es.pkl"
loaded_model = pickle.load(open(filename, 'rb'))
tf_en_vectorizer= TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5), vocabulary=pickle.load(open(vocab_en, "rb")))
tf_es_vectorizer= TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5), vocabulary=pickle.load(open(vocab_es, "rb")))


test_en= tf_en_vectorizer.transform(["Ciliary", "proton/ion", "channelrhodopsin"])
test_es= tf_es_vectorizer.transform(["Ciliario", "Átomos", "Rodopsina"])
test_enes = hstack([test_en,test_es])

prediction= loaded_model.predict(test_enes)

Result: 
array([ 1, -1,  1], dtype=int64)
'''