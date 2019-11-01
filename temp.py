#Natural language processing
import numpy as np
import pandas as pd


#reading dataframe
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)
#quoting=3 to avoid double commas

dataset.head()


#cleaning the text
#first we remove the punchuations and exlamation marks
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])


    #converting into lower case
    review=review.lower()

    #converting into list
    review=review.split()

    #it converts the word into their root
    ps=PorterStemmer()


    #now to remove unneccessary work like "this that"
    #set is using to make process fast
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

# df=pd.DataFrame(corpus)
# df.head()


#creating a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
#max-features limits the words
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
#print(x.shape)
y=dataset.iloc[:,1].values
#print(y)


#splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


#Feature scaling
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# x_train=sc.fit_transform(x_train)
# x_test=sc.transform(x_test)


#creating our model
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)

#predictions
y_pred=model.predict(x_test)
# print(y_pred)
# print(y_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))