import numpy as np
import pandas as pd
import re
import nltk

yorumlar = pd.read_excel('moviereviews.xlsx')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

corpus= []
for i in range(39):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Reviews'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    corpus.append(yorum)
    

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5)
X = cv.fit_transform(corpus).toarray() 
y = yorumlar.iloc[:,1].values 
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)



















