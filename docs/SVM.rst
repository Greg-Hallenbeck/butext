Support Vector Machine (SVM)
----------------------------

A SVM is a supervised learning model that is used for classifciation tasks. It tries to find the best line or hyperplane, depending on the dimesion of the data, to seperate the data from different classes. It also tries to maximize the distance between the boundary and the nearest point of each class. It is a very popular and accepted model used for many different scenarios, with one of those being learning textual data.

TF-IDF works with the SVM model since it does not understand words, it only understands numbers. TF-IDF assigns numerical values to each word that the model can understand. It also reduces the noise of the data as the common words get asigned zero or close to zero, while the more unique words to a class hold more weight. In the case of spam vs ham messages, there is often disinct word patterns that TF-IDF picks up on. For those reasons, TF-IDF would work well for building a SVM model.

**Importing Necessary Packages**

.. code-block :: python
	
	import butext as bax
	import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.svm import SVC
	from sklearn.linear_model import SGDClassifier
	from sklearn.metrics import classification_report
	from scipy.sparse import csr_matrix
	import matplotlib.pyplot as plt
	import seaborn as sns

**Uploading Datset**

.. code-block :: python

   spam = pd.read_csv("https://tinyurl.com/4narz8b3", sep="\t")
   spam['doc_id'] = range(len(spam)) #Need to tokenize per email, so add index column to data
   spam.head()

**Output**

.. code-block :: none

        class 	text	                                           doc_id
   0	 ham  	Go until jurong point, crazy.. Available only ...	0
   1	 ham	Ok lar... Joking wif u oni...				1
   2	 spam 	Free entry in 2 a wkly comp to win FA Cup fina...	2
   3	 ham	U dun say so early hor... U c already then say...   	3
   4	 ham	Nah I don't think he goes to usf, he lives aro...	4



**#Calculate the tfidf of data**

.. code-block :: python

	spam_tfidf = (
    spam
    .pipe(bax.tokenize, 'text')
    	.pipe(bax.stopwords, 'word')
    	.groupby('doc_id')['word']
    	.value_counts(normalize=True)
    	.reset_index()
    	.pipe(bax.tf_idf, col='doc_id')
	)
	x = spam_tfidf.sort_values(by = 'tf_idf', ascending= False)
	x = x.loc[x.tf_idf != 0]

**Output**

.. code-block :: python

	X = spam_tfidf.pivot(index="doc_id", columns="word", values="tf_idf").fillna(0) #Convert into matrix format for sklearn

.. code-block :: python

	y = spam.set_index("doc_id")["class"] # set y to class, as its what we want to predict

.. code-block :: python

	common_ids = X.index.intersection(y.index)
	X = X.loc[common_ids]
	y = y.loc[common_ids]


.. code-block :: python

	X = csr_matrix(X.values) # converts matrix to sparse as original X was dense, necessary for quick runtime

.. code-block :: python

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

.. code-block :: python

	svm_model = SVC(kernel='linear') #this solves the exact SVM, it is more exact
	svm_model.fit(X_train, y_train)

.. code-block :: python

	y_pred = svm_model.predict(X_test)
	print(classification_report(y_test, y_pred))


**Output**

.. code-block :: none
	
			 precision   recall    f1-score  support

	         ham       0.99      0.99      0.99       956
    	    	spam       0.95      0.92      0.94       156

	    accuracy                           0.98      1112
	   macro avg       0.97      0.96      0.96      1112
	weighted avg       0.98      0.98      0.98      1112


.. code-block :: python

	svm_model2 = SGDClassifier() #This estimates the svm using stochastic gradient decsent, its less exact but fatser and takes less memory
	svm_model2.fit(X_train, y_train)

.. code-block :: python

	y_pred2 = svm_model2.predict(X_test)
	print(classification_report(y_test, y_pred))

**Output**

.. code-block :: none
	
			 precision   recall    f1-score  support

	         ham       0.99      0.99      0.99       956
    	    	spam       0.95      0.92      0.94       156

	    accuracy                           0.98      1112
	   macro avg       0.97      0.96      0.96      1112
	weighted avg       0.98      0.98      0.98      1112




