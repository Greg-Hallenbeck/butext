Logistic Regression
-------------------

A Logitsitc Regression is a supervised learning model that is used for classification tasks. This model predicts the probability of classes by learning weights and biases, and using the sigmoid function to find probability. It then makes a prediciton based on this probabilty. This model increases the weights of features to make predicitons more accurate, thus it makes the model very interperatable.

TF-IDF can be used when analyzing text for this model since, like most models, it cannot undertand string inputs, only numbers, and TF-IDF is assing each word a number (weight) based of their uniqueness. This in process tells the model which words are more important to certain classes (like spam or ham emails). Since words with zero or low TF-IDF are meaningless, they won't scew with the results of this model.


**Importing Necessary Packages**

.. code-block :: python

	import butext as bax
	import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import classification_report

**Adding index column to data in order to tokenize per email**

.. code-block :: python

	spam = pd.read_csv("https://tinyurl.com/4narz8b3", sep="\t")
	spam['doc_id'] = range(len(spam)) 
	spam.head()


.. code-block :: none

		class	text						     doc_id
	0	ham	Go until jurong point, crazy.. Available only ...	0
	1	ham	Ok lar... Joking wif u oni...	1
	2	spam	Free entry in 2 a wkly comp to win FA Cup fina...	2
	3	ham	U dun say so early hor... U c already then say...	3
	4	ham	Nah I don't think he goes to usf, he lives aro...	4

.. code-block :: python
	
	log_tfidf = (
	     spam
	     .pipe(bax.tokenize, 'text')
         .pipe(bax.stopwords, 'word')
    	 .groupby('doc_id')['word']
    	 .value_counts(normalize=True)
    	 .reset_index()
    	 .pipe(bax.tf_idf, col='doc_id')
	)
	x = log_tfidf.sort_values(by = 'tf_idf', ascending= False)
	x = x.loc[x.tf_idf != 0]

**Converting into matrix format for sklearn**

.. code-block :: python

	X = log_tfidf.pivot(index="doc_id", columns="word", values="tf_idf").fillna(0) 

**Setting y to class, as its what we want to predict**

.. code-block :: python

	y = spam.set_index("doc_id")["class"] 
	
.. code-block :: python

	y = y.map({"ham": 0, "spam": 1})

.. code-block :: python

	common_ids = X.index.intersection(y.index)
	X = X.loc[common_ids]
	y = y.loc[common_ids]

.. code-block :: python

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y)

.. code-block :: python

	logreg = LogisticRegression(max_iter=1000)
	logreg.fit(X_train, y_train)


.. code-block :: python

	y_pred = logreg.predict(X_test)
	print(classification_report(y_test, y_pred))

**Output**

.. code-block :: none

	 		precision   recall    f1-score  support

	         0       0.97      1.00      0.98       963
    	    	 1       0.97      0.79      0.87       149

	    accuracy                           0.97      1112
	   macro avg       0.97      0.89      0.92      1112
	weighted avg       0.97      0.97      0.97      1112



**Top spam-indicative words**

.. code-block :: python

	coef_df = pd.DataFrame({
    "word": X.columns,
    "coef": logreg.coef_[0]
	}).sort_values(by="coef", ascending=False)
	coef_df.head(10) 

**Output**

.. code-block :: none

		word	coef
	8392	txt	3.607208
	2221	claim	3.028673
	6784	reply	2.754200
	5412	mobile	2.417796
	7690	stop	2.376611
	7154	service	2.221344
	6440	prize	2.151703
	8951	won	2.149550
	418	18	2.006830
	8523	urgent	1.952333

**Top ham- indicative words**

.. code-block :: python

	coef_df.tail(10) 

**Output**

.. code-block :: none

		word	coef
	9214	yup	-1.286121
	4258	i'll	-1.312963
	3804	got	-1.331940
	6884	road	-1.439586
	4130	home	-1.514126
	5068	lt	-1.538432
	3877	gt	-1.579734
	5855	ok	-1.611774
	8407	u	-1.616925
	4260	i'm	-1.797147


.. code-block :: python

	spam_tfidf = (
		spam
		.pipe(bax.tokenize, 'text')
		.groupby('class')['word']
		.value_counts(normalize=True)
		.reset_index()
		.pipe(bax.tf_idf, col='class')
	)
	x = spam_tfidf.sort_values(by = 'tf_idf', ascending= False)
	x = x.loc[x.tf_idf != 0]
	x

**Output**

.. code-block :: none

		lass	word	tf		idf		tf_idf
	7524	spam	claim	0.006390	0.693147	0.004429
	7528	spam	prize	0.005202	0.693147	0.003606
	35	ham	gt	0.004132	0.693147	0.002864
	7536	spam	won	0.004128	0.693147	0.002861
	37	ham	lt	0.004060	0.693147	0.002814
	...	...	...	...	...	...
	5798	ham	newsp... 0.000015	0.693147	0.000010
	5797	ham	newsc... 0.000015	0.693147	0.000010
	5828	ham	non	0.000015	0.693147	0.000010
	5827	ham	noisy	0.000015	0.693147	0.000010
	5810	ham	nike	0.000015	0.693147	0.000010

We can see that many words that are important to the regression appear here as well, showing consistency within results.



