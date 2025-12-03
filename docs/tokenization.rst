**Tokenization**
----------------

Tokenization is the process in which a particular portion of text is split into individual words or tokens. For example, the sentence "I like dogs" would become ["I" , "like", "dogs"]. While at its surface it seems like a pretty simple task, it assists with basic textual anaysis such as word counting and text frequencies all the way to Machine Learning, Natural Language Processing, and Deep Learning. In the 2017 paper "Attention Is All You Need" by Ashish Vaswani et al. which is the creation of the transfomer architecture which gave birth to the GPT decoder which is resposible for Chat-GPT, the architecture starts with tokenizing text and assign each word a vector embedding. Thus, while this task seems rudimentary, it has birthed many of the AI/Machine Learning developments in todays society.

Now we'll start with the basics of tokenization and see how it works:



**Importing Necessary Packages**

.. code-block :: python

	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
	import butext as bax



**Uploading Datset**

.. code-block :: python
	
	spam = pd.read_csv("https://tinyurl.com/4narz8b3",sep="\t")
	spam.head(5)

**Output**

.. code-block :: none

		class	text
	0	ham	Go until jurong point, crazy.. Available only ...
	1	ham	Ok lar... Joking wif u oni...
	2	spam	Free entry in 2 a wkly comp to win FA Cup fina...
	3	ham	U dun say so early hor... U c already then say...
	4	ham	Nah I don't think he goes to usf, he lives aro...


**Tokenization In Action**

Now that our data is loaded, we can use our tokenization function to find some basic information about our data. Our tokenization function is denoted by bax.tokenize and works best within pandas pipe() operator. We will also utilize our stop`words function denoted by bax.stopwords as it removes words not important to our analysis

.. code-block :: python

	spam_tokens = (
   	 	spam
    		.pipe(bax.tokenize, 'text')
    		.pipe(bax.stopwords, 'word')
		)
		spam_tokens

**Output**

.. code-block :: none
	
		class	text							word
	0	ham	Go until jurong point, crazy.. Available only ...	jurong
	0	ham	Go until jurong point, crazy.. Available only ...	point
	0	ham	Go until jurong point, crazy.. Available only ...	crazy
	0	ham	Go until jurong point, crazy.. Available only ...	available
	0	ham	Go until jurong point, crazy.. Available only ...	bugis
	...	...	... 							...
	5570	ham	The guy did some bitching but I acted like i'd...	week
	5570	ham	The guy did some bitching but I acted like i'd...	gave
	5570	ham	The guy did some bitching but I acted like i'd...	free
	5571	ham	Rofl. Its true to its name				rofl
	5571	ham	Rofl. Its true to its name				true

Now that we have out tokens, we can find plenty of useful information. Lets start with the most common words.

.. code-block :: python
	
	spam_tokens['word'].value_counts()

**Output**

.. code-block :: none

		
	word    count	
	u	1140
	2	482
	i'm	393
	ur	390
	just	371
	...	...
	3xx	1
	wc1n	1
	2667	1
	gsex	1
	chief	1


These are the most common words used in the emails within the whole dataset, however this not providing much help, so lets visualize the most common words per class (spam or ham).

.. code-block :: python	
	
	viz_data = spam_tokens.groupby('class')['word'].value_counts()
	viz_data = viz_data.reset_index(name = 'count')
	viz = viz_data.loc[viz_data['class'] == 'spam']
	viz1 = viz[0:10]
	sns.barplot(viz1, x = "word", y = 'count')
	plt.title("Most Common Words in Spam Emails")
	plt.xlabel("Word")
	plt.ylabel("Word Count")
	plt.show()

**Output**

.. image:: _build/html/_static/Tokenizationgraph1.png
	:alt: description
	:width: 400px


.. code-block :: python	

	viz = viz_data.loc[viz_data['class'] == 'ham']
	viz1 = viz[0:10]
	sns.barplot(viz1, x = "word", y = 'count')
	plt.title("Most Common Words in Ham Emails")
	plt.xlabel("Word")
	plt.ylabel("Word Count")
	plt.show()


**Output**

.. image:: _build/html/_static/Tokenizationgraph1.png
	:alt: description
	:width: 400px



Now we can analyze the top words in each class. The words in spam make sense with "free" or "claim", however, the top words in Ham make a little less sense. It almost seems like a lot of imformal text, which makes sense if emailing someone you know, but still pretty hard to interpet. This leads us to out topic of Relative Frequencies.



**Sources**

*https://arxiv.org/pdf/1706.03762*


*https://www.geeksforgeeks.org/nlp/nlp-how-tokenizing-text-sentence-words-works/*

