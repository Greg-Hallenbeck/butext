Principle Component Analysis (PCA) Example
------------------------------------------

In data science, data often contains a very high number of dimensions (i.e. columns or parameters), which makes understanding the underlying "shape" of the data difficult. To alleviate this, statisticians and data scientists may choose to project very high-dimensional data into lower-dimensional vector spaces (think: a coordinate plane).

The goal of Principal Component Analysis is to find the axes, given a set number of dimensions, which explain the most variance in the underlying data. In the context of text mining, this is typically done by finding the principal components for the "importance" of words in a corpus of texts. When doing this, every word is its own dimension, and its importance in one text represents the coordinate in that dimension for that piece of text.

By forming principal components, we no longer need to analyze differences between texts using differences in frequencies of every single word in the entire corpus. We can just use differences in principal components, which is not only computationally easier but is also visualizable if you use 2 or 3 dimensions.


**Importing Necessary Packages**

.. code-block :: python

	import butext as bax
	from sklearn.decomposition import PCA
	import pandas as pd
	from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
	import matplotlib.pyplot as plt




**PCA function**

.. code-block :: python

	def tokPCA(df, txtcolname, title, dim, preserves):
    	'''
    	df: The DataFrame being tokenized and projected into the PCA

    	txtcolname: The column of df that contains strings of texts to be tokenized

    	title: The column that contains the titles of this text

    	dim: The number of dimensions demanded for the PCA

    	preserves: A list of other column names that the user would like re-added to the PCA dataframe.

    	'''
   	reservedf = df[[title] + preserves]

    	df_tok = (
        	df
        	.pipe(bax.tokenize, txtcolname)
        	.pipe(bax.stopwords, 'word')
    	)

    	tfidf = (
    	df_tok
    	.groupby(title)['word'].value_counts(normalize = True)
    	.reset_index()
    	.pipe(bax.tf_idf, title)
    	)
    	X = tfidf.pivot(index=title, columns="word",values="tf_idf").fillna(0)
   	pca = PCA(n_components=dim)
    	X_r = pca.fit(X).transform(X)
    	pca_df = pd.DataFrame(X_r, columns=['PC1', 'PC2'], index=X.index)
    	pca_df.reset_index(inplace=True)
    	pca_df = pd.merge(reservedf, pca_df, on=title)
    	return(pca_df)



**Uploading Dataset**

.. code-block :: python

	ntflx = pd.read_csv("https://raw.githubusercontent.com/Greg-Hallenbeck/class-datasets/main/datasets/netflix.csv")
	ntflx

.. code-block :: python

	ntflx["genre"] = ""
	ntflx.loc[ntflx["genres"].str.contains("drama"), "genre"] = "drama"
	ntflx.loc[ntflx["genres"].str.contains("comedy"),"genre"] = "comedy"
	ntflx.loc[ntflx["genres"].str.contains("horror"),"genre"] = "horror"
	ntflx.loc[ntflx["genres"].str.contains("romance"),"genre"] = "romance"
	ntflx.loc[ntflx["genres"].str.contains("documentation"),"genre"] = "documentary"
	ntflx = ntflx.loc[ntflx["genre"] != ""]
	ntflx = ntflx[['id', 'description','type', 'genre', 'age_certification']]
	ntflx.head(5)

**Ouput**

.. code-block :: none

		id		description			type	genre	  age_certification
	0	ts300399	This collection includes ...	SHOW	docu..  	TV-MA
	1	tm84618		A mentally unstable Vietn...	MOVIE	drama		R
	2	tm127384	King Arthur, accompanied ...	MOVIE	comedy		PG
	3	tm70993		Brian Cohen is an average...	MOVIE	comedy		R
	4	tm190788	12-year-old Regan MacNeil...	MOVIE	horror		R



**Run PCA**

.. code-block :: python

	newdf = tokPCA(ntflx,"description","id",2,["genre","age_certification"])
	newdf.head(4)

**Output**

.. code-block :: none

		id         genre         age_certification       PC1        PC2
	0	ts300399   documentary   TV-MA                 0.029556   0.000573
	1	tm84618    drama         R                    -0.004263   0.004507
	2	tm127384   comedy        PG                   -0.003020   0.000247
	3	tm70993    comedy        R                    -0.014535  -0.001630
	4	tm190788   horror        R                    -0.039231   0.000530

**Visualize PCA**

.. code-block :: python

	plt.scatter(x=newdf.PC1,y=newdf.PC2,alpha=0.5)

**Output**

.. image:: _build/html/_static/PCA1.png
	:alt: description
	:width: 400px

**Problem:** Visualizing this PCA makes it clear that an outlier is skewing the data. Because PCA finds the dimensions meant to explain a lot of variance in the data, outliers can skew the dimensions and make them less useful for visualization.

**Identifying Outlier**

.. code-block :: python

	outlier = newdf[newdf.PC2>1]
	outlier 

**Output**

.. code-block :: none

		id         genre        age_certification        PC1        PC2
	2299    tm375302   documentary   NaN                   0.110894     4.468580

 
**Re-run PCA without outlier**

.. code-block :: python

	ntflx2 = ntflx[ntflx.id != "tm375302"]
	newdf2 = tokPCA(ntflx2,"description","id",2,["genre","age_certification"])
	groups = newdf2.groupby("genre")
	for name, group in groups:
	 plt.plot(group.PC1, group.PC2, marker='o', linestyle='',  markersize=4,alpha=0.7,label=name)
	plt.xlabel("PC1 (0.19%)")
	plt.ylabel("PC2 (0.16%)")
	plt.legend()
	plt.show()

**Output**

.. image:: _build/html/_static/PCA2.png
	:alt: description
	:width: 400px

Judging by the PCA, certain patterns begin to emerge which can tell us about what our principal components may represent. PC1 may have to do with the "seriousness" of words, while PC2 may have to do with whether words are more emotional or descriptive.

