# Code excerpts

...

```python

>>> import texthero as hero
>>> import pandas as pd
>>> 
>>> # Load an example dataset.
>>> df = pd.read_csv(
...    "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
... )[["text"]] # The dataset consists of 737 texts about sports.
>>> 
>>> # Clean and tokenize all documents.
>>> df['text_preprocessed'] = df['text'].pipe(hero.clean).pipe(hero.tokenize)
>>> 
>>> # Calculate a vector representation of the documents using TF-IDF
>>> df_tfidf = df['text_preprocessed'].pipe(hero.tfidf)
>>> 
>>> # Find a lower-dimensional representation through PCA
>>> df["pca"] = df_tfidf.pipe(hero.normalize, norm="l2").pipe(hero.pca)
>>> 
>>> # Find clusters ("topics") in the documents with KMeans.
>>> df['kmeans_labels'] = df['pca'].pipe(hero.kmeans, n_clusters=5)
>>> 
>>> # Plot
>>> hero.scatterplot(df, 'pca', color='kmeans_labels', hover_data=["text_preprocessed"], title="BBC Sport news dataset")
```
