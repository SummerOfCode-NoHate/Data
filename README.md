# Code excerpts

...

```python

>>> import texthero as hero
>>> import pandas as pd
>>> 
>>> # Load an example dataset.
>>> df = pd.read_csv(
...    "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
... )[["text"]]
>>> 
>>> df # The dataset consists of 737 texts about sports.
                                                  text
0    Claxton hunting first major medal\n\nBritish h...
1    O'Sullivan could run in Worlds\n\nSonia O'Sull...
2    Greene sets sights on world title\n\nMaurice G...
3    IAAF launches fight against drugs\n\nThe IAAF ...
4    Dibaba breaks 5,000m world record\n\nEthiopia'...
..                                                 ...
732  Agassi into second round in Dubai\n\nFourth se...
733  Mauresmo fights back to win title\n\nWorld num...
734  Federer wins title in Rotterdam\n\nWorld numbe...
735  GB players warned over security\n\nBritain's D...
736  Sharapova overcomes tough Molik\n\nWimbledon c...

[737 rows x 1 columns]

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
>>> df['kmeans_labels'] = df['tfidf'].pipe(hero.kmeans, n_clusters=5)
>>> 
>>> # Plot
>>> hero.scatterplot(df, 'pca', color='kmeans_labels', title="BBC Sport news dataset")
```
