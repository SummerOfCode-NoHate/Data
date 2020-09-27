# Code excerpts

...

```python

>>> import texthero as hero
>>> import pandas as pd
>>> 
>>> # Load an example dataset.
>>> df = pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")[["text"]]
>>> 
>>> # Clean and tokenize all documents.
>>> df['text_preprocessed'] = df['text'].pipe(hero.clean).pipe(hero.tokenize)
>>> 
>>> # Calculate tfidf; only keep terms that appear in at most 60% and at least 10 documents.
>>> df_tfidf = df['text_preprocessed'].pipe(hero.tfidf, min_df=10, max_df=0.60)
>>> 
>>> # We can see that <5% of values are non-zero, so our memory usage is only 5% of the size of the matrix).
>>> df_tfidf.sparse.density
0.04782521092418972
>>> 
>>> # And the output also looks pretty.
>>> df_tfidf
        tfidf                                 ...                                             
         000m      100m 12th 1500m 200m 400m  ... younis yousuf zaheer zealand zimbabwe zurich
0     0.00000  0.000000  0.0   0.0  0.0  0.0  ...    0.0    0.0    0.0     0.0      0.0    0.0
1     0.00000  0.000000  0.0   0.0  0.0  0.0  ...    0.0    0.0    0.0     0.0      0.0    0.0
2     0.00000  4.236648  0.0   0.0  0.0  0.0  ...    0.0    0.0    0.0     0.0      0.0    0.0
3     0.00000  0.000000  0.0   0.0  0.0  0.0  ...    0.0    0.0    0.0     0.0      0.0    0.0
4    13.27767  0.000000  0.0   0.0  0.0  0.0  ...    0.0    0.0    0.0     0.0      0.0    0.0
..        ...       ...  ...   ...  ...  ...  ...    ...    ...    ...     ...      ...    ...
732   0.00000  0.000000  0.0   0.0  0.0  0.0  ...    0.0    0.0    0.0     0.0      0.0    0.0
733   0.00000  0.000000  0.0   0.0  0.0  0.0  ...    0.0    0.0    0.0     0.0      0.0    0.0
734   0.00000  0.000000  0.0   0.0  0.0  0.0  ...    0.0    0.0    0.0     0.0      0.0    0.0
735   0.00000  0.000000  0.0   0.0  0.0  0.0  ...    0.0    0.0    0.0     0.0      0.0    0.0
736   0.00000  0.000000  0.0   0.0  0.0  0.0  ...    0.0    0.0    0.0     0.0      0.0    0.0

[737 rows x 2128 columns]

>>>
```
