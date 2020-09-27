# Code excerpts

...

```python

>>> import texthero as hero
>>> import pandas as pd
>>> import pyLDAvis
>>> 
>>> # Load an example dataset.
>>> df = pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")[["text"]]
>>> 
>>> # Clean and tokenize all documents.
>>> df['text_preprocessed'] = df['text'].pipe(hero.clean).pipe(hero.tokenize)
>>> 
>>> # Calculate a document term matrix (with tfidf)
>>> df_document_term = df['text_preprocessed'].pipe(hero.tfidf)
>>> 
>>> # Use LDA to get a matrix relating documents to abstract "topics".
>>> df_document_topic = hero.lda(df_document_term,n_components=5)
>>> 
>>> # Calculate document-topic and topic-term matrix
>>> df_document_topic, df_topic_term = hero.topic_matrices(df_document_term, df_document_topic)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
>>> 
>>> # l1-Normalize the topic matrices.
>>> df_document_topic_distribution = hero.normalize(df_document_topic, norm="l1")
>>> df_topic_term_distribution = hero.normalize(df_topic_term, norm="l1")
>>> 
>>> # Get the LDAvis figure with relevant words per topic.
>>> figure = hero.relevant_words_per_topic(
... df_document_term, df_document_topic_distribution, df_topic_term_distribution, return_figure=True
... ) 
>>> pyLDAvis.display(figure)
```
