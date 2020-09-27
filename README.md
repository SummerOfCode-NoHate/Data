# Code excerpts

...

```python

>>> import texthero as hero
>>> import pandas as pd
>>> from flair.embeddings import TransformerDocumentEmbeddings
>>> 
>>> # Load an example dataset.
>>> df = pd.read_csv(
...    "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
... )[["text"]] # The dataset consists of 737 texts about sports.
>>> 
>>> # Clean and tokenize all documents.
>>> df['text_preprocessed'] = df['text'].pipe(hero.clean).pipe(hero.tokenize)
>>> 
>>> # Load BERT embedding.
>>> embedding = TransformerDocumentEmbeddings("bert-base-uncased")
>>> 
>>> # Embed documents.
>>> df['bert_embedding'] = df['text_preprocessed'].pipe(hero.embed, embedding)
>>> 
>>> df
                                                  text  ...                                     bert_embedding
0    Claxton hunting first major medal\n\nBritish h...  ...  [4.410829544067383, -0.48115718364715576, -1.6...
1    O'Sullivan could run in Worlds\n\nSonia O'Sull...  ...  [-1.475942611694336, 1.3640360832214355, -0.06...
2    Greene sets sights on world title\n\nMaurice G...  ...  [5.091068744659424, 2.6682629585266113, -4.386...
3    IAAF launches fight against drugs\n\nThe IAAF ...  ...  [2.5341577529907227, -4.693658351898193, -3.11...
4    Dibaba breaks 5,000m world record\n\nEthiopia'...  ...  [4.672342300415039, -0.9142539501190186, -2.91...
..                                                 ...  ...                                                ...
732  Agassi into second round in Dubai\n\nFourth se...  ...  [-0.07187545299530029, 4.893282413482666, -0.6...
733  Mauresmo fights back to win title\n\nWorld num...  ...  [2.577310800552368, -4.6007585525512695, -1.75...
734  Federer wins title in Rotterdam\n\nWorld numbe...  ...  [2.9214956760406494, 5.587845325469971, 1.8893...
735  GB players warned over security\n\nBritain's D...  ...  [0.4018148183822632, 5.7114362716674805, -4.57...
736  Sharapova overcomes tough Molik\n\nWimbledon c...  ...  [12.387837409973145, -1.541538953781128, 2.345...

[737 rows x 3 columns]

```
