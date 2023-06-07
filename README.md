# Preprocessing and Classification of Persian Text

This repository contains code for preprocessing and classification of Persian text using natural language processing techniques. The code is written in Python and utilizes the `hazm` library for text preprocessing and the `gensim` library for training a Word2Vec model.

## Installation

To run the code, you need to install the required dependencies. You can install them using the following commands:

```bash
!pip install hazm
!pip install torch
!pip install gensim
```

## Preprocessing

The preprocessing code performs several steps to clean and tokenize the text. It uses the `Normalizer` from the `hazm` library to normalize the text, `word_tokenize` to tokenize the normalized text, and a combination of stemming and lemmatization techniques to extract the stems of words and remove stop words.

```python
from hazm import Normalizer, word_tokenize, Stemmer, Lemmatizer, stopwords_list
import torch
import numpy as np

# Preprocessing settings
normalizer = Normalizer()
stemmer = Stemmer()
lemmatizer = Lemmatizer()
stopwords = set(stopwords_list())

# Preprocessing function
def preprocess_text(text):
    normalized_text = normalizer.normalize(text)
    tokens = word_tokenize(normalized_text)
    preprocessed_tokens = []
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        lemmatized_token = lemmatizer.lemmatize(token)
        if stemmed_token not in stopwords:
            preprocessed_tokens.append(stemmed_token)
    preprocessed_text = ' '.join(preprocessed_tokens)
    return preprocessed_text

# Example text
text = "متن نمونه برای پیش‌پردازش."

# Preprocess the text
preprocessed_text = preprocess_text(text)

# Print the preprocessed text
print(preprocessed_text)

# Convert the preprocessed text to an embedded tensor
embedding_dim = 20
embedding = np.zeros(embedding_dim)
text_embedding = [embedding] * len(preprocessed_text.split())
text_tensor = torch.tensor(text_embedding)
print(text_tensor)
```

## Word2Vec Embedding

The code also includes an example of training a Word2Vec model on a set of sentences. The sentences are tokenized and then used to train the Word2Vec model from the `gensim` library. The trained model can be used to obtain word embeddings.

```python
from hazm import word_tokenize
from gensim.models import Word2Vec

# Dataset
sentences = [
    "جمله اول برای آموزش مدل.",
    "جمله دوم برای آموزش مدل.",
    "جمله سوم برای آموزش مدل."
]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

# Train Word2Vec model
embedding_dim = 100
model = Word2Vec(tokenized_sentences, vector_size=embedding_dim, min_count=1)

# Test words
word = "جمله"
if word in model.wv.key_to_index:
    word_embedding = model.wv[word]
    print(f"Word Embedding for '{word}': {word_embedding}")
else:
    print(f"Word '{word}' not found in the model.")
```

## License

This project is licensed under a Free License.


Please refer to the [GitHub repository](https://github.com/armansouri9/Preprocessing-and-classification-of-Persian-text) for more details and the complete code.

Note: The code samples provided here are just excerpts and may require additional
