# Word2Vec with CBOW and Skipgram
This project aims to learn high-quality word embeddings using the Word2Vec algorithm with the continuous bag of words (CBOW) and skipgram models. The goal is to build a model that can learn vector representations of words from a large corpus of text, specifically the Shams-Daftar1 Persian poem dataset, which can be used in a variety of natural language processing tasks such as sentiment analysis, text classification, and machine translation.

## Dataset
The dataset used for this project is the Shams-Daftar1 Persian poem dataset, which is a collection of poems from the Persian poet Rumi. 

## Approach
The project will use the Word2Vec algorithm to learn word embeddings from the Shams-Daftar1 Persian poem dataset. The following steps will be taken:

- Data Preparation: The dataset will be preprocessed and cleaned to remove any noise and irrelevant information.
- Corpus Creation: The preprocessed text will be used to create a corpus of text, which will be used to train the Word2Vec models.
- CBOW Model: A Word2Vec model with the continuous bag of words (CBOW) architecture will be trained on the corpus to learn word embeddings.
- Skipgram Model: A Word2Vec model with the skipgram architecture will be trained on the same corpus to learn word embeddings.
- Model Evaluation: The models will be evaluated on a word similarity task to measure the quality of the learned word embeddings.
- UI Development: A Streamlit-based user interface will be developed to allow users to interact with the trained models and explore the learned word embeddings.

## Requirements
- python
- numpy
- pandas
- tensorflow
- keras
- streamlit
