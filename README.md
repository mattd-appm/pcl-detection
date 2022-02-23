# Detection of Patronizing and Condescending Language
Detecting patronizing and condescending language (pcl) using natural language processing

## Data
The dataset under analysis was provided by SemEval and inspired by the paper “Don't Patronize 
Me! An annotated Dataset with Patronizing and Condescending Language Towards Vulnerable 
Communities” (Perez-Almendros et al., 2020)

## Methods Used
In this analysis, I used stemming, lemmatization, and extraneous word/character removal first to 
preprocess the data. I then created different word embeddings through tf-Idf and word2vec. I used 
SMOTE to balance the data before passing it through four different models. These four models included 
logistic regression on both word embeddings, Naïve Bayes classifiers on the Tf-Idf word embeddings, 
and a neural net on the word2vec word embeddings. 

## Challenges
1. Small dataset
2. Subtlety and ambiguity in definition of patronizing language
3. Unbalanced data with 9.5% positive examples
4. Overfitting

## F1 scores in Models
1. Logistic Regression Using Word2vec Embeddings
   - F1 metric in the test dataset: 0.2998
2. Logistic Regression with tf-Idf 
   - F1 metric in the test dataset: 0.3778
3. Naïve-Bayes Classifier with tf-Idf
   - F1 metric in the test dataset: 0.3806 
4. Feed-Forward Neural Network with word2vec Embeddings
   - F1 metric in the test dataset: 0.2824
