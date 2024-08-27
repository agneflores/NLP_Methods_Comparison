# NLP Methods Comparison
A detailed notebook for this project can be found here : 

______________________________________________________________________________________________________________________________________________________________________________________________________________________________

### The Objective
The objective of this project is to preprocess and analyze text data using various techniques to improve classification performance. We will apply text preprocessing methods, including stemming and lemmatizing, and utilize CountVectorizer and TfidfVectorizer for feature extraction. We will then evaluate the performance of different classification algorithms, namely Logistic Regression, Decision Tree, and Multinomial Naive Bayes, by comparing their accuracy and computational efficiency. The final goal is to identify the best-performing model and present the results, including the best parameters and scores, in a clear and comprehensive format.

______________________________________________________________________________________________________________________________________________________________________________________________________________________________

### Project Outline
1. Import required libraries

2. Connect to the data source and explore it

3. Text preprocessing:Stemming, Lemmatizing, CountVectorizer and TfidifVectorizer

4. Classification: LogisticRegression, DecisionTreeClassifier, and MultinomialNB

5. Performance analysis:accuracy and speed

6. Conclusion summary

______________________________________________________________________________________________________________________________________________________________________________________________________________________________

### The Data
The dataset below is from [kaggle]() and contains a dataset named the "ColBert Dataset" created for this [paper](https://arxiv.org/pdf/2004.12765.pdf).  The project will use the text column to classify whether or not the text was humorous. 

______________________________________________________________________________________________________________________________________________________________________________________________________________________________

### Libraries
import pandas as pd

import numpy as np

import time 

import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB

from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.corpus import stopwords

______________________________________________________________________________________________________________________________________________________________________________________________________________________________

### Summary of Results

<img width="800" alt="image" src="https://github.com/user-attachments/assets/fbf9bb9b-1b59-4182-a620-b00433653391">

______________________________________________________________________________________________________________________________________________________________________________________________________________________________

### Analysis

#### Accuracy
Logistic Regression with CountVectorizer and Stemming achieved the highest accuracy of 0.887800. This indicates that Logistic Regression performed the best in classifying the text data, with a significant margin over the other classifiers.

MultinomialNB with CountVectorizer and Stemming had an accuracy of 0.879383, which is also high but slightly lower than Logistic Regression.

DecisionTreeClassifier with TfidfVectorizer and Stemming achieved an accuracy of 0.824233. While it still performed well, it was less accurate compared to the other models.

#### Training Time
MultinomialNB had the shortest training time of 0.019012 seconds, making it the fastest classifier. This is expected as MultinomialNB is generally efficient with large datasets, particularly for text classification.

Logistic Regression took 1.802860 seconds to train, which is relatively quick but significantly longer than MultinomialNB.

DecisionTreeClassifier took 41.925832 seconds, making it the slowest among the three. This is expected since decision trees are known for their complexity, especially with large feature sets and deep trees.

______________________________________________________________________________________________________________________________________________________________________________________________________________________________

### Conclusion

If you prioritize accuracy and can afford slightly longer training times, Logistic Regression is the best choice. If you need the fastest training time and can accept a minor reduction in accuracy, MultinomialNB is preferable. DecisionTreeClassifier provides a decent accuracy but at a much higher training cost, making it less suitable if speed is a concern.

In summary, I would recommend Logistic Regression for best accuracy, while MultinomialNB  for best speed in this particular case study. The choice of vectorizer (CountVectorizer vs. TfidfVectorizer) and text preprocessing technique (Stemming vs. Lemmatizing) showed minimal impact on the overall best performing model, but these aspects could still be worth exploring further depending on specific requirements and constraints of the next task.
