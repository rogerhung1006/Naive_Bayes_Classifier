# Sentiment Analysis with Self-built Naive Bayes Classifier
## Introduction
To enhance my understanding in probabilistic machine learning classifier, I decided to initiate this sentiment analysis project. In this binary classification project, I attempted to build a Naive Bayes Classifier from sctrach. I used the NLTK movie review corpus. 
Reviews are separated into a training set (80% of the data) and a development set (10% of the data). Within each set, reviews are sorted by sentiment (positive/negative). The files are already tokenized. Each review is in its own file.

## Notes
- In the Feature Selection process, I selected the features by computing mutual infomation for each word
- For classifier evaluation, I created a confusion maxtrix that takes 'negative' as 0 and 'positive' as 1. Additionally, 
I computed precision, recall, and F1 score for each class, as well as the overall accuracy. I personally evaluated the classifier based on precision and recall

## Two assumptions of Naive Bayes model
Naive Bayes is a generative model that makes 1.) the bag of words assumption (position doesn’t matter) and 2.) the conditional independence assumption (words are conditionally independent of each other given the class).

## Analysis and improvement
At first, I tried to use the top 100 features with highest information gain; however, the result turned out to be quite disappointing. Both the F1 score and accuracy were quite low, and the precision wasn't ideal either (as shown in the following graph).

![Screen Shot 2020-03-08 at 12 28 23 AM](https://user-images.githubusercontent.com/60050802/76157216-db30dd00-60d3-11ea-9fd1-e55d11a07d51.png)

To optimize the classifier, I implemented a self-defined grid search function, which serves the purpose of looking for the best number of features and training the model based on the selected features. Lucklily, this time we derived a result that is more pleasant. 

![Screen Shot 2020-03-08 at 12 10 54 AM](https://user-images.githubusercontent.com/60050802/76157027-55139700-60d1-11ea-9b17-d74e56fa2840.png)

