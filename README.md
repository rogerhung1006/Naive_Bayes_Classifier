# Naive_Bayes_Classifier
## Introduction
This is my first attemp to build a Naive Bayes Classifier from sctrach. I used the NLTK movie review corpus. 
Reviews are separated into a training set (80% of the data) and a development set 
(10% of the data). Within each set, reviews are sorted by sentiment (positive/negative). The files are already tokenized. 
Each review is in its own file.

## Notes
1. In the Feature Selection process, I selected the features by computing mutual infomation for each word.
2. For classifier evaluation, I created a confusion maxtrix that takes 'negative' as 0 and 'positive' as 1. Additionally, 
I computed precision, recall, and F1 score for each class, as well as the overall accuracy
