# Sentiment Analysis with Self-Built Naive Bayes Classifier
## Introduction
To enhance my understanding in probabilistic machine learning classifier, I decided to initiate a sentiment analysis project. In this binary classification project, I attempted to build a Naive Bayes Classifier from sctrach and utilized the self-built classifier to conduct the sentiment classification or say, the extraction of sentiment (the positive or negative orientation that a writer expresses toward some object). In this project, I used the NLTK movie review corpus. Reviews are separated into a training set (80% of the data) and a development set (10% of the data). Within each set, reviews are sorted by sentiment (positive / negative). The files are already tokenized. Each review is in its own file.

## Notes
- In the Feature Selection process, I selected the features by computing mutual infomation for each word
- For classifier evaluation, I created a confusion maxtrix that takes 'negative' as 0 and 'positive' as 1. Additionally, 
I computed precision, recall, and F1 score for each class, as well as the overall accuracy. I personally evaluated the classifier based on precision and recall

## Two assumptions of Naive Bayes Model
Naive Bayes is a generative model that makes 1.) the bag of words assumption (position doesn’t matter) and 2.) the conditional independence assumption (words are conditionally independent of each other given the class).

## Main Function Insturction
**- Trianing function**<br >
This function serves the purpose of training the Naive Bayes Classifier. It takes two inputs, one is training data and the other is the select rate, which is the hyperparameter that user can define themselves. The training function computes the  maximum likelihood estimate. I commplete this task by using frequencies in the data. For example, I derive the maximum likelihood estimate of P(wi|c) by assuming a feature is just the existence of a word in the document’s bag of words, and computing P(wi|c) as the fraction of times the word wi appears among all words in all documents of class c. Note that I use add-1 smoothing in this case to avoid getting 0 in probability when ecountering unseen words.


![T4](https://user-images.githubusercontent.com/60050802/76669506-48c08b80-6562-11ea-82fd-d628355260c1.png)

Figure 1. The naive Bayes algorithm, using add-1 smoothing (From Jurafsky and Martin’s Speech and Language Processing (Chapter 4)<br >


**- Test function**<br >



- Evaluate function

- Select features function



## Analysis and Improvement
At first, I tried to use the top 100 features with highest information gain; however, the result turned out to be quite disappointing. Both the F1 score and accuracy were quite low, and the precision wasn't ideal either (as shown in the following graph).

![Screen Shot 2020-03-08 at 12 28 23 AM](https://user-images.githubusercontent.com/60050802/76157216-db30dd00-60d3-11ea-9fd1-e55d11a07d51.png)

To optimize the classifier, I implemented a self-defined grid search function, which serves the purpose of looking for the best number of features and training the model based on the selected features. Lucklily, this time we derived a result that is more pleasant. 

![final_result](https://user-images.githubusercontent.com/60050802/76669654-02b7f780-6563-11ea-9814-e24fc248975d.png)


To optimize the classifier, I implemented a self-defined grid search function, which serves the purpose of looking for the best number of features and training the model based on the selected features. Lucklily, this time we derived a result that is more pleasant. 
