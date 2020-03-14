# Sentiment Analysis with Self-Built Naive Bayes Classifier
## Introduction
To enhance my understanding in probabilistic machine learning classifier, I decided to initiate a sentiment analysis project. In this binary classification project, I attempted to build a Naive Bayes Classifier from sctrach and utilized the self-built classifier to conduct the sentiment classification or say, the extraction of sentiment (the positive or negative orientation that a writer expresses toward some object). In this project, I used the NLTK movie review corpus. Reviews are separated into a training set (80% of the data) and a development set (10% of the data). Within each set, reviews are sorted by sentiment (positive / negative). The files are already tokenized. Each review is in its own file. In the project, I follow the guidance from Jurafsky and Martin’s Speech and Language Processing.<br>
<br>

## Notes
- In the Feature Selection process, I selected the features by computing mutual infomation for each word
- For classifier evaluation, I created a confusion maxtrix that takes 'negative' as 0 and 'positive' as 1. Additionally, 
I computed precision, recall, and F1 score for each class, as well as the overall accuracy. I personally evaluated the classifier based on precision and recall<br>
<br>

## Two assumptions of Naive Bayes Model
Naive Bayes is a generative model that makes 1.) the bag of words assumption (position doesn’t matter) and 2.) the conditional independence assumption (words are conditionally independent of each other given the class).<br>
<br>

## Main Function Insturction
**- Trianing function**<br >
This function serves the purpose of training the Naive Bayes Classifier. It takes two inputs, one is training data and the other is the select rate, which is the hyperparameter that user can define themselves. The training function computes the  maximum likelihood estimate. I commplete this task by using frequencies in the data. For example, I derive the maximum likelihood estimate of P(wi|c) by assuming a feature is just the existence of a word in the document’s bag of words, and computing P(wi|c) as the fraction of times the word wi appears among all words in all documents of class c. Note that I use add-1 smoothing in this case to avoid getting 0 in probability when ecountering unseen words.
<br>

**- Test function**<br >
This function serves the purpose of applying our classifier to the testing data and taking the argmax to find the most probable class. Note that this function would generate a dictionary 'results' such that:<br>
results[filename][‘correct’] = correct class
results[filename][‘predicted’] = predicted class


![T4](https://user-images.githubusercontent.com/60050802/76669506-48c08b80-6562-11ea-82fd-d628355260c1.png)<br>
Figure 1. The naive Bayes algorithm, using add-1 smoothing (From Jurafsky and Martin’s Speech and Language Processing (Chapter 4)<br >
<br>

**- Evaluate function**<br >
This function would, given the results of test, compute precision, recall, and F1 score for each class, as well as the overall accuracy, and print out the performance of the classifier.
<br>

**- Select feature function**<br >
The select_features function take two inputs as well, train_set and select_rate. The function basically select features by computing mutual information for each word, which is a value that falls between 0 and 1, and selects the top n features with highest information gain. (n is decided based on the the selec_rate that user defined in the first place)
<br>

**- Grid search function**<br >
The grid_search take training set, validating set and a list of select rates as input and generates a dataframe containing the performance of classifier on the basis of different possible combination of features. In this case, I use F1 score and accuracy as my main metrics to rank the classfier.

![dataframe](https://user-images.githubusercontent.com/60050802/76670519-57f60800-6567-11ea-9582-38b7a2d4e43c.png)<br>
Figure 2. Partial result of the dataframe<br>


![plot](https://user-images.githubusercontent.com/60050802/76670542-778d3080-6567-11ea-93f4-9557a15e9fe8.png)<br>
Figure 3. Performance Plot<br>
<br>

## Analysis and Improvement
At first, without doing grid search, I tried to use the top 100 features with highest information gain; however, the result turned out to be quite disappointing. Both the F1 score and accuracy were quite low, and the precision wasn't ideal either (as shown in the following graph).

![Screen Shot 2020-03-08 at 12 28 23 AM](https://user-images.githubusercontent.com/60050802/76157216-db30dd00-60d3-11ea-9fd1-e55d11a07d51.png)

To optimize the classifier, I implemented a self-defined grid search function, which serves the purpose of looking for the best number of features and training the model based on the selected features. Lucklily, this time we derived a result that is more pleasant. 


![final_result](https://user-images.githubusercontent.com/60050802/76669654-02b7f780-6563-11ea-9814-e24fc248975d.png)



