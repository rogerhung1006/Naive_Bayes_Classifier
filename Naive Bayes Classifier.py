#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Naive Bayes Classifier and Evaluation

import os
import numpy as np
import re
from collections import defaultdict

class NaiveBayes():

    def __init__(self):
        # be sure to use the right class_dict for each data set
        self.class_dict = {0: 'neg', 1: 'pos'}
        self.prior = None
        self.v = None
        self.likelihood = None
        self.class_count = np.zeros((2,1), dtype = int)


    # Create a function that updates the conditional counts    
    def update_dict(self, x, y):
        for k,v in x.items():
            if k in y.keys():
                y[k] += 1
            else:
                y[k] = 1
        return(y)
    
    def train(self, train_set):
        self.features = self.select_features(train_set)
        # iterate over training documents
        word_count_pos = defaultdict(int)
        word_count_neg = defaultdict(int)
        
        #for root, dirs, files in os.walk(train_set):
            #if ''.join(files).startswith('.'):
                #continue
            #for name in files:
                #with open(os.path.join(root, name)) as f:
                # collect class counts and feature counts
                    #word_list = re.split(r'[^A-Za-z\']+', f.read())
                #word.append(word_list)
                
        for word in self.neg_count.keys():
            if word in self.features:
                word_count_pos[word] += self.neg_count[word]
        for word in self.pos_count.keys():
            if word in self.features:
                word_count_neg[word] += self.pos_count[word]

                
                
        # get total word counts in each class using add-1 method
        v_count = len(self.v)
        neg_total = v_count + sum(self.neg_count.values())
        pos_total = v_count + sum(self.pos_count.values())

        # calculate likelihoods with add-1 method
        self.likelihood = np.array([[], []])
        for word in self.features:
            self.likelihood = np.append(self.likelihood, [[(word_count_neg[word] + 1) / neg_total], [(word_count_pos[word] + 1) / pos_total]], axis=1)
        self.likelihood = np.log(self.likelihood)
        print(self.likelihood)


   
    def test(self, dev_set):
        results = defaultdict(dict)
        features = list(self.features)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            if ''.join(files).startswith('.'):
                continue
            for name in files:
                # Get the correct class from the root and insert it into the dict
                results[name] = {'correct': root.split(sep = '/')[-1]}
                with open(os.path.join(root, name)) as f:
                    # create a empty feature count array for each document
                    word_count = defaultdict(int)
                    # create feature vectors for each document
                    word_list = np.array(re.split(r'[^A-Za-z\']+', f.read()))
                    for word in word_list:
                        if word in features:
                            word_count[word] += 1
                    
                    feature_count = np.zeros((len(features),1))
                    for i in range(len(features)):
                        feature_count[i] = word_count[features[i]]
                    prediction = np.dot(self.likelihood, feature_count) + self.prior
                    # Get the max of the value and classify the result
                    # Get most likely class
                    if np.argmax(prediction) == 0:
                        results[name].update({'predicted' : 'neg'})
                    else: results[name].update({'predicted' : 'pos'})
      
        return results



    def evaluate(self, results):
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        
        for key in results.values():
            if key['correct'] == key['predicted'] and key['predicted'] == 'neg':
                confusion_matrix[0][0] += 1
            if key['correct'] == key['predicted'] and key['predicted'] == 'pos':
                confusion_matrix[1][1] += 1
            if key['correct'] != key['predicted'] and key['correct'] == 'pos':
                confusion_matrix[1][0] += 1
            if key['correct'] != key['predicted'] and key['correct'] == 'neg':
                confusion_matrix[0][1] += 1

        precision = confusion_matrix[0][0] / sum(confusion_matrix[0])
        recall = confusion_matrix[0][0] / np.sum(confusion_matrix, axis = 0)[0]
        F1 = 2*precision*recall / (precision+recall)
        accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1]) / confusion_matrix.sum().sum()
        print('precision = ' + str(precision) + os.linesep, 'recall = ' + str(recall) + os.linesep, 'F1 Score = ' + str(F1) + os.linesep, 'accuracy = ' + str(accuracy) + os.linesep)
        print('Confusion Matrix: ')
        print(confusion_matrix)
        

        
    def select_features(self, train_set):
        self.neg_count = defaultdict(int)
        self.pos_count = defaultdict(int)
        total_word_count = defaultdict(int)
        # iterate over training documents
        total_words = []
        temp_class_count = []
        word_count_class = []
        
        for root, dirs, files in os.walk(train_set):
            if ''.join(files).startswith('.'):
                continue
            # Calculate the sum of the word count in each class 
            word_count_class_temp = 0
            
            # temporary class count. will later be added to the class_count numpy array
            temp_class_count.append([len(files)])
            for name in files:
                with open(os.path.join(root, name)) as f:
                    word_list = re.split(r'[^A-Za-z\']+', f.read())
                    
                    # Initiate two dictionaries to store the word counts in two classes for each document
                    neg_temp = defaultdict(int)
                    pos_temp = defaultdict(int)
                    # collect total word counts
                    for i in word_list:
                        word_count_class_temp += 1
                        total_word_count[i] += 1
                        
                        # collect conditional class counts
                        if root.split(sep = "/")[-1] == 'neg':
                            neg_temp[i] = 1                                    
                        elif root.split(sep = "/")[-1] == 'pos':
                            pos_temp[i] = 1

                    self.update_dict(neg_temp, self.neg_count)
                    self.update_dict(pos_temp, self.pos_count)
                    total_words.extend(word_list)
                    
            word_count_class.append(word_count_class_temp) 
            
                    
                
        # Caculate document counts by classes. In this case, two classes
        self.class_count = (self.class_count + temp_class_count)
        
        # Calculate the V
        self.v = set([word for word in total_words if word != ''])
   
        
        # The following is code to compute the mutual information for selected features
        # Normalize counts to probabilities
        prob_class = self.class_count / sum(self.class_count)
        
        # Take logs and assign to self.prior
        self.prior = np.log(prob_class)
        
        # Here I follow the math formula for mutual infomation and hence, name the variable
        # to fisrt, second and third line
        class_entropy = 0
        for i in range(len(self.prior)):
            class_entropy += self.prior[i] * prob_class[i]
        class_entropy = -class_entropy

        mutual_info = defaultdict(int) 
        total_word_count_sum = sum(total_word_count.values())
        for word in self.v:
            word_prob = total_word_count[word] / total_word_count_sum  # p(w)
            word_prob_in_neg_class = (self.neg_count[word] + 1) / ((self.neg_count[word]) + (self.pos_count[word])) # P(c1|w)
            word_prob_in_pos_class = (self.pos_count[word] + 1) / ((self.neg_count[word]) + (self.pos_count[word])) # P(c2|w)
            not_word_in_neg_class = (sum(self.neg_count.values()) - self.neg_count[word]) / (sum(self.class_count) - # P(c1|not w)
                                            (self.neg_count[word] + self.pos_count[word]))
            not_word_in_pos_class = (sum(self.pos_count.values()) - self.pos_count[word]) / (sum(self.class_count) - # P(c2|not w)
                                            (self.neg_count[word] + self.pos_count[word]))

            conditional_entropy_word = word_prob * (word_prob_in_neg_class*np.log(word_prob_in_neg_class) + word_prob_in_pos_class*np.log(word_prob_in_pos_class))
            conditional_entropy_notword = word_prob * (not_word_in_neg_class*np.log(not_word_in_neg_class) + not_word_in_pos_class*np.log(not_word_in_pos_class))
            mutual_info[word] = class_entropy + conditional_entropy_word + conditional_entropy_notword
        mutual_info = sorted(mutual_info.items(), key=lambda x: x[1], reverse=True)[0:100]
        features_selected = set([feature[0] for feature in mutual_info])

            
        return(features_selected)
         
        
         

if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')
    results = nb.test('movie_reviews/dev')
    nb.evaluate(results)


