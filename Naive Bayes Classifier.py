#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Naive Bayes Classifier and Evaluation

import os
import numpy as np
import re
from collections import defaultdict
import pandas as pd

class NaiveBayes():

    def __init__(self):
        self.class_dict = {0: 'neg', 1: 'pos'}
        self.prior = None
        self.v = None
        self.likelihood = None
        self.class_count = np.zeros((2,1), dtype = int)
        self.neg_count = defaultdict(int)
        self.pos_count = defaultdict(int)


    # Create a function that updates the conditional counts    
    def update_dict(self, x, y):
        for k,v in x.items():
            if k in y.keys():
                y[k] += 1
            else:
                y[k] = 1
        return(y)
    
    # Train the classifier by selected features
    def train(self, train_set, feature_select_rate):
        self.features = self.select_features(train_set, feature_select_rate)
        # Iterate over training documents
        word_count_pos = defaultdict(int)
        word_count_neg = defaultdict(int)
        
        for root, dirs, files in os.walk(train_set):
            if ''.join(files).startswith('.'):
                continue
            for name in files:
                with open(os.path.join(root, name)) as f:
                #collect class counts and feature counts
                    word_list = re.split(r'[^A-Za-z\']+', f.read())
                    for word in word_list:
                        if word in self.features and root.split(sep = "/")[-1] == 'pos':
                            word_count_pos[word] += 1
                        if word in self.features and root.split(sep = "/")[-1] == 'neg':
                            word_count_neg[word] += 1

   
        # Get total word counts in each class using add-1 method
        neg_total = len(self.features) + sum(self.neg_count.values())
        pos_total = len(self.features) + sum(self.pos_count.values())

        # Calculate likelihoods with add-1 method
        self.likelihood = np.array([[], []])
        for word in self.features:
            self.likelihood = np.append(self.likelihood, [[(word_count_neg[word] + 1) / neg_total], [(word_count_pos[word] + 1) / pos_total]], axis=1)
        self.likelihood = np.log(self.likelihood)
        
        return


    
    def test(self, dev_set):
        results = defaultdict(dict)
        features = list(self.features)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            if ''.join(files).startswith('.'):
                continue
            for name in files:
                # Get the correct class from the root and insert it into the dict
                results[name] = {'correct': 0 if root.split(sep = '/')[-1] == 'neg' else 1}
                with open(os.path.join(root, name)) as f:
                    # create a empty feature count array for each document
                    word_count = defaultdict(int)
                    # create feature vectors for each document
                    word_list = np.array(re.split(r'[^A-Za-z\']+', f.read()))
                    for word in word_list:
                        if word in features:
                            word_count[word] += 1

                    feature_count = np.zeros((len(features), 1))
                    for i in range(len(features)):
                        feature_count[i] = word_count[features[i]]

                
                    prediction = np.dot(self.likelihood, feature_count) + self.prior
                    # Get the max of the value and classify the result
                    # Get most likely class
                    if np.argmax(prediction) == 0:
                        results[name].update({'predicted' : 0})   # if neg then 0 
                    else: results[name].update({'predicted' : 1}) # if pos then 1

        return results


    # Define a function to evaluate the result
    def evaluate(self, results):
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))

        for _, result in results.items():
            confusion_matrix[result['correct']][result['predicted']] += 1
            
        precision = confusion_matrix[1, 1] / confusion_matrix[:, 1].sum()
        recall = confusion_matrix[1, 1] / confusion_matrix[1, :].sum()
        F1 = (2 * precision * recall) / (precision + recall)
        accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

        print('Confusion Matrix: ')
        print(confusion_matrix)
        print('\n------- Evaluation -------\n'
              + f'Precision: {round(precision, 3)}\nRecall: {round(recall, 3)}\n'
              + f'F1 Score: {round(F1, 3)}\nAccuracy: {round(accuracy, 3)}\n')
        return(F1, accuracy)

        
    def select_features(self, train_set, select_rate):
        total_word_count = defaultdict(int)
        # iterate over training documents
        total_words = []
        temp_class_count = []

        for root, dirs, files in os.walk(train_set):
            if ''.join(files).startswith('.'):
                continue
            
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
                        if i != '':
                         # Calculate word count for each class
                            total_word_count[i] += 1
                        
                        # collect conditional class counts
                        if root.split(sep = "/")[-1] == 'neg':
                            neg_temp[i] = 1                                    
                        elif root.split(sep = "/")[-1] == 'pos':
                            pos_temp[i] = 1

                    self.update_dict(neg_temp, self.neg_count)
                    self.update_dict(pos_temp, self.pos_count)
                    total_words.extend(word_list)
                    

        # Caculate document counts by classes. In this case, two classes
        self.class_count = (self.class_count + temp_class_count)

        # Calculate the V
        self.v = set([word for word in total_words if word != ''])
           
        # The following is code to compute the mutual information for selected features
        # Normalize counts to probabilities
        prob_class = self.class_count / sum(self.class_count)
        
        # Take logs and assign to self.prior
        self.prior = np.log(prob_class)

        
        # Here I use mutual infomation to select features.
        # Class entropy
        class_entropy = 0
        for i in range(len(self.prior)):
            class_entropy += self.prior[i] * prob_class[i]
        class_entropy = -class_entropy

        # Conditional entropy
        mutual_info = defaultdict(int) 
        total_word_count_sum = sum(total_word_count.values())

        for word in self.v:
            word_prob = total_word_count[word] / total_word_count_sum  # p(w)
            word_prob_in_neg_class = (self.neg_count[word] + 1) / ((self.neg_count[word] + 2) + (self.pos_count[word])) # P(c1|w)
            word_prob_in_pos_class = (self.pos_count[word] + 1) / ((self.neg_count[word] + 2) + (self.pos_count[word])) # P(c2|w)
            not_word_in_neg_class = ((self.class_count[0][0] - self.neg_count[word]) + 1) / ((sum(self.class_count)[0] - # P(c1|not w)
                                            (self.neg_count[word] + self.pos_count[word])) + 2)
            not_word_in_pos_class = ((self.class_count[1][0] - self.pos_count[word]) + 1) / ((sum(self.class_count)[0] - # P(c2|not w)
                                            (self.neg_count[word] + self.pos_count[word])) + 2)

            conditional_entropy_word = word_prob * (word_prob_in_neg_class*np.log(word_prob_in_neg_class) + word_prob_in_pos_class*np.log(word_prob_in_pos_class))
            conditional_entropy_notword = (1 - word_prob) * (not_word_in_neg_class*np.log(not_word_in_neg_class) + not_word_in_pos_class*np.log(not_word_in_pos_class))
            mutual_info[word] = class_entropy + conditional_entropy_word + conditional_entropy_notword
            
        number_of_features = int(len(self.v) * select_rate)
        mutual_info = sorted(mutual_info.items(), key=lambda x: x[1], reverse=True)[1:number_of_features]
        features_selected = set([feature[0] for feature in mutual_info])
        #features_selected = set({'creating', 'bad', 'finest', 'made', 'weak', 'better', 'guess', 'believes', 'power', 'while', 'however', 'themes', 'sister', 'save', 'old', 'natural', 'cash', 'agent', 'colors', 'right', 'scott', 'somewhat', 'money', 'form', 'score', 'meant', 'gags', 'atrocious', 'role', 'enjoyable', 'characters', 'mission', 'then', 'using', 'screaming', 'slowly', 'terribly', 'oh', 'first', 'five', 'have', 'song', 'dialogue', 'some', 'present', 'screenwriter', 'screen', 'la', 'edge', 'director', 'sit', 'quite', 'way', 'please', 'big', 'slightly', 'search', 'almost', 'nice', 'remotely', 'harris', 'pain', 'century', 'writers', 'seagal', 'it', 'overall', 'trying', 'give', 'proves', 'guys', 'pure', 'performances', 'help', 'excellent', 'award', 'cheap', 'lives', 'guy', 'looking', 'movie', 'contrived', "tv's", 'off', 'makes', 'pay', 'kill', 'into', 'budget', 'i', 'yeah', 'honest', 'takes', 'sadly', 'comes', 'refreshing', 'visually', "isn't", 'married', "here's", 'definitely', 'fiction', 'see', 'throughout', 'together', 'light', 'brings', 'captures', 'known', 'rated', 'learn', 'rip', 'surprisingly', 'acting', 'our', 'lacks', 'supposed', 'magnificent', 'handle', 'death', 'iii', 'cop', 'ever', 'detail', 'up', 'fake', 'clever', 'particularly', 'value', 'painfully', 'stereotypes', 'cool', 'nicely', 'incredible', 'review', 'flawless', 'suppose', 'terrific', 'hank', 'eve', 'twice', 'animated', 'town', "you're", 'raised', 'your', 'flat', 'loud', 'this', 'looks', 'the', 'fantasy', 'action', 'less', 'thin', 'nonsense', 'issues', 'struggle', 'holds', 'pulp', 'choice', 'watching', 'add', 'father', 'story', 'human', 'just', 'low', 'waste', 'cinema', 'traditional', 'mediocre', 'feels', 'horrible', 'child', 'potentially', 'powerful', 'hired', 'meet', 'voice', 'sports', 'above', 'suspense', 'friendship', 'basically', 'has', 'total', 'cage', 'films', 'bottom', 'endearing', 'each', 'wife', 'drama', 'terrible', 'lethal', 'private', 'innocence', 'intense', 'notch', 'husband', 'else', 'poignant', 'called', 'care', 'society', 'film', 'feature', 'love', 'stopped', 'created', 'good', 'given', 'ben', 'most', 'expected', 'cartoon', 'robin', 'blame', 'worth', 'ending', 'so', 'much', 'fear', 'delivers', 'look', 'weeks', 'explained', 'stunning', 'maintains', 'details', 'disturbing', 'heart', 'than', 'bother', 'various', 'credits', "it's", 'appear', 'clearly', 'plot', 'today', 'mature', 'man', 'refuses', 'generally', 'a', 'dead', 'b', 'lord', 'period', 'they', 'also', 'enjoyed', 'frankly', 'bringing', 'edward', 'study', 'delight', 'believe', 'minutes', 'thirty', 'extremely', 'entertaining', 'dream', 'generate', 'does', 'pretty', 'states', 'able', 'mind', 'neither', 'disappointment', 'predictable', 'needed', 'compelling', 'happy', 'experiences', 'finds', 'decent', 'general', 'sequel', 'subject', 'been', 'are', 'beautiful', 'though', 'throw', 'bit', 'bore', 'sloppy', 'know', 'told', 'atmosphere', 'perfect', 'helps', 'whom', 'directed', 'touches', 'central', 'which', 'woo', 'used', 'sat', 'may', 'american', 'political', 'trailer', 'quiet', 'true', 'saw', 'thanks', 'extra', 'unbelievable', 'accent', 'small', 'enjoy', 'other', 'perhaps', 'earlier', 'important', "let's", 'gives', 'outstanding', 'would', 'shows', 'intelligent', 'saved', 'towards', 'involves', 'released', 'us', 'what', 'special', 'tired', 'person', 'generic', 'kevin', 'that', 'solid', 'nature', 'here', 'minute', 'did', 'day', 'mood', 'course', 'sweet', 'development', 'embarrassing', 'wasted', 'making', 'presumably', 'ideas', 'beautifully', 'both', 'them', 'whatsoever', 'magic', 'how', 'rent', 'ludicrous', 'approach', 'stock', 'events', 'language', 'young', 'villains', 'until', 'garbage', 'suffers', 'robert', 'surprise', 'memorable', 'loved', 'least', 'typical', 'villain', 'such', 'whether', 'of', 'unlike', 'animation', 'back', 'goes', 'very', 'opens', 'although', 'tedious', 'girls', 'offer', 'change', 'fashioned', 'kids', 'views', 'strange', 'brilliant', 'set', 'explanation', 'freddie', 'keep', 'due', 'pointless', 'job', 'uninteresting', 'sorry', 'problem', 'producers', 'musical', 'stupidity', 'car', 'eventually', 'remain', 'luckily', 'their', 'anger', 'ridiculous', 'wonderfully', 'america', 'write', 'satisfying', 'rich', 'watchable', 'force', 'greatest', 'soundtrack', 'asleep', 'often', 'around', 'wait', 'had', 'schumacher', 'an', 'themselves', 'breasts', 'thriller', 'lousy', 'strong', 'sub', 'yes', 'going', 'fantastic', 'inane', 'will', 'success', 'looked', 'dennis', 'thankfully', "that's", 'upon', 'watch', 'inept', 'saturday', 'carry', 'work', 'promising', 'irritating', 'find', 'sex', 'vision', 'dressed', 'use', 'before', 'everything', 'sequence', 'interesting', 'doubt', 'bomb', 'might', 'attention', "what's", 'folks', 'too', 'from', 'except', 'damon', 'running', 'fun', 'its', 'who', 'empty', 'anyway', 'for', 'clich', 'despite', 'named', 'anyone', 'badly', 'as', 'culture', 'on', 'personal', 'high', 'naked', 'truly', 'date', 'somewhere', 'class', 'become', 'take', 'caught', 'skip', 'tragedy', 'bland', 'cheesy', 'means', 'feel', 'wars', "can't", 'problems', 'frightening', 'away', 'theme', 'emotional', 'friends', 'nowhere', 'crap', 'giving', 'considered', 'check', 'comedy', 'repetitive', 'behind', 'long', 'unintentional', 'relationship', 'realistic', 'me', 'fully', 'release', 'if', 'none', 'killed', 'out', 'impression', 'chase', 'years', "we're", 'gary', 'or', 'develops', 'his', 'grade', 'new', 'being', "doesn't", 'setting', 'hand', 'adventure', 'supposedly', 'version', 'entertainment', 'keeps', 'unfortunately', 'including', 'against', 'start', 'dramatic', 'was', 'extraordinary', 'screenplay', 'name', 'whose', 'with', 'straight', 'halfway', 'decided', 'names', 'courage', 'among', 'idea', 'poorly', 'annoying', 'scream', 'blow', 'place', 'talents', 'half', "you'd", 'famous', 'real', 'all', 'controversial', 'disappointing', 'major', 'chuckle', 'nearly', 'failed', 'planet', 'uninvolving', 'vehicle', 'flaws', 'catch', 'ad', 'obviously', 'nevertheless', 'career', 'by', 'seemed', 'dumb', 'key', 'fine', 'frank', 'remains', 'failure', 'wonderful', 'king', 'apparently', 'and', 'tony', 'others', 'rare', 'simple', 'never', 'production', 'oscar', 'court', 'led', 'seem', "aren't", 'remarkable', 'italian', 'pace', 'understanding', 'social', 'should', 'more', 'men', 'short', 'adult', 'thing', 'dull', 'tale', 'thought', 'headed', 'horribly', 'supporting', 'ways', 'wants', "wasn't", 'son', 'greater', 'history', 'became', 'stuff', 'paced', 'contrast', 'awful', 'era', 'rather', 'speech', 'harry', 'complex', 'decades', 'zero', 'share', 'sequences', 'nomination', 'succeeds', 'bunch', 'grand', 'united', 'thrillers', 'lines', 'portrayed', 'boring', 'could', 'no', 'final', 'fascinating', 'law', "they're", 'leads', 'these', 'get', 'many', 'superb', 'sole', 'common', 'novel', 'be', 'ride', 'matter', 'ridiculously', 'guilty', 'won', 'create', 'chemistry', 'lies', 'masterpiece', 'allows', 'promise', 'touch', 'easily', 'moving', 'over', 'respectively', 'cast', 'think', 'begins', 'lonely', 'time', 'near', 'got', 'thinking', 'war', 'appreciate', 'based', 'emotions', 'someone', 'side', 'pathetic', 'he', 'unfunny', 'easy', 'delightful', 'product', 'line', 'falls', 'alas', 'sake', 'deserves', 'even', 'tv', 'comic', 'seeing', 'amounts', 'why', 'project', 'anywhere', 'minor', 'her', 'follow', 'spectacular', 'lame', 'must', 'at', 'unless', 'laugh', 'dark', 'only', 'worse', 'viewer', 'classic', 'indeed', 'answer', 'epic', 'fares', 'material', 'stupid', 'himself', 'cliche', 'experience', 'laughable', 'do', 'either', 'played', 'about', 'same', "i've", 'treat', 'porn', 'there', 'dog', 'successful', 'between', 'exceptional', 'potential', 'structure', "she's", 'well', 'my', 'subtle', 'tries', 'completely', 'written', 'matt', 'wrong', 'game', 'yet', "i'm", 'joy', 'silly', 'excuse', 'academy', 'deal', 'sets', 'move', 'julie', 'freedom', 'certainly', 'sam', 'jennifer', 'age', 'enough', 'include', 'david', 'understand', 'those', 'master', 'nothing', 'doctor', 'balance', 'follows', 'beginning', 'perfectly', "didn't", 'interest', 'art', 'emotionally', 'past', 'warm', 'fair', 'normal', 'happens', 'directing', 'similar', 'michael', "couldn't", 'world', 'uninspired', 'parents', 'random', 'reality', 'future', 'fits', 'whatever', 'hilarious', 'home', 'moments', "year's", 'times', 'insult', 'mother', 'road', 'cinematography', 'ten', 'during', 'filmmakers', 'ill', 'image', 'one', 'filled', 'realize', 'knows', 'whole', 'touching', 'ordinary', 'offensive', 'forgot', 'works', 'batman', 'loves', 'gag', "there's", 'without', 'bored', 'barely', 'partner', 'attempt', 'talent', 'were', 'inside', 'visual', 'compared', 'early', 'german', 'initially', 'detailed', 'teacher', 'unique', 'point', 'ass', 'him', 'seconds', 'entire', 'disappears', 'lets', 'turkey', 'effective', 'female', 'attempts', 'seriously', 'surprised', 'in', 'music', "he's", 'dreams', 'eyes', 'chief', 'best', 'impressive', 'leave', 'sucks', 'equally', 'hey', 'feeling', 'any', 'process', 'several', "i'd", 'stuck', 'like', 'color', 'great', 'joke', 'modern', 'unconvincing', 'complete', 'done', 'portrayal', 'appears', 'anti', 'figure', 'poor', 'hour', 'loving', 'depth', 'reason', 'turn', 'undercover', 'obvious', 'highly', 'tom', 'jolie', 'video', 'people', 'cameron', 'always', 'you', 'vulnerable', 'beauty', 'performance', 'year', 'white', 'mean', 'actually', 'artist', 'idiotic', 'absurd', 'cold', 'still', 'picture', 'when', 'ones', 'worst', 'surprising', 'family', "wouldn't", 'gets', 'aspect', 'school', 'amazing', 'view', 'direction', 'painful', 'script', 'remake', 'jokes', 'terms', 'winning', 'life', 'dreadful', 'ends', 'breathtaking', 'sense', 'images', 'insulting', 'viewers', 'reveal', 'own', 'different', 'winner', 'brain', 'everyone', 'knew', 'born', 'buy', 'martin', 'maybe', 'seen', 'spirit', 'imaginative', 'exciting', 'sees', 'through', 'result', 'worthy', 'puts', 'subplot', 'mess', 'presented', 'need', 'creates', 'steals', 'especially', 'sometimes', 'sounds',
                                 #'bill', 'lifeless', 'not', 'turns', 'religion', 'under', 'fails', 'took', 'superior', "don't", 'redeeming', 'witty'})


        return(features_selected)
         
    # grid search on different selection rates
    def grid_search(self, train_set, validation_set, select_rates):
        grid_results = pd.DataFrame(columns=['F1', 'accuracy'])
        for select_rate in select_rates:
            self.__init__()
            self.train(train_set, feature_select_rate=select_rate)
            results_valid = self.test(validation_set)
            (grid_results.loc[select_rate, 'F1'], 
            grid_results.loc[select_rate, 'accuracy']) = self.evaluate(results_valid)
            

        return grid_results


if __name__ == '__main__':
    select_rates = [x / 100 for x in range(1, 100, 5)]
    nb = NaiveBayes()
    gs_result = nb.grid_search('movie_reviews/train', 'movie_reviews/dev', select_rates=select_rates)
    print(gs_result)
    gs_result.plot()
    plt.show()
        
    
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train', feature_select_rate=0)
    results = nb.test('movie_reviews/dev')
    nb.evaluate(results)

