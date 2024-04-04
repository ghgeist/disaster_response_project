### To Do
- Fix the layout of this readme.

### Housekeeping

- Add doc strings

### Ideas

- I think it would be cool to do a visualization so we can see how the tweets are clustering
- I think seeing the clusters over time would be especially useful


### Overview
The data are really messages that were sent during disaster events

### To Do
- Create a machine learning pipeline to categorize these events so that you can send the messages to the appropriate disaster relief agency
- Create a web app where an emergency work can input a new message and get classification results in several categories.
- Create some visualizations of the data


### ML Overview
- The script uses a custom tokenize function using nltk to case normalize, lemmatize and tokenize text. This function is used in the machine learning pipeline to vectorize and then apply TF-IDF to the text
- The script builds a pipeline that processes text and then performs multi-output classification on the **36 categories** in the dataset. GridSearchCV is used to find the best parameters for the model.
    - I guess related is a category even though it's not a boolean?


### Note
- it looks like we'll need to classify the message as direct, social or news?
- 

Here are all of the nltk downloads that you'll need in order to run this code
These downloads only have to be done once per virtual environment


### Future Directions
- Optimize Grid Search for Weighted F1-Score instead of Accuracy
- Use a translation API to ensure tweet translation consistency

# The Problem

The problem here is that I'm pretty sure that my classes are imbalanced. If this is the case, accuracy is probably not a good metric to evaluate the model's performance. The weighted F1 metric is probably better because it accounts for the balance between precision and recall, along with support for the given class. 

#Code to pull in the nltk imports
import nltk
nltk.download([
    'words,
    'punkt', 
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'wordnet'
    ])

Original paramters
{"clf__estimator__min_samples_split": 2, "clf__estimator__n_estimators": 100, "vect__ngram_range": [1, 1]}
Best parameters
{"clf__estimator__min_samples_split": 2, "clf__estimator__n_estimators": 200, "vect__ngram_range": [1, 2]}