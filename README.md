### To Do

- Understand how many of the original tweets are in English and how many of them are in Creole
- Understand how many of them have been translated
    - If they have been translated, how many times have they been translated?

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


#Code to pull in the nltk imports
import nltk
nltk.download([
    'words,
    'punkt', 
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'wordnet'
    ])