<p align="center">
  <img width="400" height="400" src="">
</p>

# Signal Storm: Leveraging Machine Learning to Identify Requests for Help During Natural Disasters

# Project Overview

# Installation and Setup

## Codes and Resources Used
- **Editor:** VSCode
- **Python Version:** 3.12.0

## Python Packages Used
- **General Purpose:** 
- **Data Manipulation:**
- **Data Visualization:** 
- **NLTK Resources:** works, punkt, averaged_perception_tagger, maxent_ne_chunker, wordnet

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data\01_raw\disaster_messages.csv data\01_raw\disaster_categories.csv data\02_stg\stg_disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models\train_classifier.py data/DisasterResponse.db models\classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


# Data

## Source Data
### Technical notes


### Context


### Summary
- The script uses a custom tokenize function using nltk to case normalize, lemmatize and tokenize text. This function is used in the machine learning pipeline to vectorize and then apply TF-IDF to the text
- The script builds a pipeline that processes text and then performs multi-output classification on the **36 categories** in the dataset. GridSearchCV is used to find the best parameters for the model.

## Data Preprocessing

# Results
## Tuning the Model
Originally, the model was optimized for accuracy:

Original parameters for accuracy. 
{"clf__estimator__min_samples_split": 2, "clf__estimator__n_estimators": 100, "vect__ngram_range": [1, 1]}
Best parameters for accuracy
{"clf__estimator__min_samples_split": 2, "clf__estimator__n_estimators": 200, "vect__ngram_range": [1, 2]}

However, this results in imbalanced classes. The weighted F1 metric is probably better because it accounts for the balance between precision and recall, along with support for the given class.
## Trend Analysis

# Conclusion and Recommendations

# Future Directions
- Optimize Grid Search for Weighted F1-Score instead of Accuracy
- Use a translation API to ensure tweet translation consistency

# Footnotes
[^1]:

# License
[MIT License](https://opensource.org/license/mit/)
