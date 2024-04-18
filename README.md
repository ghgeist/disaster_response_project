![Alt text](images/image.png)

# Signal Storm: Leveraging Machine Learning to Identify Requests for Help During Natural Disasters

# Project Overview
This code creates a machine learning pipeline that can be used to classify tweets sent during an emergency so that help can be sent from an appropriate agency. The project also includes a website where individuals can input new messages and get classification results in several categories. 

# Installation and Setup

## Codes and Resources Used
- **Editor:** VSCode
- **Python Version:** 3.12.0

## Python Packages Used
- **General Purpose:** numpy, pandas
- **Data Manipulation:** SQLAlchemy
- **Data Visualization:** matplotlib, plotly
- **Natural Language Processing:** nltk
- **NLTK Resources:** punkt, averaged_perception_tagger, maxent_ne_chunker, wordnet
- **Machine Learning:** scikit-learn, joblib
- **Web App:** Flask, Bootstrap

## Instructions
*_Note_*: If you're using a virtual environment, please make sure its activated before you run these commands. 
1. To set up the database and machine learning model, run the following commands:
    - To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data\01_raw\disaster_messages.csv data\01_raw\disaster_categories.csv data\02_stg\stg_disaster_response.db`
    - To run ML pipeline that trains classifier and saves the resulting model:
        `python models\train_classifier.py data\02_stg\stg_disaster_response.db models\classifier.pkl`
        - **WARNING**: If you're running the pipeline locally, this might take a few minutes. The script will run use n-1 cores.
2. To run the Flask app:
  - Go to `app` directory: `cd app`
  - Run the web app: `python run.py`
  - Copy http://127.0.0.1:3000 or the equivalent into your browser to view the app
    - *_Note:_* This is the local host, and is restricted to your local machine. The second address is the network address of your server which can be access from any machine on your local network.

# Data
The model was built on a combination of the following two data sets:
- **disaster_messages.csv**
  - Contains messages set during the disaster. Each message is labeled with one or more disaster-related categories, such as "water", "food", "medical help", etc.
  - Messages can be in a variety of languages.'original' messages are predominately in Haitian Creole that were translated into English. The corresponding note or English translation is in the 'message' column.
  - Messages are classified into the genres There are three values: direct, news and social
- **disaster_categories.csv**
  - Contains the corresponding categories for each message in the disaster_messages dataset. Each category is represented by a binary value (0 or 1), indicating whether the message belongs to that category or not.
  - the 'related' column indicates if the message is _related_ to the disaster or not. In the raw data, there are three possible values: 1 (related), 0 (not related) and 2 (ambiguous). The ambiguous messages have been dropped from the training set.

# Model Design
The model is designed as a machine learning pipeline that processes text and classifies it into one of the **36 categories** in the dataset. The pipeline consists of three main steps:

1. **Text Processing**: The text data is first processed using a custom `tokenize` function from the nltk library. This function normalizes the case, lemmatizes and tokenizes the text. It also handles URL detection and replacement, punctuation removal and stop word removal.

2. **Vectorization and TF-IDF Transformation**: The processed text is then vectorized using `CountVectorizer` with the custom tokenizer. After vectorization, a `TF-IDF` transformation is applied to the vectorized data.

3. **Multi-output Classification**: The transformed data is classified using a `RandomForestClassifier`.

The trained model is saved to a pickle file for future use.

# Tuning the Model for Accuracy
I used GridSearchCV to tune the model for accuracy, and tested the following parameters:
| Parameter                              | Values                              |
|----------------------------------------|-------------------------------------|
| `vect__ngram_range`                    | ((1, 1), (1, 2))                    |
| `clf__estimator__n_estimators`         | [50, 100, 200]                      |
| `clf__estimator__min_samples_split`    | [2, 3, 4]                           |

This process resulted in the following 'optimized' values:
| Parameter                                   | Original Value | Optimized Value |
|---------------------------------------------|----------------|-----------------|
| `vect__ngram_range`                         | (1, 1)         | (1, 2)          |
| `clf__estimator__n_estimators`              | 100            | 200             |
| `clf__estimator__min_samples_split`         | 2              | 2               |

Here median percent changes between the two models per output class:
| Output Class   | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| 0              | -0.02%    | 0.01%  | 0.00%    |
| 1              | 4%        | -8.10% | -5.60%   |
| Macro Avg      | 0.00%     | -0.41% | -0.75%   |
| Weighted Avg   | -0.06%    | 0.00%  | -0.04%   |

The data shows that the optimized model increase precision by .04% for relevant tweets (1), but decrease recall and the f1-score (-8.% and -5.6% respectively). This means that the optimized model is detecting positive cases more accurately, but at the expense of being able to detect all positive cases. This is not a trade that we want to make because we want to make sure that we're capturing as many true positive requests for help. In addition, changing `vect__ngram_range` from (1, 1) to (1, 2) and `clf__estimator__n_estimators` from 100 to 200 increased training time from approximately one minute to seven minutes (a 700% increase in computational time).

# Conclusion and Recommendations

## Conclusion
The machine learning pipeline developed in this project demonstrates a promising approach to classifying disaster-related messages into 36 categories. However, the model's performance varies across different classes, with some classes achieving high precision at the cost of reduced recall. This trade-off is not ideal for our use case, as we aim to capture as many true positive requests for help as possible.

The model's performance was optimized using GridSearchCV, which significantly increased the computational time. While this resulted in improved precision for some classes, the overall F1-score, which balances precision and recall, and decreases for others. This suggests that the model's performance could be further improved.

## Recommendations
1. **Optimize Grid Search for Weighted F1-Score instead of Accuracy**: Accuracy is not always the best metric for evaluating a model's performance, especially for imbalanced datasets. Optimizing for the weighted F1-score, which considers both precision and recall, could lead to a more balanced model.

2. **Use a Translation API for Consistent Tweet Translations**: The dataset contains messages in various languages, and the quality of translations can significantly impact the model's performance. Using a reliable translation API could ensure consistent and accurate translations.

3. **Consider Class Imbalance**: Some classes in the dataset have significantly fewer samples than others, which can bias the model towards the majority classes. Techniques such as oversampling the minority classes or undersampling the majority classes could help address this issue.

4. **Feature Engineering**: Additional features could be engineered from the text data to potentially improve the model's performance. For example, the length of the message, the number of words, or the presence of certain keywords could be useful features.

# License
[MIT License](https://opensource.org/license/mit/)
