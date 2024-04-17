![alt text](image.png)

# Signal Storm: Leveraging Machine Learning to Identify Requests for Help During Natural Disasters

# Project Overview
This code creates a machine learning pipeline that can be used to classify tweets sent during an emergency so that help can be sent from the appropriate agency. The project also includes a website where individuals can input new messages and get classification results in several categories. 

# Installation and Setup

## Codes and Resources Used
- **Editor:** VSCode
- **Python Version:** 3.12.0

## Python Packages Used
- **General Purpose:** numpy, pandas
- **Data Manipulation:** SQLAlchemy
- **Data Visualization:** matplotlib, plotly
- **Natural Language Processing:** nltk
- **NLTK Resources:** works, punkt, averaged_perception_tagger, maxent_ne_chunker, wordnet
- **Machine Learning:** scikit-learn, joblib
- **Web App:** Flask, Bootstrap

## Instructions
*_Note_*: If you're using a virtual environment, please make sure its activated before you run these commands. 
1. To set up the database and machine learning model, run the following commands:
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data\01_raw\disaster_messages.csv data\01_raw\disaster_categories.csv data\02_stg\stg_disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models\train_classifier.py data/DisasterResponse.db models\classifier.pkl`
        - **WARNING**: If you're running the pipeline locally, this might take a few minutes. The script will run use n-1 cores.
2. To run the Flask app:
  - Go to `app` directory: `cd app`
  - Run your web app: `python run.py`
  - Copy http://127.0.0.1:3000 or the equivalent into your browser to view the app
    - *_Note:_* This is the local host, and is restricted to your local machine. The second address is the network address of your server which can be access from any machine on your local network.

# Data
The model was built on a combination of the following two data sets:
- **disaster_messages.csv**
  - Contains messages set during the disaster. Each message is labeled with one or more disaster-related categories, such as "water", "food", "medical help", etc.
  - Messages can be in a variety of languages.'original' messages seem to be messages in Haitian Creole that were translated into English. The corresponding note or English translation is in the 'message' column.
  - Messages are classified into the genres There are three values: direct, news and social
- **disaster_categories.csv**
  - Contains the corresponding categories for each message in the disaster_messages dataset. Each category is represented by a binary value (0 or 1), indicating whether the message belongs to that category or not.
  - the 'related' column indicates if the message is _related_ to the disaster or not. There are three possible values: 1 (related), 0 (not related) and 2 (ambiguous)

# Model Design
The model is designed as a machine learning pipeline that processes text and classifies it into one of the **36 categories** in the dataset. The pipeline consists of three main steps:

1. **Text Processing**: The text data is first processed using a custom `tokenize` function from the nltk library. This function normalizes the case, lemmatizes, and tokenizes the text. It also handles URL detection and replacement, punctuation removal, and stop word removal.

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

However, when we look at the median percent change between the two models per output class:
| output_class   | precision | recall | f1-score |
|----------------|-----------|--------|----------|
| 0              | -0.02     | 0.01   | 0.00     |
| 1              | 0.66      | -1.52  | -0.57    |
| 2              | 52.38     | 0.00   | 20.75    |
| macro avg      | 0.00      | -0.43  | -0.44    |
| weighted avg   | 0.00      | -0.00  | -0.02    |

We see that the data shows the new ML model significantly boosts precision in class 2, but at the cost of reduced recall in class 1. This trade-off decreases the F1-score for class 1, indicating a shift towards specialization in the model's performance across classes. This is not a trade that we want to make because we want to make sure that we're capturing as many true positive requests for help. In addition, changing `vect__ngram_range` from (1, 1) to (1, 2) and `clf__estimator__n_estimators` from 100 to 200 increased training time from approximately two minutes to eight minutes (a 300% increase in computational time).

# Conclusion and Recommendations

## Conclusion
The machine learning pipeline developed in this project demonstrates a promising approach to classifying disaster-related messages into 36 categories. However, the model's performance varies across different classes, with some classes achieving high precision at the cost of reduced recall. This trade-off is not ideal for our use case, as we aim to capture as many true positive requests for help as possible.

The model's performance was optimized using GridSearchCV, which significantly increased the computational time. While this resulted in improved precision for some classes, the overall F1-score, which balances precision and recall, decreased for others. This suggests that the model's performance could be further improved.

## Recommendations
1. **Optimize Grid Search for Weighted F1-Score instead of Accuracy**: Accuracy is not always the best metric for evaluating a model's performance, especially for imbalanced datasets. Optimizing for the weighted F1-score, which considers both precision and recall, could lead to a more balanced model.

2. **Use a Translation API for Consistent Tweet Translations**: The dataset contains messages in various languages, and the quality of translations can significantly impact the model's performance. Using a reliable translation API could ensure consistent and accurate translations.

3. **Consider Class Imbalance**: Some classes in the dataset have significantly fewer samples than others, which can bias the model towards the majority classes. Techniques such as oversampling the minority classes or undersampling the majority classes could help address this issue.

4. **Experiment with Different Models**: While the RandomForestClassifier performed reasonably well, other models might yield better results. Experimenting with different types of models, such as Support Vector Machines (SVM) or neural networks, could potentially improve the model's performance.

5. **Feature Engineering**: Additional features could be engineered from the text data to potentially improve the model's performance. For example, the length of the message, the number of words, or the presence of certain keywords could be useful features.

# License
[MIT License](https://opensource.org/license/mit/)
