import pandas as pd
import numpy as np
import fasttext
import re
import nltk
from tqdm import tqdm
from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from settings import settings
from train_fasttext import preprocess_text

def load_and_merge_test_data(path_tweets, path_labels):
    # Load and merge tweets and labels fof test data
    tweets = pd.read_csv(path_tweets, sep='\t', names=['id', 'tweet'])
    labels = pd.read_csv(path_labels, names=['id', 'label'])
    data = pd.merge(tweets, labels, on='id')
    return data

def make_predictions(model, test_data: dict) -> List:
    # Make predictions using a given model and test data
    predictions = []
    for tweet in tqdm(test_data['tweet']):
        prediction = model.predict(tweet)[0][0]
        predictions.append(prediction)
    return predictions

def main():
    # Load the trained models
    model_a = fasttext.load_model("model_fasttext_a.bin")
    model_b = fasttext.load_model("model_fasttext_b.bin")
    model_c = fasttext.load_model("model_fasttext_c.bin")

    # Load the test dataset for subtasks
    test_data_a = load_and_merge_test_data(path_test_solid_tweets_a, path_test_solid_labels_a)
    test_data_b = load_and_merge_test_data(path_test_solid_tweets_b, path_test_solid_labels_b)
    test_data_c = load_and_merge_test_data(path_test_solid_tweets_c, path_test_solid_labels_c)

    # Preprocess the labels in the test dataset
    test_data_a['label'] = test_data_a['label'].map({'NOT': '__label__NOT', 'OFF': '__label__OFF'})
    test_data_b['label'] = test_data_b['label'].map({'UNT': '__label__UNT', 'TIN': '__label__TIN'})
    test_data_c['label'] = test_data_c['label'].map({'IND': '__label__IND', 'GRP': '__label__GRP', 'OTH': '__label__OTH'})

    # Preprocess the text data in the test dataset 
    test_data_a['tweet'] = test_data_a['tweet'].apply(
        lambda x: preprocess_text(x, enable_preprocessing, enable_stopwords_removal))
    test_data_b['tweet'] = test_data_b['tweet'].apply(
        lambda x: preprocess_text(x, enable_preprocessing, enable_stopwords_removal))
    test_data_c['tweet'] = test_data_c['tweet'].apply(
        lambda x: preprocess_text(x, enable_preprocessing, enable_stopwords_removal))

    # Predict the labels for the test dataset and store them in a list for subtasks
    predictions_a = make_predictions(model_a, test_data_a)
    predictions_b = make_predictions(model_b, test_data_b)
    predictions_c = make_predictions(model_c, test_data_c)

    # Open the output file
    with open(f'results_fasttext_data{dataset_option}_preprocess{enable_preprocessing}.txt', 'w') as f:

        # Evaluate the model using a classification report for subtasks
        report_a = classification_report(test_data_a['label'], predictions_a, 
            target_names=['NOT', 'OFF'], zero_division=1)
        f.write("Report for Task A:\n")
        f.write(report_a)
        f.write("\n\n")

        report_b = classification_report(test_data_b['label'], predictions_b, 
            target_names=['UNT', 'TIN'], zero_division=1)
        f.write("Report for Task B:\n")
        f.write(report_b)
        f.write("\n\n")

        report_c = classification_report(test_data_c['label'], predictions_c, 
            target_names=['IND', 'GRP', 'OTH'], zero_division=1)
        f.write("Report for Task C:\n")
        f.write(report_c)
        f.write("\n\n")

if __name__ == "__main__":

    # Use the settings from settings.py
    dataset_option = settings['dataset_option']
    enable_preprocessing = settings['enable_preprocessing']
    enable_stopwords_removal = settings['enable_stopwords_removal']
    path_test_solid_tweets_a = settings['path_test_solid_tweets_a']
    path_test_solid_tweets_b = settings['path_test_solid_tweets_b']
    path_test_solid_tweets_c = settings['path_test_solid_tweets_c']
    path_test_solid_labels_a = settings['path_test_solid_labels_a']
    path_test_solid_labels_b = settings['path_test_solid_labels_b']
    path_test_solid_labels_c = settings['path_test_solid_labels_c']

    # Download the list of stopwords from NLTK for data preprocessing
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    main()