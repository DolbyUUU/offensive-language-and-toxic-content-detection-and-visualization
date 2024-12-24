import pandas as pd
import numpy as np
import fasttext
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle
from settings import settings

def preprocess_text(text, enable_preprocessing, enable_stopwords_removal=False):
    if enable_preprocessing:

        # Lowercase the text
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters and numbers
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)

        if enable_stopwords_removal:
            # Not recommended because stopwords contain "no", "not", etc.

            # Tokenization
            tokens = word_tokenize(text)

            # Remove stopwords
            tokens = [token for token in tokens if token not in stop_words]

            # Join the tokens back into a single string
            text = ' '.join(tokens)

    return text

def upsample(df, label_column):
    # Find the class with the max number of instances
    max_size = df[label_column].value_counts().max()

    # Create an empty DataFrame to store the upsampled data
    upsampled_df = pd.DataFrame()

    for label in df[label_column].unique():
        label_df = df[df[label_column]==label]
        if not label_df.empty:
            upsampled_label_df = label_df.sample(max_size, replace=True)
            upsampled_df = pd.concat([upsampled_df, upsampled_label_df])

    # Shuffle the DataFrame rows
    upsampled_df = shuffle(upsampled_df)

    return upsampled_df

def load_and_preprocess_data(dataset_option):
    # Check the provided dataset option and load the corresponding dataset(s)
    if dataset_option == 1:
        # Load the OLID training dataset
        train_data = pd.read_csv(path_train_olid, 
            sep='\t', names=['id', 'text', 'subtask_a', 'subtask_b', 'subtask_c'])

        # Prepare the OLID data
        train_data.loc[:, 'subtask_a'] = np.where(train_data['subtask_a'] == 'OFF', '__label__OFF', '__label__NOT')
        train_data.loc[:, 'subtask_b'] = np.where(train_data['subtask_b'] == 'TIN', '__label__TIN', '__label__UNT')
        train_data.loc[:, 'subtask_c'] = '__label__' + train_data['subtask_c'].str.upper()

        # Return the original OLID data
        train_data_a = train_data_b = train_data_c = train_data

    elif dataset_option == 2:
        # Load the SOLID training dataset
        train_data = pd.read_csv(path_train_solid, 
            sep='\t', names=['id', 'text', 'average', 'std'])

        # Filter the SOLID data
        train_data_a, train_data_b, train_data_c = process_solid_data(train_data)

    elif dataset_option == 3:
        # Load both the SOLID and OLID training datasets
        solid_train_data = pd.read_csv(path_train_solid, 
            sep='\t', names=['id', 'text', 'average', 'std'])
        olid_train_data = pd.read_csv(path_train_olid, 
            sep='\t', names=['id', 'text', 'subtask_a', 'subtask_b', 'subtask_c'])

        # Filter the SOLID data
        solid_train_data_a, solid_train_data_b, solid_train_data_c = process_solid_data(solid_train_data)

        # Prepare the OLID data
        olid_train_data.loc[:, 'subtask_a'] = np.where(olid_train_data['subtask_a'] == 'OFF', '__label__OFF', '__label__NOT')
        olid_train_data.loc[:, 'subtask_b'] = np.where(olid_train_data['subtask_b'] == 'TIN', '__label__TIN', '__label__UNT')
        olid_train_data.loc[:, 'subtask_c'] = '__label__' + olid_train_data['subtask_c'].str.upper()

        # Return the original OLID data
        olid_train_data_a = olid_train_data_b = olid_train_data_c = olid_train_data

        # Combine SOLID and OLID data
        train_data_a = pd.concat([solid_train_data_a, olid_train_data_a])
        train_data_b = pd.concat([solid_train_data_b, olid_train_data_b])
        train_data_c = pd.concat([solid_train_data_c, olid_train_data_c])

    else:
        raise ValueError("Invalid dataset_option. Use 1 for OLID, 2 for SOLID, and 3 for both.")

    # Shuffle the dataset
    train_data_a = shuffle(train_data_a)
    train_data_b = shuffle(train_data_b)
    train_data_c = shuffle(train_data_c)
    
    return train_data_a, train_data_b, train_data_c

def process_solid_data(train_data):
    # Load the subtask distant datasets
    task_a_distant = pd.read_csv(path_train_solid_a, sep='\t', 
        names=['id', 'average_a', 'std_a'])
    task_b_distant = pd.read_csv(path_train_solid_b, sep='\t', 
        names=['id', 'average_b', 'std_b'])
    task_c_distant = pd.read_csv(path_train_solid_c, sep='\t', 
        names=['id', 'average_c_ind', 'average_c_grp', 'average_c_oth', 'std_c_ind', 'std_c_grp', 'std_c_oth'])

    # Merge the training and subtask distant datasets on 'id'
    train_data = pd.merge(train_data, task_a_distant, on='id', how='left')
    train_data = pd.merge(train_data, task_b_distant, on='id', how='left')
    train_data = pd.merge(train_data, task_c_distant, on='id', how='left')

    # Explicitly convert 'average' and 'std' columns to float
    for column in ['average', 'std', 'average_a', 'std_a', 'average_b', 'std_b', 
                   'average_c_ind', 'average_c_grp', 'average_c_oth', 'std_c_ind', 
                   'std_c_grp', 'std_c_oth']:
        try:
            train_data[column] = pd.to_numeric(train_data[column], errors='coerce')
        except ValueError:
            print(f"Could not convert data in column {column}")

    # Filter the training dataset for FastText used in the SOLID paper
    train_data_a = train_data[(train_data['average_a'] < 0.20) | (train_data['average_a'] > 0.70)]
    train_data_a.loc[:, 'subtask_a'] = np.where(train_data_a['average_a'] < 0.20, '__label__NOT', '__label__OFF')

    train_data_b = train_data[(train_data['average_b'] < 0.35) | (train_data['average_b'] > 0.65)]
    train_data_b.loc[:, 'subtask_b'] = np.where(train_data_b['average_b'] > 0.65, '__label__UNT', '__label__TIN')

    # For subtask C, filter instances with label 'TIN' in subtask B
    # Choose the class with the highest average probability
    train_data_c = train_data_b[train_data_b['subtask_b'] == '__label__TIN']
    train_data_c = train_data_c[(train_data_c['average_c_ind'] > 0.80) | 
                                (train_data_c['average_c_grp'] > 0.70) | 
                                (train_data_c['average_c_oth'] > 0.65)]

    train_data_c['subtask_c'] = train_data_c[['average_c_ind', 'average_c_grp', 'average_c_oth']].idxmax(axis=1)
    train_data_c['subtask_c'] = '__label__' + train_data_c['subtask_c'].str.slice(start=10).str.upper()
    
    return train_data_a, train_data_b, train_data_c

def process_and_train_models(dataset_option, enable_preprocessing):
    # Load and preprocess data
    train_data_a, train_data_b, train_data_c = load_and_preprocess_data(dataset_option)
    train_data_a['text'] = train_data_a['text'].apply(
        lambda x: preprocess_text(x, enable_preprocessing, enable_stopwords_removal))
    train_data_b['text'] = train_data_b['text'].apply(
        lambda x: preprocess_text(x, enable_preprocessing, enable_stopwords_removal))
    train_data_c['text'] = train_data_c['text'].apply(
        lambda x: preprocess_text(x, enable_preprocessing, enable_stopwords_removal))

    # Upsample the training data for subtask B and C, not A
    train_data_b = upsample(train_data_b, 'subtask_b')
    train_data_c = upsample(train_data_c, 'subtask_c')

    # Save the upsampled training data to .txt files
    train_data_a[['text', 'subtask_a']].to_csv('train_data_fasttext_a.txt', index=False, sep=' ', header=False)
    train_data_b[['text', 'subtask_b']].to_csv('train_data_fasttext_b.txt', index=False, sep=' ', header=False)
    train_data_c[['text', 'subtask_c']].to_csv('train_data_fasttext_c.txt', index=False, sep=' ', header=False)

    # Check training data labels and instances for subtasks
    print(f"\nTraining data summary: dataset_option={dataset_option}, enable_preprocessing={enable_preprocessing}")
    print(f"\nLevel A: Offensive Language Detection")
    print(f"Column names: {train_data_a.columns}")
    print(f"Number of rows: {train_data_a.shape[0]}")
    print(f"\nLevel B: Categorization of Offensive Language")
    print(f"Column names: {train_data_b.columns}")
    print(f"Number of rows: {train_data_b.shape[0]}")
    print(f"\nLevel C: Offensive Language Target Identification")
    print(f"Column names: {train_data_c.columns}")
    print(f"Number of rows: {train_data_c.shape[0]}")

    # Train the FastText models for subtasks
    if enable_hyperparameters:
        model_a = fasttext.train_supervised('train_data_fasttext_a.txt', 
            lr=lr_a, wordNgrams=wordNgrams_a, ws=ws_a, loss=loss_a)
        model_b = fasttext.train_supervised('train_data_fasttext_b.txt', 
            lr=lr_b, wordNgrams=wordNgrams_b, ws=ws_b, loss=loss_b)
        model_c = fasttext.train_supervised('train_data_fasttext_c.txt', 
            lr=lr_c, wordNgrams=wordNgrams_c, ws=ws_c, loss=loss_c)
    else:
        model_a = fasttext.train_supervised('train_data_fasttext_a.txt')
        model_b = fasttext.train_supervised('train_data_fasttext_b.txt')
        model_c = fasttext.train_supervised('train_data_fasttext_c.txt')

    # Save the trained models
    model_a.save_model("model_fasttext_a.bin")
    model_b.save_model("model_fasttext_b.bin")
    model_c.save_model("model_fasttext_c.bin")

def main():
    # Process the data and train the model for each subtask
    process_and_train_models(dataset_option, enable_preprocessing)

if __name__ == "__main__":

    # Use the settings from settings.py
    dataset_option = settings['dataset_option']
    enable_preprocessing = settings['enable_preprocessing']
    enable_stopwords_removal = settings['enable_stopwords_removal']
    enable_hyperparameters = settings['enable_hyperparameters']
    lr_a = settings['lr_a']
    lr_b = settings['lr_b']
    lr_c = settings['lr_c']
    wordNgrams_a = settings['wordNgrams_a']
    wordNgrams_b = settings['wordNgrams_b']
    wordNgrams_c = settings['wordNgrams_c']
    ws_a = settings['ws_a']
    ws_b = settings['ws_b']
    ws_c = settings['ws_c']
    loss_a = settings['loss_a']
    loss_b = settings['loss_b']
    loss_c = settings['loss_c']
    path_train_olid = settings['path_train_olid']
    path_train_solid = settings['path_train_solid']
    path_train_solid_a = settings['path_train_solid_a']
    path_train_solid_b = settings['path_train_solid_b']
    path_train_solid_c = settings['path_train_solid_c']

    # Download the list of stopwords from NLTK for data preprocessing
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    main()