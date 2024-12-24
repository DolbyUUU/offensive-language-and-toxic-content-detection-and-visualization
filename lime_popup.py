import os
import fasttext
import numpy as np
import pandas as pd
import random
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from settings import settings
from train_fasttext import preprocess_text
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import re

# The classes returned by FastText are in descending order of the probabilities
# We might get the negative class (class 0) first if its probability is higher
# Ensure the order of labels is consistent
def predict_fn_a(model, texts):
    preprocessed_texts = [preprocess_text(text, settings['enable_preprocessing'], 
        settings['enable_stopwords_removal']) for text in texts]
    labels, probabilities = model.predict(preprocessed_texts, k=2)
    sorted_probabilities = []
    for label, prob in zip(labels, probabilities):
        if label[0] == '__label__NOT':
            sorted_probabilities.append([prob[1], prob[0]]) # Swap the order
        else:
            sorted_probabilities.append([prob[0], prob[1]])
    return np.array(sorted_probabilities)

def predict_fn_b(model, texts):
    preprocessed_texts = [preprocess_text(text, settings['enable_preprocessing'], 
        settings['enable_stopwords_removal']) for text in texts]
    labels, probabilities = model.predict(preprocessed_texts, k=2)
    sorted_probabilities = []
    for label, prob in zip(labels, probabilities):
        if label[0] == '__label__UNT':
            sorted_probabilities.append([prob[1], prob[0]]) # Swap the order
        else:
            sorted_probabilities.append([prob[0], prob[1]])
    return np.array(sorted_probabilities)

def predict_fn_c(model, texts):
    preprocessed_texts = [preprocess_text(text, settings['enable_preprocessing'], 
        settings['enable_stopwords_removal']) for text in texts]
    labels, probabilities = model.predict(preprocessed_texts, k=3)
    sorted_probabilities = []
    for label, prob in zip(labels, probabilities):
        label_prob_dict = {label[i]: prob[i] for i in range(3)}
        sorted_probabilities.append([label_prob_dict.get('__label__IND', 0), 
                                     label_prob_dict.get('__label__GRP', 0),
                                     label_prob_dict.get('__label__OTH', 0)])
    return np.array(sorted_probabilities)

class CustomDialog(simpledialog.Dialog):
    def body(self, master):
        self.geometry("350x125")  # Set the width and height of the window
        self.var = tk.StringVar(value="")
        tk.Label(master, text='Choose an option:', justify='center').pack()  # Create a label
        self.rb1 = ttk.Radiobutton(master, text='Manually Enter Text in English', variable=self.var, value='1')
        self.rb2 = ttk.Radiobutton(master, text='Randomly Select a Tweet from SOLID Database', variable=self.var, value='2')
        self.rb1.pack(anchor='w')
        self.rb2.pack(anchor='w')
        return self.rb1  # initial focus

    def apply(self):
        self.result = self.var.get()

class TextInputDialog(simpledialog.Dialog):
    def body(self, master):
        self.text = tk.Text(master, width=50, height=20)  # Set the dimensions as per requirements
        self.text.pack()
        return self.text

    def apply(self):
        self.result = self.text.get("1.0", 'end-1c')  # get text from Text widget

class RandomTextDialog(simpledialog.Dialog):
    def __init__(self, master, df):
        self.df = df
        self.text_var = tk.StringVar()
        self.select_random_text()
        super().__init__(master)

    def body(self, master):
        self.geometry("350x175")
        tk.Label(master, text='Randomly Selected Text:\n').pack()
        self.label = tk.Label(master, textvariable=self.text_var, wraplength=300)
        self.label.pack()
        tk.Button(master, text="Dislike", command=self.select_random_text, width=10).pack()
        return None  # initial focus

    def select_random_text(self):
        random_index = random.randint(0, len(self.df) - 1)
        text = self.df['tweet'].iloc[random_index]
        safe_text = re.sub(r'[^\u0000-\uFFFF]', '', text)
        self.text_var.set(safe_text)

    def apply(self):
        self.result = self.text_var.get()

def get_text():
    ROOT = tk.Tk()
    ROOT.withdraw()
    ROOT.title("Your Text Here")
    dialog = TextInputDialog(ROOT)
    return dialog.result

def main():
    # Create dictionaries to map labels to readable text
    model_a_labels = {"__label__OFF": "Offensive", "__label__NOT": "Not Offensive"}
    model_b_labels = {"__label__TIN": "Targeted", "__label__UNT": "Untargeted"}
    model_c_labels = {"__label__IND": "Individual", "__label__GRP": "Group", "__label__OTH": "Others"}
    
    # Load the SOLID test dataset
    test_solid_data = pd.read_csv(path_test_solid_tweets_a, sep='\t', names=['id', 'tweet'])

    ROOT = tk.Tk()
    ROOT.withdraw()
    ROOT.title("Offensive Language Detection")
    dialog = CustomDialog(ROOT)
    USER_INP = dialog.result

    if USER_INP == '1':
        sample_text = get_text()
    elif USER_INP == '2':
        # Get a random sample text
        dialog2 = RandomTextDialog(ROOT, test_solid_data)
        sample_text = dialog2.result
    else:
        messagebox.showerror("Error", "Invalid selection. Please choose an option.")
        return

    # Preprocess the sample text
    sample_text_preprocessed = preprocess_text(
        sample_text, settings['enable_preprocessing'], settings['enable_stopwords_removal'])
    # Remove newline characters to avoid errors
    sample_text_preprocessed = sample_text_preprocessed.replace('\n', ' ')

    # Model A prediction
    prediction_a = model_a.predict(sample_text_preprocessed)[0][0]
    print(f'\nText: {sample_text_preprocessed}\n')
    print(f'Model A prediction: {model_a_labels[prediction_a]}\n')

    # If Model A predicts "Offensive", proceed with Model B
    if prediction_a == '__label__OFF':
        # Model B prediction
        prediction_b = model_b.predict(sample_text_preprocessed)[0][0]
        print(f'Model B prediction: {model_b_labels[prediction_b]}\n')

        # If Model B predicts "Targeted", proceed with Model C
        if prediction_b == '__label__TIN':
            # Model C prediction
            prediction_c = model_c.predict(sample_text_preprocessed)[0][0]
            print(f'Model C prediction: {model_c_labels[prediction_c]}\n')
        else:
            print('Model B predicted "Untargeted", so no further predictions were made.\n')
    else:
        print('Model A predicted "Not Offensive", so no further predictions were made.\n')

    # Create LimeTextExplainer
    explainer_a = LimeTextExplainer(class_names=["Offensive", "Not Offensive"])
    explainer_b = LimeTextExplainer(class_names=["Targeted", "Untargeted"])
    explainer_c = LimeTextExplainer(class_names=["Individual", "Group", "Others"])

    # Explain the prediction of model_a
    exp_a = explainer_a.explain_instance(sample_text_preprocessed, 
        lambda x: predict_fn_a(model_a, x), num_features=num_features)
    print("\nExplanation for Model A saved in:")
    exp_a.show_in_notebook(text=True)

    # Save the explanation to an HTML file
    with open('explanation_a.html', 'w', encoding='utf-8') as f:
        f.write(exp_a.as_html())

    # Check if Model A's prediction is "Offensive"
    if model_a.predict(sample_text_preprocessed)[0][0] == '__label__OFF':
        # Explain the prediction of model_b
        exp_b = explainer_b.explain_instance(sample_text_preprocessed, 
            lambda x: predict_fn_b(model_b, x), num_features=num_features)
        print("\nExplanation for Model B saved in:")
        exp_b.show_in_notebook(text=True)

        # Save the explanation to an HTML file
        with open('explanation_b.html', 'w', encoding='utf-8') as f:
            f.write(exp_b.as_html())

        # Check if Model B's prediction is "Targeted"
        if model_b.predict(sample_text_preprocessed)[0][0] == '__label__TIN':
            # Explain the prediction of model_c
            exp_c = explainer_c.explain_instance(sample_text_preprocessed, 
                lambda x: predict_fn_c(model_c, x), top_labels=3, num_features=num_features)
            print("\nExplanation for Model C saved in:")
            exp_c.show_in_notebook(text=True)

            # Save the explanation to an HTML file
            with open('explanation_c.html', 'w', encoding='utf-8') as f:
                f.write(exp_c.as_html())

if __name__ == "__main__":
    # Use the settings from settings.py
    path_test_solid_tweets_a = settings['path_test_solid_tweets_a']
    num_features = settings['num_features']

    # Remove the files if exist
    for file in ['explanation_a.html', 'explanation_b.html', 'explanation_c.html']:
        if os.path.exists(file):
            os.remove(file)

    model_a = fasttext.load_model("model_fasttext_a.bin")
    model_b = fasttext.load_model("model_fasttext_b.bin")
    model_c = fasttext.load_model("model_fasttext_c.bin")

    main()