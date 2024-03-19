'''
Identification of Clinically-Relevant Text Content using Natural Language Processing.

This script loads a dataset containing text data, cleans and preprocesses this
text data, carries out a basic exploratory data analysis (EDA) and utilises a
Multinomial Naive Bayes model for classification, and evaluates the model's
performance.
'''

# Import dependencies.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns

# Load data.
def load_data(data_path: str) -> pd.DataFrame:
    '''
    Loads the dataset from a CSV file.

    Input:
        - data_path (str): Path to the CSV file.

    Output:
        - df (pd.DataFrame): Loaded DataFrame containing the dataset.
    '''
    df = pd.read_csv(data_path)
    return df

# Check dataset contents.
def check_data(df: pd.DataFrame) -> None:
    '''
    Checks and provides a summary of the dataset contents.

    Input:
        - df (pd.DataFrame): DataFrame containing the dataset.

    Output:
        None
    '''
    print(df.head(), '\n')
    
    num_null_pre = df.isnull().sum() # Check missing data count.
    print('Missing data count: \n', num_null_pre, '\n')
    if num_null_pre.any() > 0:
        df = df.dropna()
        num_null_post = df.isnull().sum()
        print('Missing data removed: \n', num_null_post, '\n')
    
    for column in df.columns: # Check data types count.
        data_types_count = df[column].apply(type).value_counts()
        print(f'Counts of data types in {column} column: ', data_types_count, '\n')

    label_counts = df['label'].value_counts() # Check count per data label.
    print('Label data counts: ', label_counts, '\n')

    bar_x = ['Healthy', 'Illness-indicative'] # Visualise count per data label.
    bar_y = [label_counts[0], label_counts[1]]
    bar_colours = ['lightblue', 'lightcoral']
    plt.bar(bar_x, bar_y, color = bar_colours, zorder = 2, width = 0.5)
    plt.xlabel('Label', fontweight = 'bold')
    plt.ylabel('Count', fontweight = 'bold')
    plt.title('Count of Text Data Labelled as Either Healthy or Indicative of Mental Illness', fontweight = 'bold')
    plt.yticks(range(0, max(label_counts) + 1, 10000))
    plt.grid(axis = 'y', linestyle = '--', zorder = 1)
    plt.show()

# Preprocess data.
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocesses the text data.

    Input:
        - df (pd.DataFrame): DataFrame containing the text data.

    Output:
        - df (pd.DataFrame): Preprocessed DataFrame.
    '''
    clean_text = []
    
    lemmatiser = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    for text in df['text']:
        text = str(text)
        text = text.lower() # Convert to lowercase.
        text = re.sub(r'http\S+', '', text) # Remove URLs.
        text = text.encode('ascii', 'ignore').decode('ascii') # Remove emojis.
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # Remove non-alphanumeric characters.
        
        # Reduce individual words to root form and remove stop words.
        words = word_tokenize(text)
        words = [lemmatiser.lemmatize(word) for word in words if len(word) > 2 and word not in stop_words]

        clean_text.append(' '.join(words))
        
    df['text'] = clean_text
    
    return df

# Generate word cloud of text data.
def generate_wordcloud(text_data: pd.Series, title: str, colourscheme: str) -> None:
    '''
    Generates a word cloud of the text data.

    Input:
        - text_data (pd.Series): Series containing the text data.
        - title (str): Title of the word cloud.
        - colourscheme (str): Colormap for the word cloud.

    Output:
        None
    '''
    text_string = ' '.join(text_data)
    wordcloud = WordCloud(width = 1000,
                  height = 500,
                  background_color = 'white',
                  collocations = False,
                  colormap = colourscheme)
    wordcloud.generate(text_string)
    plt.figure(figsize = (10, 5))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title, fontweight = 'bold', pad = 7)
    plt.show()
    
# Visualise frequencies of most common words in data.
def plot_word_frequencies(text_data: pd.Series, title: str, colourscheme: str) -> None:
    '''
    Visualises the frequencies of the most common words in the text data.

    Parameters:
        - text_data (pd.Series): Series containing the text data.
        - title (str): Title of the plot.
        - colourscheme (str): Colormap for the plot.

    Returns:
        None
    '''
    text_string = ' '.join(text_data)
    all_words = text_string.split()
    all_words_count = Counter(all_words)
    top_words = all_words_count.most_common(20)
    
    words = [word[0] for word in top_words]
    counts = [count[1] for count in top_words]
    
    plt.figure(figsize = (12, 8))
    plt.barh(words, counts, color = colourscheme)
    plt.xlabel('Frequency', fontsize = 14, fontweight = 'bold')
    plt.ylabel('Words', fontsize = 14, fontweight = 'bold')
    plt.title(title, fontsize = 16, fontweight = 'bold')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()    

# Exploratory Data Analysis (EDA).
def explore_data(df: pd.DataFrame) -> None:
    '''
    Performs exploratory data analysis by generating word clouds and visualising word frequencies for all text data, healthy text data, and illness-indicative text data.

    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        None
    '''
    # Generate word clouds.
    all_text_data = df['text']
    generate_wordcloud(all_text_data, 'Word Cloud of All Text Data', 'viridis')
    
    healthy_data = df[df['label'] == 0]
    healthy_text_data = healthy_data['text']
    generate_wordcloud(healthy_text_data, 'Word Cloud of Healthy Text Data', 'winter')
    
    illness_data = df[df['label'] == 1]
    illness_text_data = illness_data['text']
    generate_wordcloud(illness_text_data, 'Word Cloud of Text Data Indicative of Mental Illness', 'autumn_r')

    # Visualise top word frequencies.
    plot_word_frequencies(all_text_data, 'Most Frequent Words in All Text Data', 'limegreen')
    plot_word_frequencies(healthy_text_data, 'Most Frequent Words in Healthy Text Data', 'skyblue')
    plot_word_frequencies(illness_text_data, 'Most Frequent Words in Illness-Indicative Text Data', 'lightcoral')

# Build, apply and evaluate Multinomial Naive Bayes model.
def run_model(df: pd.DataFrame) -> None:
    '''
    Implements the Multinomial Naive Bayes model for text classification.

    Parameters:
        - df (pd.DataFrame): DataFrame containing the dataset with 'text' and 'label' columns.

    Returns:
        None
    '''
    X_train, X_test, y_train, y_test = train_test_split(df['text'],
                                                        df['label'],
                                                        test_size = 0.2
                                                        )
    vectoriser = CountVectorizer()
    X_train_vectorised = vectoriser.fit_transform(X_train)
    X_test_vectorised = vectoriser.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_vectorised, y_train)
    
    y_pred = model.predict(X_test_vectorised)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}\n')
    print('Confusion Matrix:\n', conf_matrix)
    print('\nClassification Report:\n', class_report)
    
    sns.heatmap(conf_matrix,
                annot = True,
                fmt = 'd',
                cmap = 'Greens',
                cbar = False,
                xticklabels = ['Healthy', 'Illness'],
                yticklabels = ['Healthy', 'Illness'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Heatmap')
    plt.show()
    

# Call functions to perform investigation.
data_path = 'C:/Users/Oscar/Documents/Projects/mental_health_nlp/text_data3.csv'
df = load_data(data_path)

check_data(df)

df = preprocess_data(df)

explore_data(df)

run_model(df)

