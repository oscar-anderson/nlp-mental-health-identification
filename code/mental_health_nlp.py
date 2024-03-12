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
df = pd.read_csv('C:/Users/Oscar/Documents/Projects/mental_health_nlp/text_data3.csv')

# Preprocessing.
num_null = df.isnull().sum() # Check missing data count.
print('Missing data count: \n', num_null, '\n')

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

# EDA.
# Word clouds.
all_text_data = df['text']
all_text_string = ' '.join(all_text_data)
wordcloud = WordCloud(width = 1000,
                      height = 500,
                      background_color = 'white',
                      collocations = False,
                      colormap = 'viridis')
wordcloud.generate(all_text_string)
plt.figure(figsize = (10, 5)) # Visualise word cloud of all text data.
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud of All Text Data',
          fontweight = 'bold',
          pad = 7)
plt.show()

healthy_data = df[df['label'] == 0]
healthy_text_data = healthy_data['text']
healthy_text_string = ' '.join(healthy_text_data)
wordcloud = WordCloud(width = 1000,
                      height = 500,
                      background_color = 'white',
                      collocations = False,
                      colormap = 'winter')
wordcloud.generate(healthy_text_string)
plt.figure(figsize = (10, 5)) # Visualise word cloud of healthy text data.
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud of Healthy Text Data',
          fontweight = 'bold',
          pad = 7)
plt.show()

illness_data = df[df['label'] == 1]
illness_text_data = illness_data['text']
illness_text_string = ' '.join(illness_text_data)
wordcloud = WordCloud(width = 1000,
                      height = 500,
                      background_color = 'white',
                      collocations = False,
                      colormap = 'autumn_r')
wordcloud.generate(illness_text_string)
plt.figure(figsize = (10, 5)) # Visualise word cloud of ill text data.
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud of Text Data Indicative of Mental Illness',
          fontweight = 'bold',
          pad = 7)
plt.show()

# Word frequencies.
all_words = all_text_string.split()
all_words_count = Counter(all_words)
top_words = all_words_count.most_common(20)

words = [word[0] for word in top_words]
counts = [count[1] for count in top_words]

plt.figure(figsize = (12, 8))
plt.barh(words, counts, color = 'limegreen')
plt.xlabel('Frequency', fontsize = 14, fontweight = 'bold')
plt.ylabel('Words', fontsize = 14, fontweight = 'bold')
plt.title('Most Frequent Words in All Text Data', fontsize = 16, fontweight = 'bold')
plt.gca().invert_yaxis()
plt.grid()
plt.show()


healthy_words = healthy_text_string.split()
healthy_words_count = Counter(healthy_words)
top_healthy_words = healthy_words_count.most_common(20)

words = [word[0] for word in top_healthy_words]
counts = [count[1] for count in top_healthy_words]

plt.figure(figsize = (12, 8))
plt.barh(words, counts, color = 'skyblue')
plt.xlabel('Frequency', fontsize = 14, fontweight = 'bold')
plt.ylabel('Words', fontsize = 14, fontweight = 'bold')
plt.title('Most Frequent Words in Healthy Text Data', fontsize = 16, fontweight = 'bold')
plt.gca().invert_yaxis()
plt.grid()
plt.show()


illness_words = illness_text_string.split()
illness_words_count = Counter(illness_words)
top_illness_words = illness_words_count.most_common(20)

words = [word[0] for word in top_illness_words]
counts = [count[1] for count in top_illness_words]

plt.figure(figsize = (12, 8))
plt.barh(words, counts, color = 'lightcoral')
plt.xlabel('Frequency', fontsize = 14, fontweight = 'bold')
plt.ylabel('Words', fontsize = 14, fontweight = 'bold')
plt.title('Most Frequent Words in Illness-Indicative Text Data', fontsize = 16, fontweight = 'bold')
plt.gca().invert_yaxis()
plt.grid()
plt.show()
    
# Modelling.
# Multinomial Naive Bayes.
# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text data into a matrix of token counts.
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train the Multinomial Naive Bayes model.
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

# Make predictions on the test set.
y_pred = nb_model.predict(X_test_vectorized)

# Evaluate the model.
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}\n')
print('Confusion Matrix:\n', conf_matrix)
print('\nClassification Report:\n', class_report)
    
# Visualise results.
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Greens', cbar = False,
            xticklabels = ['Healthy', 'Illness'], yticklabels = ['Healthy', 'Illness'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.show()
