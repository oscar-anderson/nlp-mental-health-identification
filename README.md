# Identification of Clinically-Relevant Text Content using Natural Language Processing


## Repository Organisation:

- README.md: This file.
- [**/code**](https://github.com/oscar-anderson/nlp-mental-health-identification/tree/main/code): A folder containing the Python script that implements the analysis.
- [**/plot**](https://github.com/oscar-anderson/nlp-mental-health-identification/tree/main/plot): A folder containing the figures output from the script, visualising the data and analysis at different stages of the project.
  - /[EDA](https://github.com/oscar-anderson/nlp-mental-health-identification/tree/main/plot/EDA): A subfolder containing the plots output from the Exploratory Data Analysis portion of the project.
  - /[results](https://github.com/oscar-anderson/nlp-mental-health-identification/tree/main/plot/results): A subfolder containing the plots output from the model prediction and evaluation portion of the project.

## Dataset

The dataset used for this project constitutes an amended version of the [Suicide and Depression Detection dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data), published by user [Nikhileswar Komati](https://www.kaggle.com/nikhileswarkomati) on [Kaggle.com](https://www.kaggle.com).

The original dataset comprises a collection of 232,074 social media posts from the 'r/SuicideWatch' and 'r/depression' subreddits of the Reddit online platform, collected using the Pushshift API. Specifically, this dataset includes all of the posts shared to 'r/SuicideWatch' from 16th December 2008 to 2nd January 2021, as well as all of the posts shared to 'r/depression' from 1st January 2009 to 2nd January 2021. The dataset is structured with two main columns: 'text' and 'class', wherein the 'text' column contains the written Reddit post text, while the 'class' column categorises each entry with one of two labels: 'suicide' or 'non-suicide', indicating whether the post content indicates suicidal ideation or not.

For the current project, a more balanced portion of data was taken from the healthy and mental illness categories. Specifically, the resulting working dataset contained 185,406 Reddit posts comprising 93,769 healthy and 91,638 illness-indicative text entries, each categorised using the labels '0' or '1', respectively.

## Code overview

### Loading Data:
The script begins by importing the Python libraries required to undertake the project. These included NumPy, pandas, matplotlib, re, NLTK, wordcloud, sklearn and seaborn. The dataset, `text_data3.csv`, is then loaded into a pandas DataFrame for subsequent analysis.

### Preprocessing
A number of essential data preprocessing steps are then performed.
#### 1) Missing values:
  - The script checks for missing data in the DataFrame. If there are any null or NaN values, the rows containing these empty cells are removed.

#### 2) Data types:
  - For each column in the DataFrame, the script counts the occurrences of different data types and prints the results. This allows for the checking of any anomalous data types within each column of the dataset.

#### 3) Lowercase all text data:
  - All text data in the 'text' column is converted to lowercase, to ensure consistency.

#### 4) Remove URLs:
  - A regular expression is used to remove any URLs from the text data.

#### 5) Remove emojis:
  - Emojis, identified by their non-ASCII nature and unique character sets, are removed to ensure compatibility between the data and processing packages.

#### 6) Remove non-alphanumeric characters:
  - A regular expression is used to remove non-alphanumeric characters from the text data.

#### 7) Lemmatisation:
  - Each text entry is then tokenised, allowing for the lemmatisation of individual words using NLTK's WordNetLemmatizer.
 
#### 8) Remove stop words:
  - Using NLTK's English stop words set, stop words (e.g. 'the', 'and', 'is', etc.) are removed.

### Exploratory Data Analysis (EDA)
Several steps are then taken to explore patterns within the dataset.
#### 1) Distribution of healthy and illness-indicative data:
  - The script calculates and visualises the counts of each category in the 'label' column to show and understand the distribution of healthy and illness-indicative data in the dataset.
#### 2) Word cloud generation:
- Word clouds are used to visualise the words used throughout all of the text data, within the healthy text data and within the illness-indicative text data.
#### 3) Word frequency plots:
- Bar charts are used to display the most frequently used words throughout all of the text data, within the healthy text data and within the illness-indicative text data.

### Multinomial Naive Bayes:
The script then implements a Multinomial Naive Bayes algorithm for the classification of the text data, specifically used to determine whether a given piece of text is indicative of mental illness. The following key steps were undertaken to achieve this:

#### 1) Dataset splitting
- The dataset is split using an 80-20 split ratio, with 80% of the dataset being used to train the model and the remaining 20% being used for testing.

#### 2) Text vectorisation
- The text data is vectorised into a matrix of token counts, encapsulating the frequency of each word.

#### 3) Model training
- The model is then trained on the vectorised training data.

#### 4) Prediction and evaluation
- Utilising the trained model, predictions are generated from the test set, outputting performance metrics including the model's accuracy; the confusion matrix, showing the distribution of predicted labels compared to actual labels; and the classification report, presenting precision, recall, and F1-score metrics for the healthy and illness-indicative data categories.

## Results

- **Accuracy**: 90%
- **Confusion Matrix**:

|   | Predicted Healthy | Predicted Illness |
|---|-------------------|-------------------|
| **True Healthy** | 15837 | 2876 |
| **True Illness** | 720 | 17649 |

- **Classification Report**:

|                | Precision | Recall | F1-Score | Support |
|--------------- |-----------|--------|----------|---------|
| Healthy        | 0.96      | 0.85   | 0.90     | 18713   |
| Illness        | 0.86      | 0.96   | 0.91     | 18369   |
| Accuracy       |           |        | 0.90     | 37082   |
| Macro Avg      | 0.91      | 0.90   | 0.90     | 37082   |
| Weighted Avg   | 0.91      | 0.90   | 0.90     | 37082   |

As demonstrated by our found results, the constructed model demonstrates strong performance in identifying mental health indicators from text.

## Conclusion:

The project successfully explores and models mental health-related text data, using this information to build a robust and accurate Multinomial Naive Bayes classifier capable of identifying text entries as indicative of mental illness or not.
