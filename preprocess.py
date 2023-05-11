# PREPROCESSING FOR TASK 1

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset into a Pandas dataframe
df = pd.read_csv('english_dataset.tsv', sep= '\t')
print(df.columns)

# Remove unwanted columns
df.drop(['task_2', 'task_3'], axis=1, inplace=True)

print(df.columns)
# Remove URLs and mentions
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r"http\S+", "", x))
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r"@\S+", "", x))

# Convert text to lowercase
df['tweet'] = df['tweet'].apply(lambda x: x.lower())

# Remove non-alphanumeric characters
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Lemmatize words
# lemmatizer = WordNetLemmatizer()
# df['tweet'] = df['tweet'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Encode labels as integers
label_map = {'NOT': 0, 'HOF': 1}
df['label'] = df['task_1'].apply(lambda x: label_map[x])

# Save preprocessed dataset to a new CSV file
df.to_csv('preprocessed_hasoc_dataset_task1.csv', index=False)






# Load the dataset into a Pandas dataframe
df = pd.read_csv('hasoc2019_en_test-2919.tsv', sep= '\t')
print(df.columns)

# Remove unwanted columns
df.drop(['task_2', 'task_3'], axis=1, inplace=True)

print(df.columns)
# Remove URLs and mentions
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r"http\S+", "", x))
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r"@\S+", "", x))

# Convert text to lowercase
df['tweet'] = df['tweet'].apply(lambda x: x.lower())

# Remove non-alphanumeric characters
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Lemmatize words
# lemmatizer = WordNetLemmatizer()
# df['tweet'] = df['tweet'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Encode labels as integers
label_map = {'NOT': 0, 'HOF': 1}
df['label'] = df['task_1'].apply(lambda x: label_map[x])

# Save preprocessed dataset to a new CSV file
df.to_csv('preprocessed_hasoc_test_dataset_task1.csv', index=False)