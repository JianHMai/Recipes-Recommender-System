import os
import glob
import re
import pandas as pd
import csv
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from math import floor

def interaction_files():
    interaction_files = [i for i in glob.glob('interactions*.{}'.format("csv"))]
    combined_csv = pd.concat([pd.read_csv(f) for f in interaction_files ])
    combined_csv.to_csv( "interactions.csv", index=False, encoding='utf-8')

def raw_interactions_file():
    with open('RAW_interactions.csv', newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        with open('cleaned_interactions.csv', 'w', newline='', encoding="utf-8") as csvfile2: 
            csvwriter = csv.writer(csvfile2)
            csvwriter.writerow(['user_id', 'recipe_id', 'date', 'rating', 'review'])
            for data in reader:
                user_id = data[0]
                recipe_id = data[1]
                date = data[2]
                rating = data[3]
                review = data[4]

                review = review.replace('.',' ').translate(str.maketrans('', '', string.punctuation)).lower().replace('<br/>', ' ')
                review_tokenize = word_tokenize(review)
                final_review = ''

                # Import stemmer
                ps = PorterStemmer()
                # Import list from NLTK library that contains stop words
                stop_words = set(stopwords.words('english'))

                # Loop to go through list of words
                for word in review_tokenize: 
                    # Check in word is a stop word
                    if not word in stop_words:
                        # Calls the porter algo to stem
                        final_review += ps.stem(word) + " " 
                # writing the data rows 
                csvwriter.writerow([user_id, recipe_id, date,rating, final_review])
    csvfile.close
    csvfile2.close

def create_interactions_dataset(num):
    data = pd.read_csv("interactions.csv")
    data = data[['u', 'i', 'rating']]
    data = shuffle(data)
    if num == 50:
        data = data[:-floor(data.shape[0]*0.50)]
    elif num == 25: 
        data = data[:-floor(data.shape[0]*0.75)]
    data.to_csv('interactions_' + str(num)+ '.csv', sep=',', encoding='utf-8', index = False)