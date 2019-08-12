import pandas as pd
import sys
import os
import numpy as np

r_cols = ["user_id","movie_id","rating","unix_timestamp"]

# change dataframe training and test data to numpy nd-array
def generate_train_data(train_file_location,test_file_location,n=943,m=1682):
    train_data = pd.read_csv(train_file_location, sep = "\t", names = r_cols, encoding="latin-1")
    test_data = pd.read_csv(test_file_location, sep= "\t", names = r_cols, encoding = "latin-1")
    user_record = train_data.user_id.tolist()
    movie_record = train_data.movie_id.tolist()
    ratings_record = train_data.rating.tolist()
    rating_matrix = np.zeros([n,m])
    sigma_matrix = np.zeros([n,m])
    for i in range(len(user_record)):
        rating_matrix[user_record[i]-1,movie_record[i]-1] = ratings_record[i]
        sigma_matrix[user_record[i]-1,movie_record[i]-1] = 1
    # load test_data
    user_record_test = test_data.user_id.tolist()
    movie_record_test = test_data.movie_id.tolist()
    ratings_record_test = test_data.rating.tolist()
    rating_matrix_test = np.zeros([n,m])
    print('data load finish')
    for i in range(len(user_record_test)):
        rating_matrix_test[user_record_test[i]-1,movie_record_test[i]-1] = ratings_record_test[i]
    return sigma_matrix, rating_matrix, rating_matrix_test
