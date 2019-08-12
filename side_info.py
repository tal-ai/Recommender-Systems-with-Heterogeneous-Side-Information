import pandas as pd
import sys
import os
from time import time
import numpy as np

def change_gender(x):
    # user side information
    # M = 1, F = 0
    if x == 'M':
        return 1
    else:
        return 0

def normalize_QP_matrix(x):
    return x/x.sum(axis=0)

def user_flat(user_info_file):
    if not os.path.exists('data/X.txt'):
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        users = pd.read_csv(user_info_file, sep='|', names=u_cols,
                    encoding='latin-1')
        values = users['sex'].apply(lambda x:change_gender(x))
        users['new_sex'] = np.array(values)
        user_age = np.array(users.age.tolist())
        users['new_age'] = (user_age-np.mean(user_age))/np.std(user_age)
        user_side_info = np.array(users[['new_sex','new_age']]).T
        print('[Info]: finish generate user flat side info')
        np.savetxt('data/X.txt', user_side_info)
    else:
        print('[Info]:already have user flat information')

def user_hierarchy(user_info_file,occupation_file):
    if not os.path.exists('data/user_hierarchy.txt'):
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        users = pd.read_csv(user_info_file, sep='|', names=u_cols,
                    encoding='latin-1')
        # user hierarchical structure
        occupation = pd.read_csv(occupation_file,sep='\t',encoding = 'latin-1',header=None,names=['occu'])
        # occupation encoding
        occupation_dict = dict(zip(occupation.occu.tolist(),range(occupation.shape[0])))
        user_hier_id = users.user_id.tolist()
        user_hier_occu = users.occupation.tolist()
        user_hier_occu_id = list(map(occupation_dict.get,user_hier_occu))
        users_hierarchical_matrix = np.zeros([len(user_hier_id),len(occupation_dict)])
        for i in range(len(user_hier_id)):
            users_hierarchical_matrix[user_hier_id[i]-1,user_hier_occu_id[i]]=1
        print('[Info]: finish generate user hierarchy info matrix')
        np.savetxt('data/user_hierarchy.txt',normalize_QP_matrix(users_hierarchical_matrix))
    else:
        print('[Info]:already have user hierarchical information')

def item_flat(item_info_file):
    if not os.path.exists('data/Y.txt'):
        # movies contain two level hierarchical structures and movie side informatin(title,release_date)
        m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url','unknown', 'Action', 'Adventure', 'Animation',"Children's",
        "Comedy", 'Crime','Documentary', 'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
        movies = pd.read_csv(item_info_file, sep='|', names=m_cols,
                encoding='latin-1')
        movies_flat = movies[['unknown', 'Action', 'Adventure', 'Animation',"Children's","Comedy", 'Crime','Documentary', 'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']]
        movies_flat_matrix = np.array(movies_flat).T
        print('[Info]: finish generate item flat side info')
        np.savetxt('data/Y.txt',movies_flat_matrix)
    else:
        print('[Info]:already have item flat side information')

def item_hierarchy(item_info_file):
    if not os.path.exists('data/item_hierarchy.txt'):
        # movies contain two level hierarchical structures and movie side informatin(title,release_date)
        m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url','unknown', 'Action', 'Adventure', 'Animation',"Children's",
        "Comedy", 'Crime','Documentary', 'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
        movies = pd.read_csv(item_info_file, sep='|', names=m_cols,
                encoding='latin-1')
        movies_hierarchical = movies[['unknown', 'Action', 'Adventure', 'Animation',"Children's","Comedy", 'Crime','Documentary', 'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']]
        movies_hierarchical_matrix = np.array(movies_hierarchical)
        print('[Info]: finish generate item hierarchy side info')
        np.savetxt('data/item_hierarchy.txt',normalize_QP_matrix(movies_hierarchical_matrix))
    else:
        print('[Info]:already have item hierarchy information')

if __name__ == "__main__":
    user_flat('data/ml-100k/u.user')
    user_hierarchy('data/ml-100k/u.user','data/ml-100k/u.occupation')
    item_flat('data/ml-100k/u.item')
    item_hierarchy('data/ml-100k/u.item')