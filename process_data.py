import os
import pandas as pd


data_dir = 'dataset/ml-100k'
genre_file = os.path.join(data_dir, 'u.genre')
item_file = os.path.join(data_dir, 'u.item')
user_file = os.path.join(data_dir, 'u.user')
user_item_file = os.path.join(data_dir, 'u.data')


def create_item_genre(genre_file, item_file):
    df_genre = pd.read_csv(genre_file, '|', names=['genre',  'index'])
    genre_lst = df_genre["genre"].tolist()
    df_item = pd.read_csv(item_file, '|', names=["item_id", "name", "year", "dummy", "url", *genre_lst])
    df = df_item.drop(["name", "year", "dummy", "url"], axis=1)
    return df 

def create_user_occupation(user_file):
    df_user_occupation = pd.read_csv(user_file, '|', names=['user_id',  'age', 'sex', 'occupation', 'zipcode'])
    return df_user_occupation.drop(['age', 'sex', 'zipcode'], axis=1)

def create_user_item(user_item_file):
    df_user_item = pd.read_csv(user_item_file, '\t', names=['user_id',  'item_id', 'rating', 'time_stamp'])
    return df_user_item

def create_user_item_genre_occupation(df_item_genre, df_user_item, df_user_occupation):
    df_user_item_occupation = df_user_item.merge(df_user_occupation, on="user_id", how = 'inner')
    df_final = df_user_item_occupation.merge(df_item_genre, on="item_id", how = 'inner')
    df_final = df_final.drop("time_stamp", axis=1)
    return df_final


def save_csv(df_user_item_genre, path='csv_file/user_item_genre_occupation.csv'):
    df_user_item_genre.to_csv(path)

df_item_genre = create_item_genre(genre_file, item_file)

df_user_item = create_user_item(user_item_file)

df_user_occupation = create_user_occupation(user_file)

df_user_item_genre_occupation = create_user_item_genre_occupation(df_item_genre, df_user_item, df_user_occupation)


save_csv(df_user_item_genre_occupation)