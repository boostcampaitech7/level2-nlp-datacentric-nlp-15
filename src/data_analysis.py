import os

parent_dir = os.path.dirname(os.getcwd())
data_path_train = os.path.join(parent_dir, 'data', 'train.csv')
data_path_test = os.path.join(parent_dir, 'data', 'test.csv')

# open csv file as pandas dataframe
import pandas as pd
train_data = pd.read_csv(data_path_train)
test_data = pd.read_csv(data_path_test)

def func1():
    # save csv as utf-8
    new_data_path = os.path.join(parent_dir, 'data', 'train_utf16.csv')
    train_data.to_csv(new_data_path, index=False, encoding='utf-16')

    input_texts = train_data['text']
    random_10 = input_texts.sample(10)
    print(random_10)

def func2():
    ten_data = test_data.sample(10)
    new_data_path = os.path.join(parent_dir, 'data', 'test_short.csv')
    ten_data.to_csv(new_data_path, index=False, encoding='utf-8')

func2()