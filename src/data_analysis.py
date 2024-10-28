import os

parent_dir = os.path.dirname(os.getcwd())
data_path = os.path.join(parent_dir, 'data', 'train.csv')

# open csv file as pandas dataframe
import pandas as pd
data = pd.read_csv(data_path)

# save csv as utf-8
new_data_path = os.path.join(parent_dir, 'data', 'train_utf16.csv')
data.to_csv(new_data_path, index=False, encoding='utf-16')

input_texts = data['text']
random_10 = input_texts.sample(10)
print(random_10)