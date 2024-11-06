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

def func3():
    random_sample_200 = 200

    # random sample 200 data rows
    random_sample = train_data.sample(random_sample_200)
    random_sample.to_csv(os.path.join(parent_dir, 'data', 'train_sample_200.csv'), index=False)

def func4():
    # this function reads train csv text and show how many korean characters are in the text
    import re
    korean_char = re.compile('[가-힣]')
    train_data['korean_char_count'] = train_data['text'].apply(lambda x: len(korean_char.findall(x)))

    # show in pandas dataframe bar plot
    train_data['korean_char_count'].plot(kind='hist', bins=50, title='Korean Character Count in Text')

    import matplotlib.pyplot as plt
    plt.show()

    print(train_data['korean_char_count'].describe())

#func2()
#func3()
func4()