
from spam_detection_api.preprocessing.data_loading import DataLoader
print('DataLoader imported')
from spam_detection_api.preprocessing.preprocessor import preprocess_dataset


data_path = '../data/raw/dataset.csv'

data_loader = DataLoader(data_path=data_path)
df = data_loader.get_data()
print(f"Before: \n {df['text'].iloc[0]}")

df = preprocess_dataset(data=df, target_column='label', email_column='text')

print(f"\nAfter: \n {df['processed_text'].iloc[0]}")

df.to_csv('../data/preprocessed/test_run.csv')
