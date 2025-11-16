import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_DIR = 'data/aptos2019'
INPUT_CSV = os.path.join(DATA_DIR, 'train_1.csv')
OUTPUT_DIR = 'data'
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

def create_data_splits():
    print("Creating stratified data splits...")
    df = pd.read_csv(INPUT_CSV)
    df['id_code'] = df['id_code'] + '.png'
    
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df['diagnosis'], random_state=RANDOM_STATE
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        stratify=train_val_df['diagnosis'],
        random_state=RANDOM_STATE
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_split.csv'), index=False)
    print("Data splits created successfully.")

if __name__ == '__main__':
    create_data_splits()
