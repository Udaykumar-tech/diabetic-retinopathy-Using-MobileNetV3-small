import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

DATA_DIR = 'data/aptos2019'
REPORTS_DIR = 'reports/figures'
CSV_PATH = os.path.join(DATA_DIR, 'train_1.csv')

def perform_eda():
    print("Performing Exploratory Data Analysis (EDA)...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='diagnosis', data=df)
    plt.title('Distribution of Diabetic Retinopathy Grades (APTOS 2019)')
    plt.xlabel('Diagnosis Grade')
    plt.ylabel('Number of Images')
    plot_path = os.path.join(REPORTS_DIR, 'class_distribution.png')
    plt.savefig(plot_path)
    print(f"Class distribution plot saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    perform_eda()
