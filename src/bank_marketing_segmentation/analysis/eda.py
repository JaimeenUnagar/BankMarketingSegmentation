import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_target_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='y', data=data)
    plt.title('Distribution of Target Variable (y)')
    plt.xlabel('Subscription to Term Deposit')
    plt.ylabel('Count')
    plt.show()

def plot_univariate_analysis(data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.histplot(data['age'], bins=30, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution')

    sns.countplot(y='job', data=data, ax=axes[0, 1])
    axes[0, 1].set_title('Job Distribution')

    sns.countplot(y='education', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Education Distribution')

    sns.countplot(x='marital', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Marital Status Distribution')

    plt.tight_layout()
    plt.show()

def plot_bivariate_analysis(data):
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    sns.countplot(y='job', hue='y', data=data, ax=axes[0])
    axes[0].set_title('Job Distribution by Target Variable')

    sns.countplot(y='education', hue='y', data=data, ax=axes[1])
    axes[1].set_title('Education Distribution by Target Variable')

    sns.countplot(x='marital', hue='y', data=data, ax=axes[2])
    axes[2].set_title('Marital Status Distribution by Target Variable')

    plt.tight_layout()
    plt.show()

def plot_correlation_analysis(data):
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = data[numerical_features].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix for Numerical Features')
    plt.show()

    high_corr_threshold = 0.75
    highly_correlated_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > high_corr_threshold:
                highly_correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

    high_corr_df = pd.DataFrame(highly_correlated_pairs, columns=['Feature 1', 'Feature 2', 'Correlation Coefficient'])
    high_corr_df.sort_values(by='Correlation Coefficient', ascending=False, inplace=True)

    return high_corr_df

def display_descriptive_statistics(data):
    return data.describe().transpose()

    
