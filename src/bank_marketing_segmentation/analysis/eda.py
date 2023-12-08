import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Distribution of the target variable 'y'
plt.figure(figsize=(10, 6))
sns.countplot(x='y', data=bank_data)
plt.title('Distribution of Target Variable (y)')
plt.xlabel('Subscription to Term Deposit')
plt.ylabel('Count')
plt.show()

# Univariate Analysis of some key features
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Age Distribution
sns.histplot(bank_data['age'], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution')

# Job Distribution
sns.countplot(y='job', data=bank_data, ax=axes[0, 1])
axes[0, 1].set_title('Job Distribution')

# Education Distribution
sns.countplot(y='education', data=bank_data, ax=axes[1, 0])
axes[1, 0].set_title('Education Distribution')

# Marital Status Distribution
sns.countplot(x='marital', data=bank_data, ax=axes[1, 1])
axes[1, 1].set_title('Marital Status Distribution')

plt.tight_layout()
plt.show()


# Bivariate Analysis for Categorical Features
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Job vs Target Variable
sns.countplot(y='job', hue='y', data=bank_data, ax=axes[0])
axes[0].set_title('Job Distribution by Target Variable')

# Education vs Target Variable
sns.countplot(y='education', hue='y', data=bank_data, ax=axes[1])
axes[1].set_title('Education Distribution by Target Variable')

# Marital Status vs Target Variable
sns.countplot(x='marital', hue='y', data=bank_data, ax=axes[2])
axes[2].set_title('Marital Status Distribution by Target Variable')

plt.tight_layout()
plt.show()

# Correlation Analysis for Numerical Features
# Selecting numerical features for correlation analysis
numerical_features = bank_data.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = bank_data[numerical_features].corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Numerical Features')
plt.show()

# Identifying Highly Correlated Features
# We use a threshold of 0.75 for high correlation
high_corr_threshold = 0.75
highly_correlated_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > high_corr_threshold:
            highly_correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

# Creating a DataFrame for better visualization
high_corr_df = pd.DataFrame(highly_correlated_pairs, columns=['Feature 1', 'Feature 2', 'Correlation Coefficient'])
high_corr_df.sort_values(by='Correlation Coefficient', ascending=False, inplace=True)

high_corr_df


bank_data.describe().transpose()





