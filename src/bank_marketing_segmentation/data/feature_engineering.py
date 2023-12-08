# Creating a composite economic indicator
bank_data['composite_economic_indicator'] = bank_data[['euribor3m', 'emp.var.rate', 'nr.employed']].mean(axis=1)

# Implementing Binning for the 'age' feature
# Defining age bins
age_bins = [0, 30, 40, 50, 60, 100]
age_labels = ['<30', '30-40', '40-50', '50-60', '>60']

# Creating a new column for binned age
bank_data['age_group'] = pd.cut(bank_data['age'], bins=age_bins, labels=age_labels, right=False)

# Displaying the first few rows of the modified dataset
bank_data[['age', 'age_group', 'composite_economic_indicator']].head()
#ank_data.head()



# Transform 'pdays' to a binary feature where '1' indicates the client was previously contacted (pdays < 999)
# and '0' indicates the client was not previously contacted (pdays = 999)
bank_data['pdays_contacted'] = bank_data['pdays'].apply(lambda x: 0 if x == 999 else 1)

# Now, we can drop the original 'pdays' column
bank_data = bank_data.drop('pdays', axis=1)

# Display the first few rows to verify the transformation
bank_data.head()



numerical_columns_to_normalize = ['age', 'campaign', 'previous', 'composite_economic_indicator']



from sklearn.preprocessing import StandardScaler
# Initializing the StandardScaler
scaler = StandardScaler()

# Selecting only the numerical features to normalize
df_numerical = bank_data_dropped[numerical_columns_to_normalize]

# Applying StandardScaler to these features
df_numerical_normalized = pd.DataFrame(scaler.fit_transform(df_numerical), columns=numerical_columns_to_normalize)

# Dropping the old numerical features from the original DataFrame
df_categorical = bank_data_dropped.drop(numerical_columns_to_normalize, axis=1)

# Combining the normalized numerical features with the categorical features
normalized_bank_data = pd.concat([df_numerical_normalized, df_categorical.reset_index(drop=True)], axis=1)

# Displaying the first few rows of the normalized dataset
normalized_bank_data.head()


from sklearn.preprocessing import LabelEncoder

# Create a copy of the filtered DataFrame
encoded_bank_data = normalized_bank_data.copy()

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# List of columns to label encode
columns_to_encode = ['job', 'marital', 'education', 'month', 'day_of_week', 'poutcome']

# Apply label encoding to each categorical column
for column in columns_to_encode:
    encoded_bank_data[column] = label_encoder.fit_transform(encoded_bank_data[column])

# Binary encoding using map
binary_mappings = {
    'y': {'no': 0, 'yes': 1},
    'default': {'no': 1, 'unknown': 0, 'yes': 2},
    'contact': {'cellular': 1, 'telephone': 0},
    'loan': {'yes': 1, 'no': 0, 'unknown': 2},
    'housing': {'yes': 1, 'no': 0, 'unknown': 2}
}

# Apply binary encoding mappings
for column, mapping in binary_mappings.items():
    encoded_bank_data[column] = encoded_bank_data[column].map(mapping)

encoded_bank_data