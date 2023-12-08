# Check if categorical columns have nulls
categorical = bank_data.select_dtypes(include='object').columns

for x in categorical:
    if bank_data[x].isnull().any() or (bank_data[x] == 'unknown').any():
        print("There are nulls", x)
    else:
        print("No Nulls in", x)

# Check if numerical columns have nulls
categorical = bank_data.select_dtypes(include=['int64', 'float64']).columns

for x in categorical:
    if bank_data[x].isnull().any() or (bank_data[x] == 'unknown').any():
        print("There are nulls", x)
    else:
        print("No Nulls in", x)




# Print the unique values in the categorical columns to check if bad data exists
categorical = bank_data.select_dtypes(include='object').columns

for x in categorical:
    print(f"{x}: {bank_data[x].unique()}")

#counting Unknowns
x = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'pdays']

for i in x:
    unknown_count = (bank_data[i] == 'unknown').sum()
    print(f"{i}: {bank_data[i].unique()}, Number of 'unknown': {unknown_count}")

#identifying modes and Unknowns in columns
x = ['job', 'marital', 'education', 'default', 'housing', 'loan']

for column in x:
    mode_value = bank_data[column].mode().iloc[0]  # .mode() returns a Series, use .iloc[0] to get the first mode
    unknown_count = (bank_data[column] == 'unknown').sum()
    print(f"{column} - Mode:{mode_value}, Unknowns: {unknown_count}")



# Get unique value counts for all columns
unique_value_counts = {}

for column in bank_data.columns:
    unique_counts = bank_data[column].value_counts()
    unique_value_counts[column] = unique_counts

# Print the unique value counts for each column
for column, counts in unique_value_counts.items():
    print(f"Unique value counts for '{column}':\n{counts}\n")



# Get a list of numerical columns
numerical_columns = bank_data.select_dtypes(include=['number']).columns.tolist()

# Print the list of numerical columns
print("Numerical Features in the DataFrame:")
print(numerical_columns)


# Dropping the specified columns
columns_to_drop = ['duration', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'age_group']
bank_data_dropped = bank_data.drop(columns=columns_to_drop)

# Displaying the first few rows of the updated dataset
bank_data_dropped.head()


# Get a list of numerical columns
numerical_columns = bank_data_dropped.select_dtypes(include=['number']).columns.tolist()

# Print the list of numerical columns
print("Numerical Features in the DataFrame:")
print(numerical_columns)


# Get a list of categorical columns
categorical_columns = normalized_bank_data.select_dtypes(include=['object']).columns.tolist()

# Print categorical columns and their unique value counts
for column in categorical_columns:
    unique_values = bank_data[column].unique()
    num_unique = len(unique_values)
    value_counts = bank_data[column].value_counts()
    
    print(f"Column: {column}")
    print(f"Number of Unique Values: {num_unique}")
    print(f"Unique Values: {unique_values}")
    print("Value Counts:")
    print(value_counts)
    print("\n")



# Filtering out instances where 'default' is 'yes'
normalized_bank_data = normalized_bank_data[normalized_bank_data['default'] != 'yes']

# Dropping records with 'unknown' in 'job' and 'marital'
normalized_bank_data = normalized_bank_data[~normalized_bank_data['job'].isin(['unknown'])]
normalized_bank_data = normalized_bank_data[~normalized_bank_data['marital'].isin(['unknown'])]

# Displaying the first few rows of the updated dataset to confirm changes
normalized_bank_data.head()



