import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def transform_pdays_to_binary(data):
    """
    Transform 'pdays' to a binary feature.

    Parameters:
    data (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: DataFrame with transformed 'pdays'.
    """
    data['pdays_contacted'] = data['pdays'].apply(lambda x: 0 if x == 999 else 1)
    return data.drop('pdays', axis=1)

def normalize_numerical_features(data, columns_to_normalize):
    """
    Normalize specified numerical features.

    Parameters:
    data (pd.DataFrame): The DataFrame to process.
    columns_to_normalize (list): List of column names to normalize.

    Returns:
    pd.DataFrame: DataFrame with normalized numerical features.
    """
    scaler = StandardScaler()
    df_numerical = data[columns_to_normalize]
    df_numerical_normalized = pd.DataFrame(scaler.fit_transform(df_numerical), columns=columns_to_normalize)
    return df_numerical_normalized

def encode_categorical_features(data, columns_to_encode, binary_mappings):
    """
    Encode categorical features.

    Parameters:
    data (pd.DataFrame): The DataFrame to process.
    columns_to_encode (list): List of column names to label encode.
    binary_mappings (dict): Dictionary for binary encoding.

    Returns:
    pd.DataFrame: DataFrame with encoded features.
    """
    label_encoder = LabelEncoder()
    for column in columns_to_encode:
        data[column] = label_encoder.fit_transform(data[column])
    
    for column, mapping in binary_mappings.items():
        data[column] = data[column].map(mapping)
    
    return data

def feature_engineering(bank_data):
    """
    Main function to perform feature engineering.

    Parameters:
    bank_data (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: DataFrame after feature engineering.
    """
    bank_data = transform_pdays_to_binary(bank_data)

    numerical_columns_to_normalize = ['age', 'campaign', 'previous', 'composite_economic_indicator']
    df_numerical_normalized = normalize_numerical_features(bank_data, numerical_columns_to_normalize)

    df_categorical = bank_data.drop(numerical_columns_to_normalize, axis=1)
    normalized_bank_data = pd.concat([df_numerical_normalized, df_categorical.reset_index(drop=True)], axis=1)

    columns_to_encode = ['job', 'marital', 'education', 'month', 'day_of_week', 'poutcome']
    binary_mappings = {
        'y': {'no': 0, 'yes': 1},
        'default': {'no': 1, 'unknown': 0, 'yes': 2},
        'contact': {'cellular': 1, 'telephone': 0},
        'loan': {'yes': 1, 'no': 0, 'unknown': 2},
        'housing': {'yes': 1, 'no': 0, 'unknown': 2}
    }
    encoded_bank_data = encode_categorical_features(normalized_bank_data, columns_to_encode, binary_mappings)

    return encoded_bank_data

