import numpy as np
import seaborn as sns
from src.bank_marketing_segmentation.data.data_loader import load_data
from src.bank_marketing_segmentation.data.preprocessing import preprocess_data
from src.bank_marketing_segmentation.data.feature_engineering import feature_engineering
from src.bank_marketing_segmentation.analysis.eda import plot_target_distribution, plot_univariate_analysis, plot_bivariate_analysis, plot_correlation_analysis, display_descriptive_statistics
from src.bank_marketing_segmentation.models.gaussian_naive_bayes import GaussianNaiveBayes, KFGaussianNaiveBayes
from src.bank_marketing_segmentation.models.logistic_regression import LogisticRegression
from src.bank_marketing_segmentation.models.svm import NonLinearSVM
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

def main():
    # Load data
    bank_data = load_data('bank_data.xlsx')

    # EDA 
    sns.set(style="darkgrid")  # Set the aesthetic style of the plots

    # Call the EDA functions
    print("\t\t\tExploratory Data Analysis:")
    plot_target_distribution(bank_data)
    plot_univariate_analysis(bank_data)
    plot_bivariate_analysis(bank_data)
    plot_correlation_analysis(bank_data)
    descriptive_stats = display_descriptive_statistics(bank_data)
    print(descriptive_stats)

    # Preprocess data
    preprocessed_data = preprocess_data(bank_data)

    # Feature Engineering
    encoded_bank_data = feature_engineering(preprocessed_data)
    
    # Model Training and Evaluation
    X = encoded_bank_data.drop('y', axis=1)  # Features
    y = encoded_bank_data['y']               # Target
    
    print("\n\t\t\tLogistic Regression Model:")
    lr = LogisticRegression(X, y, learning_rate=0.1e-5, epsilon=0.00005, max_iterations=2500)

    # Running model with different resampling strategies
    print("\nWithout Resampling:")
    lr.run_model(ldpara=0)  # Without resampling
    print("\nWith Over-sampling:")
    lr.run_model(ldpara = 0, resampling_strategy='over')  # Over-sampling
    print("\nWith Under-sampling:")
    lr.run_model(ldpara = 0, resampling_strategy='under') # Under-sampling
    print("\nWith SMOTE:")
    lr.run_model(ldpara = 0, resampling_strategy='smote') # SMOTE

    # Running model with KFold and different resampling strategies
    num_splits = 5
    print("\n\t\t\tKFold Cross-validation:")
    print("\nWithout Resampling:")
    lr.run_kfold(num_splits, ldpara=0)  # Without resampling
    print("\nWith Over-sampling:")
    lr.run_kfold(num_splits, ldpara=0, resampling_strategy='over') # Over-sampling
    print("\nWith Under-sampling:")
    lr.run_kfold(num_splits, ldpara=0, resampling_strategy='under')# Under-sampling
    print("\nWith SMOTE:")
    lr.run_kfold(num_splits, ldpara=0, resampling_strategy='smote')# SMOTE
    
    #GNB
    # Instantiate and use your Gaussian Naive Bayes model
    print("\n\t\t\tGaussian Naive Bayes Model:")
    gnb = GaussianNaiveBayes(X, y)

    # Run the Gaussian Naive Bayes model with different resampling strategies
    print("Gaussian Naive Bayes without resampling:")
    gnb.fit()
    print("\nGaussian Naive Bayes with over-sampling:")
    gnb.fit(resampling_strategy='over')
    print("\nGaussian Naive Bayes with under-sampling:")
    gnb.fit(resampling_strategy='under')
    print("\nGaussian Naive Bayes with SMOTE:")
    gnb.fit(resampling_strategy='smote')
    
    # K-Fold Gaussian Naive Bayes model usage
    kfgnb = KFGaussianNaiveBayes(X, y)

    # Run KFold cross-validation with different resampling strategies
    print("\nKFold Gaussian Naive Bayes without resampling:")
    kfgnb.fit_kfold()
    print("\nKFold Gaussian Naive Bayes with over-sampling:")
    kfgnb.fit_kfold(resampling_strategy='over')
    print("\nKFold Gaussian Naive Bayes with under-sampling:")
    kfgnb.fit_kfold(resampling_strategy='under')
    print("\nKFold Gaussian Naive Bayes with SMOTE:")
    kfgnb.fit_kfold(resampling_strategy='smote')
    
    #SVM
    # Prepare data for SVM model
    print("\n\t\t\tSupport Vector Machine (SVM) Model:")
    X_svm = encoded_bank_data.drop('y', axis=1).values
    y_svm = encoded_bank_data['y'].replace(0, -1).values  # Convert class labels for SVM
    
    # Split the data into training and test sets for SVM
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size=0.25, random_state=11, stratify=y_svm)
 

    X_resampled, y_resampled = X_train_svm, y_train_svm
 
    # Specify the desired number of samples for each class
    desired_samples_per_class = 500
 
    # Select the specified number of samples for each class
    X_sampled = []
    y_sampled = []
 
    for class_label in np.unique(y_resampled):
        class_indices = np.where(y_resampled == class_label)[0]
   
        # Randomly select samples if there are more than the desired number
        selected_indices = np.random.choice(class_indices, size=min(desired_samples_per_class, len(class_indices)), replace=False)
        
        X_sampled.append(X_resampled[selected_indices])
        y_sampled.append(y_resampled[selected_indices])
 
    X_sampled = np.concatenate(X_sampled, axis=0)
    y_sampled = np.concatenate(y_sampled, axis=0)
 
    #Create and run your SVM model
    nl_svm = NonLinearSVM(X_sampled, y_sampled, C=1.0, gamma=0.1)
    
    # Run the SVM model with various resampling strategies
    print("\nNon-Linear SVM Model without resampling:")
    nl_svm.run_model(X_test_svm, y_test_svm)  # No resampling
    print("\nNon-Linear SVM Model with over-sampling:")
    nl_svm.run_model(X_test_svm, y_test_svm, resampling_strategy='over')
    print("\nNon-Linear SVM Model with under-sampling:")
    nl_svm.run_model(X_test_svm, y_test_svm, resampling_strategy='under')
    print("\nNon-Linear SVM Model with SMOTE:")
    nl_svm.run_model(X_test_svm, y_test_svm, resampling_strategy='smote')

if __name__ == '__main__':
    main()

    

