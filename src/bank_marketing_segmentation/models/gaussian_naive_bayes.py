import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, balanced_accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import pandas as pd

class GaussianNaiveBayes:

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
    
    def fitDistribution(self, data):
        mean = np.mean(data)
        std = np.std(data)
        dist = norm(mean,std)
        return dist
    
    def probability(self, X, dist, prior):
        prob = prior
        count = 0
        for each in dist:
          prob = prob * each.pdf(X[count])
          count +=1
        return prob
    
    def fit(self, resampling_strategy=None):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=18)

        # Resampling
        if resampling_strategy:
            if resampling_strategy == 'over':
                ros = RandomOverSampler(random_state=42)
                X_train, y_train = ros.fit_resample(X_train, y_train)
            elif resampling_strategy == 'under':
                rus = RandomUnderSampler(random_state=42)
                X_train, y_train = rus.fit_resample(X_train, y_train)
            elif resampling_strategy == 'smote':
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

          
        # Convert to NumPy arrays if they are pandas objects
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        
        # Separate the dataset by class
        X0_train = X_train[y_train == 0] 
        X1_train = X_train[y_train == 1]
        
        prior_0 = len(X0_train) / len(X_train)
        prior_1 = len(X1_train) / len(X_train)

        dist0 = [self.fitDistribution(X0_train[:, i]) for i in range(X_train.shape[1])]
        dist1 = [self.fitDistribution(X1_train[:, i]) for i in range(X_train.shape[1])]

        # Assign test set and distributions to instance variables
        self.X_test = X_test
        self.y_test = y_test
        self.dist0 = dist0
        self.dist1 = dist1
        self.prior_0 = prior_0
        self.prior_1 = prior_1

        # Now use the predict method for predictions
        actual = self.y_test
        pred = self.predict(self.X_test)

        print('Balanced accuracy:', balanced_accuracy_score(actual, pred))
        print('Confusion Matrix:\n', confusion_matrix(actual, pred))
        print('\nClassification Report:\n', classification_report(actual, pred))


        fpr, tpr, _ = roc_curve(actual, pred)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def predict(self, X):
      pred = []
      for sample in X:
        py0 = self.probability(sample, self.dist0, self.prior_0)
        py1 = self.probability(sample, self.dist1, self.prior_1)
        y_predict = np.argmax([py0, py1])
        pred.append(y_predict)
      return pred


# Prepare your data
X = encoded_bank_data.drop('y', axis=1)
y = encoded_bank_data['y']

# Instantiate and use your Gaussian Naive Bayes model
gnb = GaussianNaiveBayes(X, y)

# Run the model without resampling
gnb.fit()

# Run the model with over-sampling
gnb.fit(resampling_strategy='over')

# Run the model with under-sampling
gnb.fit(resampling_strategy='under')

# Run the model with SMOTE
gnb.fit(resampling_strategy='smote')  



# KFold

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import matplotlib.pyplot as plt

class KFGaussianNaiveBayes:

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def fitDistribution(self, data):
        mean = np.mean(data)
        std = np.std(data)
        dist = norm(mean, std)
        return dist
    
    def probability(self, X, dist, prior):
        prob = prior
        count = 0
        for each in dist:
          prob = prob * each.pdf(X[count])
          count += 1
        return prob
    
    def fit_kfold(self, n_splits=5, resampling_strategy=None):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=11)

        for fold, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            print(f"Fold {fold + 1}:")

            X_train_fold, X_test_fold = self.X[train_idx], self.X[test_idx]
            y_train_fold, y_test_fold = self.y[train_idx], self.y[test_idx]

            # Apply resampling based on the chosen strategy
            if resampling_strategy == 'over':
                ros = RandomOverSampler(random_state=42)
                X_train_fold, y_train_fold = ros.fit_resample(X_train_fold, y_train_fold)
            elif resampling_strategy == 'under':
                rus = RandomUnderSampler(random_state=42)
                X_train_fold, y_train_fold = rus.fit_resample(X_train_fold, y_train_fold)
            elif resampling_strategy == 'smote':
                smote = SMOTE(random_state=42)
                X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

            
            # Fit the model on the fold data
            self.fit_model(X_train_fold, y_train_fold, X_test_fold, y_test_fold)

    def fit_model(self, X_train, y_train, X_test, y_test):
        # Separate the dataset by class
        X0_train = X_train[y_train == 0] 
        X1_train = X_train[y_train == 1]

        # Calculate priors
        prior_0 = len(X0_train) / len(X_train)
        prior_1 = len(X1_train) / len(X_train)

        # Fit distributions
        dist0 = [self.fitDistribution(X0_train[:, i]) for i in range(X_train.shape[1])]
        dist1 = [self.fitDistribution(X1_train[:, i]) for i in range(X_train.shape[1])]

        pred = self.predict(X_test, dist0, dist1, prior_0, prior_1)
        print('Balanced accuracy:', balanced_accuracy_score(y_test, pred))
        print('Confusion Matrix:\n', metrics.confusion_matrix(y_test, pred))
        print('\nClassification report:\n', classification_report(y_test, pred))

        mat = metrics.confusion_matrix(y_test, pred)
        TP = mat[1, 1]
        FP = mat[0, 1]
        TN = mat[0, 0]
        FN = mat[1, 0]

        Cfp, Cfn, Btp, Btn = 1, 10, 20, 5

        # Calculate Total Cost (TC) and Total Benefit (TB)
        TC = (Cfp * FP) + (Cfn * FN)
        TB = (Btp * TP) + (Btn * TN)

        print("\nLet's assume a small cost analysis where we have the following")
        print(f"\n Costs and benefits: Cfp = {Cfp}, Cfn = {Cfn}, Btp = {Btp}, Btn = {Btn}")
        print(f"\n Net Benefit: Total Benefit - Total Cost = {TB} - {TC} = {TB - TC}")


        fpr, tpr, _ = roc_curve(y_test, pred)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
  

    def predict(self, X, dist0, dist1, prior_0, prior_1):
        pred = []
        for sample in X:
            py0 = self.probability(sample, dist0, prior_0)
            py1 = self.probability(sample, dist1, prior_1)
            y_predict = np.argmax([py0, py1])
            pred.append(y_predict)
        return pred

   

# Usage with encoded_bank_data
X = encoded_bank_data.drop('y', axis=1).values
y = encoded_bank_data['y'].values

# Instantiate the Gaussian Naive Bayes model
kfgnb = KFGaussianNaiveBayes(X, y)

# Run KFold cross-validation with different resampling strategies
print("KFold without resampling:")
kfgnb.fit_kfold()

print("\nKFold with over-sampling:")
kfgnb.fit_kfold(resampling_strategy='over')

print("\nKFold with under-sampling:")
kfgnb.fit_kfold(resampling_strategy='under')

print("\nKFold with SMOTE:")
kfgnb.fit_kfold(resampling_strategy='smote')

