import math
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.model_selection import train_test_split

class LogisticRegression:
        
    def __init__(self, X, y, learning_rate, epsilon, max_iterations):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.theta = None

    def split_data(self, resampling_strategy=None):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        
        if resampling_strategy == 'over':
            ros = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        elif resampling_strategy == 'under':
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        elif resampling_strategy == 'smote':
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def add_X0(self, X):
        return np.column_stack([np.ones([X.shape[0], 1]), X])

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X.dot(self.theta)))

    def normalize_train(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / std
        X = self.add_X0(X)
        return X, mean, std

    def normalize_test(self, X, mean, std):
        X = (X - mean) / std
        X = self.add_X0(X)
        return X

    def apply_smote(self, X, y):
        smote = SMOTE(random_state=11)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def cost_function(self, X, y, ldpara):
        sig = self.sigmoid(X)
        pred = y * np.log(sig) + (1 - y) * np.log(1 - sig) 
        cost = pred.sum()
        reg_term = (ldpara / (2 * X.shape[0])) * np.sum(self.theta[1:]**2)
        return -cost + reg_term

    def cost_derivative(self, X, y, ldpara):
        sig = self.sigmoid(X)
        grad = (sig - y).dot(X)
        reg_term = 2*(ldpara) * np.concatenate(([0], self.theta[1:]))
        return grad  + reg_term

    def confusion_matrix(self, X, y_true):
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)
        return cm 

    def gradient_descent(self, X, y, ldpara, m):
        errors = []
        prev_error = float('inf')

        for i in tqdm(range(self.max_iterations)):
            regularized_term = (ldpara / m) * np.concatenate(([0], self.theta[1:]))
            self.theta = self.theta - self.learning_rate * (self.cost_derivative(X, y, ldpara) + regularized_term)
            error = self.cost_function(X, y, ldpara)
            diff = prev_error - error

            errors.append(abs(error))

            if diff < self.epsilon:
                print("Model stopped learning")
                break
        return errors

    def predict(self, X):
        return np.around(self.sigmoid(X))

    def predict_proba(self, X):
        return self.sigmoid(X)

    def run_model(self, ldpara=0, resampling_strategy=None):
        X_train, X_test, y_train, y_test = self.split_data(resampling_strategy)
        X_train, mean, std = self.normalize_train(X_train)
        X_test = self.normalize_test(X_test, mean, std)

        self.theta = np.ones(X_train.shape[1], dtype=np.float64)
        errors = self.gradient_descent(X_train, y_train, ldpara, X_train.shape[0])
        self.plot_cost(errors)
        self.metrics_calc(X_test, y_test)
        self.plot_roc(X_test, y_test)

    def run_kfold(self, k, ldpara, resampling_strategy=None):
        kf = KFold(n_splits=k, shuffle=True, random_state=11)
        fold_count = 1

        for train_idx, test_idx in kf.split(self.X):
            print(f"Fold {fold_count}:")

            X_train_fold, X_test_fold = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train_fold, y_test_fold = self.y.iloc[train_idx], self.y.iloc[test_idx]

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

            X_train_fold, mean, std = self.normalize_train(X_train_fold)
            X_test_fold = self.normalize_test(X_test_fold, mean, std)

            self.theta = np.ones(X_train_fold.shape[1], dtype=np.float64)
            self.gradient_descent(X_train_fold, y_train_fold, ldpara, X_train_fold.shape[0])
            
            print("\nTest Set Metrics:")
            self.metrics_calc(X_test_fold, y_test_fold)
            self.plot_roc(X_test_fold, y_test_fold)

            fold_count += 1
            
    def metrics_calc(self, X, y, threshold=0.4):
        probabilities = self.predict_proba(X)
        adjusted_predictions = (probabilities >= threshold).astype(int)

        TP = np.sum((adjusted_predictions == 1) & (y == 1))
        FP = np.sum((adjusted_predictions == 1) & (y == 0))
        TN = np.sum((adjusted_predictions == 0) & (y == 0))
        FN = np.sum((adjusted_predictions == 0) & (y == 1))

        print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
        cm = confusion_matrix(y, adjusted_predictions)
        print("\nConfusion Matrix:")
        print(cm)
        print('\nClassification report:\n', classification_report(y, adjusted_predictions))
        balanced_acc = balanced_accuracy_score(y, adjusted_predictions)
        print("\nBalanced Accuracy:", round(balanced_acc*100, 2), "%")
        Cfp, Cfn, Btp, Btn = 1, 10, 20, 5
        TC = (Cfp * FP) + (Cfn * FN)
        TB = (Btp * TP) + (Btn * TN) 
        print("\nLet's assume a small cost analysis where we have the following")
        print(f"\n Costs and benefits: Cfp =  {Cfp}, Cfn = {Cfn}, Btp = {Btp}, Btn = {Btn}")
        print(f"\n Net Benefit: Total Benefit - Total Cost = {TB} - {TC} = {TB - TC}" )

    def plot_cost(self, cost_sequence):
        s = np.array(cost_sequence)
        t = np.arange(s.size)
        fig, ax = plt.subplots()
        ax.plot(t, s)
        ax.set(xlabel='Iterations', ylabel='Cost', title='Cost Trend')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, shadow=True)
        plt.show()

    def plot_roc(self, X_test, y_test):
        fig, ax = plt.subplots(figsize=(6, 4))

        preds = self.predict_proba(X_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)

        ax.plot(fpr, tpr, 'blue', label='AUC = {:.2f}'.format(roc_auc))
        ax.plot([0, 1], [0, 1], '--')
        ax.set_title('ROC Curve', fontsize=20)
        ax.set_ylabel('TP', fontsize=20)
        ax.set_xlabel('FP', fontsize=15)
        ax.legend(loc='lower right', prop={'size': 16})
        plt.show()


    
# Your data preparation and instantiation of LogisticRegression
X = encoded_bank_data.drop('y', axis=1)  # Features from encoded_bank_data
y = encoded_bank_data['y']               # Target from encoded_bank_data
# Instantiate the LogisticRegression class with your encoded data
lr = LogisticRegression(X, y, learning_rate=0.1e-5, epsilon=0.00005, max_iterations=2500)

# Example usage of the model with different resampling strategies
lr.run_model(ldpara = 0)  # Run without resampling
lr.run_model(ldpara = 0, resampling_strategy='over')  # Run with over-sampling
lr.run_model(ldpara = 0, resampling_strategy='under')  # Run with under-sampling
lr.run_model(ldpara = 0, resampling_strategy='smote')  # Run with SMOTE


# Example usage of the model with different resampling strategies for KFold
num_splits = 5  # For example, use 5 splits
lr.run_kfold(num_splits, ldpara=0, resampling_strategy=None)   # Run KFold without resampling
lr.run_kfold(num_splits, ldpara=0, resampling_strategy='over') # Run KFold with over-sampling
lr.run_kfold(num_splits, ldpara=0, resampling_strategy='under')# Run KFold with under-sampling
lr.run_kfold(num_splits, ldpara=0, resampling_strategy='smote')# Run KFold with SMOTE