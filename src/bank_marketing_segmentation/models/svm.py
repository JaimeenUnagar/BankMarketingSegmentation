# Soft Margin SVM using SMO
# Kernel trick - Radial basis function 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


class NonLinearSVM:
    def __init__(self, X, y, C=1.0, gamma=1.0, tol=1e-3, max_iter=100):
        self.X = X
        self.y = y
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter

    def rbf_kernel(self, X1, X2):
        return np.exp(-self.gamma * np.linalg.norm(X1 - X2) ** 2)

    def fit(self):
        n_samples, n_features = self.X.shape

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.rbf_kernel(self.X[i], self.X[j])

        alpha = np.zeros(n_samples)
        b = 0.0

        for _ in range(self.max_iter):
            for i in range(n_samples):
                # Compute the predicted class label
                f_i = np.sum(alpha * self.y * K[i, :]) + b
                # Compute the error
                E_i = f_i - self.y[i]

                if (self.y[i] * E_i < -self.tol and alpha[i] < self.C) or (self.y[i] * E_i > self.tol and alpha[i] > 0):
                    # Randomly select another index j, different from i
                    j = i
                    while j == i:
                        j = np.random.randint(n_samples)

                    # Compute the predicted class label for j
                    f_j = np.sum(alpha * self.y * K[j, :]) + b
                    # Compute the error for j
                    E_j = f_j - self.y[j]

                    # Save old values of alpha
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]

                    # Compute L and H, the bounds on new possible alpha values
                    if self.y[i] != self.y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if L == H:
                        continue

                    # Compute eta, the similarity between the data points i and j
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]

                    if eta >= 0:
                        continue

                    # Compute and clip the new value for alpha_j
                    alpha[j] = alpha[j] - (self.y[j] * (E_i - E_j)) / eta
                    alpha[j] = max(L, min(H, alpha[j]))

                    # Check if alpha_j has changed significantly
                    if np.abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    alpha[i] = alpha[i] + self.y[i] * self.y[j] * (alpha_j_old - alpha[j])

                    # Compute the bias terms
                    b1 = b - E_i - self.y[i] * (alpha[i] - alpha_i_old) * K[i, i] - \
                         self.y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - self.y[i] * (alpha[i] - alpha_i_old) * K[i, j] - \
                         self.y[j] * (alpha[j] - alpha_j_old) * K[j, j]

                    # Update bias term
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

        # Save the support vectors and corresponding labels
        self.support_vectors = self.X[alpha > 1e-5]
        self.support_labels = self.y[alpha > 1e-5]
        self.alpha = alpha[alpha > 1e-5]
        self.b = b

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for alpha, support_label, support_vector in zip(self.alpha, self.support_labels, self.support_vectors):
                s += alpha * support_label * self.rbf_kernel(X[i], support_vector)
            y_pred[i] = np.sign(s + self.b)
        return y_pred

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        return acc

    def run_model(self, X_test, y_test, resampling_strategy=None):
        # Apply preprocessing and resampling based on the chosen strategy
        if resampling_strategy == 'over':
            ros = RandomOverSampler(random_state=42)
            self.X, self.y = ros.fit_resample(self.X, self.y)
        elif resampling_strategy == 'under':
            rus = RandomUnderSampler(random_state=42)
            self.X, self.y = rus.fit_resample(self.X, self.y)
        elif resampling_strategy == 'smote':
            smote = SMOTE(random_state=42)
            self.X, self.y = smote.fit_resample(self.X, self.y)
        
        self.fit()
        acc_train = self.accuracy(self.X, self.y)
        acc_test = self.accuracy(X_test, y_test)

        print("Training Accuracy:", round(acc_train * 100, 2), "%")
        print("Test Accuracy:", round(acc_test * 100, 2), "%")

        y_pred_test = self.predict(X_test)
        report = classification_report(y_test, y_pred_test)
        print("Classification Report:")
        print(report)

        balanced_acc = balanced_accuracy_score(y_test, self.predict(X_test))
        print(balanced_acc*100) 

        y_pred_labels = self.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred_labels, labels=[-1, 1])
        print("\nConfusion Matrix:")
        print(conf_matrix)

        TN, FP, FN, TP = conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]

        # Additional metrics using cost analysis
        Cfp, Cfn, Btp, Btn = 1, 10, 20, 5
        TC = (Cfp * FP) + (Cfn * FN)
        TB = (Btp * TP) + (Btn * TN) 
        net_benefit = TB - TC
        print("\nLet's assume a small cost analysis where we have the following")
        print(f"Costs and benefits: Cfp = {Cfp}, Cfn = {Cfn}, Btp = {Btp}, Btn = {Btn}")
        print(f"Net Benefit: Total Benefit - Total Cost = {TB} - {TC} = {net_benefit}" )


