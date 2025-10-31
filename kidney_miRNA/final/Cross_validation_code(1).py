import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

# File Paths
input_file_path = (r"/content/multiclass dataset.csv")
output_directory = os.path.dirname("/content/multiclass dataset.csv")
output_min_accuracy_plot_path = os.path.join(output_directory, 'min_accuracy_of_classifiers.png')
output_max_accuracy_plot_path = os.path.join(output_directory, 'max_accuracy_of_classifiers.png')
output_mean_accuracy_plot_path = os.path.join(output_directory, 'mean_accuracy_of_classifiers.png')
output_classification_results_path = os.path.join(output_directory, "stratified_cross_validation_results.txt")

# Data
df = pd.read_csv("/content/multiclass dataset.csv") 
X = df.drop('Outcome', axis=1)
y = df['Outcome']


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define classifiers with hyperparameters
classifiers = {
    "SVM": SVC(),
    "ANN": MLPClassifier(),
    "KNN": KNeighborsClassifier(),
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "NB": GaussianNB(),
    "DISCR": LinearDiscriminantAnalysis()
}

# StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# storing results
classifier_names = list(classifiers.keys())
min_accuracy_list = []
max_accuracy_list = []
mean_accuracy_list = []


for train_index, test_index in skf.split(X_scaled, y):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    clf.fit(X_train_fold, y_train_fold)
    y_pred_fold = clf.predict(X_test_fold)
    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
    lst_accu_stratified.append(accuracy_fold)

# Handlers
with open(output_classification_results_path, "w") as file_su:
    for clf_name, clf in classifiers.items():
        # cross-validation
        lst_accu_stratified = []
        for train_index, test_index in skf.split(X_scaled, y):
            X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            clf.fit(X_train_fold, y_train_fold)
            y_pred_fold = clf.predict(X_test_fold)
            accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
            lst_accu_stratified.append(accuracy_fold)

        # evaluation metrics
        mean_accuracy = np.mean(lst_accu_stratified)
        min_accuracy = np.min(lst_accu_stratified)
        max_accuracy = np.max(lst_accu_stratified)

        # Append results
        min_accuracy_list.append(min_accuracy)
        max_accuracy_list.append(max_accuracy)
        mean_accuracy_list.append(mean_accuracy)

        # Write results
        file_su.write(f"Classifier: {clf_name}\n")
        file_su.write(f"Mean Accuracy: {mean_accuracy}\n")
        file_su.write(f"Minimum Accuracy: {min_accuracy}\n")
        file_su.write(f"Maximum Accuracy: {max_accuracy}\n\n")

        # Print results
        print("Classifier:", clf_name)
        print("Mean Accuracy:", mean_accuracy)
        print("Minimum Accuracy:", min_accuracy)
        print("Maximum Accuracy:", max_accuracy)

# Minimum Accuracy plot
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, min_accuracy_list, color='red')
plt.xlabel('Classifiers')
plt.ylabel('Minimum Accuracy')
plt.title('Minimum Accuracy of Different Classifiers with 10-Fold Stratified Cross Validation')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(output_min_accuracy_plot_path)
plt.show()

# Maximum Accuracy plot
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, max_accuracy_list, color='blue')
plt.xlabel('Classifiers')
plt.ylabel('Maximum Accuracy')
plt.title('Maximum Accuracy of Different Classifiers with 10-Fold Stratified Cross Validation')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(output_max_accuracy_plot_path)
plt.show()

# Mean Accuracy plot
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, mean_accuracy_list, color='green')
plt.xlabel('Classifiers')
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy of Different Classifiers with 10-Fold Stratified Cross Validation')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(output_mean_accuracy_plot_path)
plt.show()
