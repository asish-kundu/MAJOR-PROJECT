import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

# File Paths
input_file_path = r"merged_all_kidney_miRNA_labeled_data_processed_with_head.csv"
output_directory = os.path.dirname(input_file_path)
output_best_features_path = os.path.join(output_directory, "best_features_DT_Lasso.csv")
output_classification_results_path = os.path.join(output_directory, "Lasso_DT_classification_result.csv")
output_top_feature_rank_path = os.path.join(output_directory, "top_400_feature_ranks_Lasso.csv")

# Read Data
df = pd.read_csv(input_file_path,sep='\t')
y = df.iloc[:, -1].values
X = df.iloc[:, 1:-1].values

# Lists to store results
num_features_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# File Handlers
with open(output_best_features_path, "w") as f_best, \
        open(output_classification_results_path, "w") as file_su:

    for i in range(10, 401, 10):
        # Lasso model
        lasso = Lasso(alpha=0.01)

        # Feature selection
        sfm = SelectFromModel(lasso, max_features=i)
        X_selected = sfm.fit_transform(X, y)

        # feature indices
        selected_indices = sfm.get_support(indices=True)
        best_features_names = df.columns[selected_indices]
        f_best.writelines('\n'.join(best_features_names) + "\n")

        # Rank 
        feature_ranks = np.arange(1, len(selected_indices) + 1)

        # store the ranks
        feature_rank_df = pd.DataFrame({
            'Feature': best_features_names,
            'Rank': feature_ranks,
        })

        # Write 
        feature_rank_df.to_csv(output_top_feature_rank_path, index=False)

        # DT
        DT = DecisionTreeClassifier()

        # Performing 10-fold cross-validation and getting predicted labels
        y_pred_cv = cross_val_predict(DT, X_selected, y, cv=10)

        # Getting the cross-validation accuracy scores
        accuracy_scores = cross_val_score(DT, X_selected, y, cv=10)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y, y_pred_cv)
        precision = precision_score(y, y_pred_cv, average='macro')
        recall = recall_score(y, y_pred_cv, average='macro')
        f1 = f1_score(y, y_pred_cv, average='macro')

        # Append results to lists
        num_features_list.append(i)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # Write results to file
        file_su.write(f"Number of features: {i}\n")
        file_su.write(f"Accuracy: {accuracy}\n")
        file_su.write(f"Precision: {precision}\n")
        file_su.write(f"Recall: {recall}\n")
        file_su.write(f"F1 Score: {f1}\n\n")

        # Print results
        print("Number of features:", i)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_features_list, accuracy_list, label='Accuracy', marker='o')
plt.plot(num_features_list, precision_list, label='Precision', marker='o')
plt.plot(num_features_list, recall_list, label='Recall', marker='o')
plt.plot(num_features_list, f1_list, label='F1 Score', marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Metrics')
plt.title('Performance Metrics vs. Number of Features')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_directory, 'performance_metrics_vs_num_features_Lasso.png'))
