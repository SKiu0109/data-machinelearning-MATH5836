import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix



# hide warnings
import warnings
warnings.filterwarnings('ignore')

# fetch dataset 
column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", 
                "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
abalone_data = pd.read_csv('abalone.data', header=None, names=column_names)  

X = abalone_data.loc[:, abalone_data.columns != 'Rings']
y = abalone_data['Rings']

# Data Processing 
# convert the values of sex M, F, I to 0, 1, 2
X['Sex'].replace(['M', 'F', 'I'], [0, 1, 2], inplace=True)

# Display the first few rows to verify the transformation
print(X.head())

# plot the correlation matrix
df = pd.concat([X, y], axis = 1)
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm', linewidths=0.5)
plt.title("Correlation of abalone dataset")
plt.savefig('Correlation_heatmap.png')

# plot the most correlated features of diameter and shell weight against the Rings
plt.figure(figsize=(10, 6))
plt.scatter(X['Diameter'], y, alpha=0.6, color='darkorange')
plt.title('Scatter Plot: Diameter vs Rings')
plt.xlabel('Diameter')
plt.ylabel('Rings (Age)')
plt.savefig('Scatter Plot: Diameter vs Rings')

plt.figure(figsize=(10, 6))
plt.scatter(X['Shell weight'], y, alpha=0.6, color='royalblue')
plt.title('Scatter Plot: Shell Weight vs Rings')
plt.xlabel('Shell Weight (grams)')
plt.ylabel('Rings (Age)')
plt.savefig('Scatter Plot: Shell Weight vs Rings')

# Histogram for Shell weight
plt.figure(figsize=(10, 6))
sns.histplot(X['Shell weight'], bins=30, kde=True, color='blue', alpha=0.7)
plt.title('Histogram: Shell Weight')
plt.xlabel('Shell Weight (grams)')
plt.ylabel('Frequency')
plt.savefig('Histogram: Shell Weight')

# Histogram for Diameter
plt.figure(figsize=(10, 6))
sns.histplot(X['Diameter'], bins=30, kde=True, color='blue', alpha=0.7)
plt.title('Histogram: Diameter')
plt.xlabel('Diameter')
plt.ylabel('Frequency')
plt.savefig('Histogram: Diameter')

# Histogram for Rings (Age)
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True, color='blue', alpha=0.7)
plt.title('Histogram: Rings (Age)')
plt.xlabel('Rings (Age)')
plt.ylabel('Frequency')
plt.savefig('Histogram: Rings (Age)')


# Modelling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)



# Linear Regression 
linreg = LinearRegression().fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print("Unnormalized Linear Regression Results")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Incorporation of normalisation linear regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_train, y_train)

# print the R2 and RMSE
y_pred_norm = pipe.predict(X_test)
print("Normalised Linear Regression Results")
print(f"Normalised R2: {r2_score(y_test, y_pred_norm):.4f}")
print(f"Normalised RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_norm)):.4f}")

# Regression Result Visualization
# Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Regression Result: Actual vs Predicted')
plt.xlabel('Actual Ring Age')
plt.ylabel('Predicted Ring Age')
plt.savefig('Regression Result: Actual vs Predicted')



# Logisitic Regression 
# data cleaning to convert the y_values to binary classifcation
y_train_binary = y_train.copy()
y_train_binary.loc[y_train_binary < 7] = 0
y_train_binary.loc[y_train_binary >= 7] = 1

y_test_binary = y_test.copy()
y_test_binary.loc[y_test_binary < 7] = 0
y_test_binary.loc[y_test_binary >= 7] = 1

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train_binary)
logreg = LogisticRegression(random_state=42).fit(X_train, y_train_binary)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
y_pred_binary = mlp.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title(f'ROC curve without normalisation {np.round(roc_auc_score(y_test_binary, y_pred_proba), 2)}')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.savefig('Logisitc_regression_no_normalising')

print("Unnormalised Logistic Regression Results")
print(confusion_matrix(y_test_binary, y_pred_binary))
print(f"AUC: {roc_auc_score(y_test_binary, logreg.predict_proba(X_test)[:, 1]):.4f}")

# create logisitc regression with normalised data
pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
pipe.fit(X_train, y_train_binary)
y_pred_proba_norm = pipe.predict(X_test)

# plot the ROC curve and the classifation scores
fpr, tpr, thresholds = roc_curve(y_test_binary, pipe.predict_proba(X_test)[:,1])
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title(f'ROC curve with normalisation {np.round(roc_auc_score(y_test_binary, y_pred_proba), 2)}')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.savefig('Logisitc_regression_with_normalising')

print("Normalised Logistic Regression Results")
print(confusion_matrix(y_test_binary, y_pred_proba_norm))
print(f"Normalised AUC: {roc_auc_score(y_test_binary, pipe.predict_proba(X_test)[:, 1]):.4f}")



# Develop a linear/logistic regression model with 2 features
# we will develop a model with the 2 most correlated features with the Rings - those where Shell weight and Diameter
# Extract two features (Diameter and Shell weight)
X_train_2 = X_train[['Shell weight', 'Diameter']]
X_test_2 = X_test[['Shell weight', 'Diameter']]

# Linear regression models: two features
linreg_2 = LinearRegression().fit(X_train_2, y_train)
y_pred_2 = linreg_2.predict(X_test_2)
print("Two Features Linear Regression Results ")
print(f"R2: {r2_score(y_test, y_pred_2):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_2)):.4f}")

# Logistic regression models: two features
logreg_2 = LogisticRegression(random_state=42).fit(X_train_2, y_train_binary)
y_pred_binary_2 = logreg_2.predict(X_test_2)
print("Two Features Logistic Regression Results ")
print(confusion_matrix(y_test_binary, y_pred_binary_2))
print(f"AUC: {roc_auc_score(y_test_binary, logreg_2.predict_proba(X_test_2)[:, 1]):.4f}")



# Neural network trained with SGD
# build a model with MLPCLassifier

# have 2 hidden layers with 10 neurons each
mlp = MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=100000, random_state=42)
mlp.fit(X_train, y_train_binary)

y_pred_binary = mlp.predict(X_test)
y_pred_proba = mlp.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test_binary, y_pred_proba)

# plot the ROC curve and the classifation scores
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title(f'ROC curve {np.round(roc_auc_score(y_test_binary, y_pred_proba), 2)}')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.savefig('Neural Network ROC')
plt.show()


param_grid = {
    'hidden_layers' : [1, 2, 3],
    'hidden_layer_neurons' : [16, 32, 64],
    'learning_rate' : [0.001, 0.01, 0.1]
}

print("----------------------")
print("Beginning of hyper parameter searching")
print("----------------------")

# loop through the parameters and print the results when training the model
for hidden_layers in param_grid['hidden_layers']:
    for hidden_layer_neurons in param_grid['hidden_layer_neurons']:
        for learning_rate in param_grid['learning_rate']:
            # Create the hidden_layer_sizes tuple dynamically
            hidden_layer_sizes = tuple([hidden_layer_neurons] * hidden_layers)

            # Define and train the model
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate, random_state=42)
            mlp.fit(X_train, y_train_binary)

            # Predict and calculate ROC AUC
            y_pred_proba = mlp.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test_binary, y_pred_proba)
            acc = accuracy_score(y_test_binary, y_pred_binary)

            # Print the configuration and its corresponding ROC AUC score
            print(f'Hidden layers: {hidden_layers}, Neurons per layer: {hidden_layer_neurons}, Learning rate: {learning_rate}, ROC AUC: {np.round(roc_auc, 3)},  acc: {np.round(acc, 3)}')

print("----------------------")
print("End of hyper parameter searching")
print("----------------------")

# all the models had performed strongly on accuracy with most ranging from 92+. The area under curve scores varies a lot more with the range going from 0.68 to 0.84.
# The model with the best roc is the one with 2 hidden layers, 32 neurons per layer and a learning rate of 0.1, with a roc of 0.848 and accurary of 0.924

f1_scores = []
accuracy_scores = []
roc_auc_scores = []
precision_scores = []
recall_scores = []

# loop through the parameters and print the results when training the model

for i in range(30):
    mlp = MLPClassifier(hidden_layer_sizes = (32, 32), learning_rate_init = 0.1, random_state = i)
    mlp.fit(X_train, y_train_binary)
    y_pred_binary = mlp.predict(X_test)

    f1_scores.append(f1_score(y_test_binary, y_pred_binary))
    accuracy_scores.append(accuracy_score(y_test_binary, y_pred_binary))
    roc_auc_scores.append(roc_auc_score(y_test_binary, y_pred_binary))
    precision_scores.append(precision_score(y_test_binary, y_pred_binary))
    recall_scores.append(recall_score(y_test_binary, y_pred_binary))

print(f'Average F1 score: {np.round(np.mean(f1_scores), 3)} with std: {np.round(np.std(f1_scores), 3)}')
print(f'Average Accuracy score: {np.round(np.mean(accuracy_scores), 3)} with std: {np.round(np.std(accuracy_scores), 3)}')
print(f'Average AUC score: {np.round(np.mean(roc_auc_scores), 3)} with std: {np.round(np.std(roc_auc_scores), 3)}')
print(f'Average Precision score: {np.round(np.mean(precision_scores), 3)} with std: {np.round(np.std(precision_scores), 3)}')
print(f'Average Recall score: {np.round(np.mean(recall_scores), 3)} with std: {np.round(np.std(recall_scores), 3)}')


'''
It is worth noting the model selected with 2 hidden layers of 32 neurons each and
 a learning rate of 0.1 on average had a similar accuracy score to the first test. 
 However the average AUC value is significantly lower at 0.741 than the one 
 done in the first experiment of 0.848. In all scenarios the model is fed the same 
 training and testing data both in the hyper parameter optimisation and the 30 experiments, 
 with the only difference being the random_state provided to the neural net. 
 The lower performance of the model with high standard deviation could be 
 to the high learning rate - the highest of the 3 learning rates tested. 
 In the first trial run the model may have been lucky to find the optimal parameters 
 and had failed to do so in the 30 following experiments.
'''
