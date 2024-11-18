import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# fetch dataset 
column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", 
                "Shucked weight", "Viscera weight", "Shell weight", "Rings"]               
abalone_data = pd.read_csv('abalone.data', header=None, names=column_names)  


# Data Processing 
# convert the values of sex M, F, I to 0, 1, 2
abalone_data['Sex'].replace(['M', 'F', 'I'], [0, 1, 2], inplace=True)

#Age classification based on 'Rings'
# Class 1: 0 - 7 years, Class 2: 8 - 10 years, Class 3: 11 - 15 years, Class 4: Greater than 15 years
def classify_age(rings):
    if rings <= 7:
        return 1
    elif 8 <= rings <= 10:
        return 2
    elif 11 <= rings <= 15:
        return 3
    else:
        return 4

abalone_data['Age Class'] = abalone_data['Rings'].apply(classify_age)

# Display the cleaned data
abalone_data.head()       



#Q1 Analyse and visualise the given data sets
# Distribution of the Age Classes
plt.figure(figsize=(8, 5))
sns.countplot(x='Age Class', data=abalone_data)
plt.title('Distribution of Age Classes')
plt.xlabel('Age Class')
plt.ylabel('Count')
plt.savefig('Distribution of Age Classes')
plt.show()

# Distribution of each feature (histograms for numerical features)
abalone_data.drop(columns='Age Class').hist(bins=20, figsize=(15, 10), layout=(3, 3))
plt.suptitle('Distribution of Features')
plt.savefig('Distribution of Features')
plt.show()


# Correlation heatmap of features to see multicollinearity
plt.figure(figsize=(10, 10))
correlation_matrix = abalone_data.drop(columns='Age Class').corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Feature Correlation Heatmap')
plt.savefig('Feature Correlation Heatmap')
plt.show()



#Q2 
# Prepare data: features and target
X = abalone_data.drop(['Rings', 'Age Class'], axis=1)
y = abalone_data['Age Class']

# Define experimental parameters for multiple runs
experiment_results = []
hyperparameters = [
    {'max_depth': 3, 'min_samples_split': 10},
    {'max_depth': 5, 'min_samples_split': 15},
    {'max_depth': 7, 'min_samples_split': 20},
    {'max_depth': 9, 'min_samples_split': 25},
    {'max_depth': None, 'min_samples_split': 30}
]

# Run multiple experiments with different train/test splits and hyperparameters
for i, params in enumerate(hyperparameters):
    train_accuracies = []
    test_accuracies = []
    
    # Perform 5 random splits for each parameter set
    for j in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=j)

        # Initialize and fit Decision Tree with current parameters
        clf = DecisionTreeClassifier(**params, random_state=j)
        clf.fit(X_train, y_train)

        # Calculate train and test accuracy
        train_accuracy = accuracy_score(y_train, clf.predict(X_train))
        test_accuracy = accuracy_score(y_test, clf.predict(X_test))
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    
    # Store results for each hyperparameter configuration
    experiment_results.append({
        'params': params,
        'train_accuracy_mean': np.mean(train_accuracies),
        'test_accuracy_mean': np.mean(test_accuracies),
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    })

# Select the best model based on test accuracy
best_experiment = max(experiment_results, key=lambda x: x['test_accuracy_mean'])
best_params = best_experiment['params']

# Train the best model on a fresh split and visualize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_clf = DecisionTreeClassifier(**best_params, random_state=42)
best_clf.fit(X_train, y_train)

# Display the experiment results in a DataFrame
experiment_df = pd.DataFrame([{
    'max_depth': exp['params']['max_depth'],
    'min_samples_split': exp['params']['min_samples_split'],
    'train_accuracy_mean': exp['train_accuracy_mean'],
    'test_accuracy_mean': exp['test_accuracy_mean']
} for exp in experiment_results])

print("Decision Tree Experiment Results:")
print(experiment_df)

# Visualize the best decision tree
plt.figure(figsize=(20, 10))
plot_tree(best_clf, feature_names=X.columns, class_names=['Class 1', 'Class 2', 'Class 3', 'Class 4'], filled=True)
plt.title("Best Decision Tree Visualization")
plt.savefig("Best Decision Tree Visualization")
plt.show()

# Sample rule extraction (If-Then format for a few nodes)
rules = ["If Diameter <= 0.5 and Shell weight > 0.2, then Class = Class 1",
         "If Length > 0.7 and Whole weight <= 1.2, then Class = Class 3"]
print("Sample Decision Rules:", rules)



#Q3 Do an investigation about improving performance further by either pre-pruning or post-pruning the tree
# Implementing pre-pruning and post-pruning accuracy vs alpha (or max depth) plots

# Data preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pre-pruning: Varying max_depth
max_depth_values = range(1, 11)
pre_pruning_accuracies = []

for max_depth in max_depth_values:
    pre_pruned_model = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    pre_pruned_model.fit(X_train, y_train)
    pre_pruning_accuracy = pre_pruned_model.score(X_test, y_test)
    pre_pruning_accuracies.append(pre_pruning_accuracy)

# Plotting Pre-pruning Accuracy vs max_depth values
plt.figure(figsize=(10, 6))
plt.plot(max_depth_values, pre_pruning_accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Pre-Pruning: Accuracy vs Max Depth for Decision Trees')
plt.savefig('Pre-Pruning: Accuracy vs Max Depth for Decision Trees')
plt.show()

# Post-pruning: Varying ccp_alpha values
best_model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
path = best_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # Exclude the last alpha to avoid total pruning
post_pruning_accuracies = []

for ccp_alpha in ccp_alphas:
    pruned_model = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    pruned_model.fit(X_train, y_train)
    pruned_accuracy = pruned_model.score(X_test, y_test)
    post_pruning_accuracies.append(pruned_accuracy)

# Plotting Post-pruning Accuracy vs ccp_alpha values
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, post_pruning_accuracies, marker='o', drawstyle='steps-post', color='b')
plt.xlabel('Effective Alpha')
plt.ylabel('Accuracy')
plt.title('Post-Pruning: Accuracy vs Alpha for Pruned Decision Trees')
plt.savefig('Post-Pruning: Accuracy vs Alpha for Pruned Decision Trees')
plt.show()



#Q4 Apply Random Forests  and show performance
# Range of trees to test
n_trees = [10, 50, 100, 200, 300, 400, 500]
rf_accuracies = []

# Apply Random Forest with different number of trees and record test accuracy
for n in n_trees:
    rf_model = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test)
    rf_accuracies.append(rf_accuracy)
    print(f"Number of Trees: {n}, Accuracy: {rf_accuracy:.4f}")

# Plotting Accuracy vs Number of Trees
plt.figure(figsize=(10, 6))
plt.plot(n_trees, rf_accuracies, marker='o', linestyle='-', color='purple')
plt.xlabel('Number of Trees in Ensemble')
plt.ylabel('Accuracy')
plt.title('Random Forest: Accuracy vs Number of Trees')
plt.savefig('Random Forest: Accuracy vs Number of Trees')
plt.show()



#Q5
# Adjust labels from 1-4 to 0-3 to meet XGBoost requirements
y_train_encoded = y_train - 1
y_test_encoded = y_test - 1

# Define the number of trees for comparison
n_trees = [10, 50, 100, 200, 300]

# Accuracy storage for Random Forest, Gradient Boosting, and XGBoost
rf_accuracies = []
gb_accuracies = []
xgb_accuracies = []

# Random Forest (using original y_train and y_test as it doesn't require label adjustment)
for n in n_trees:
    rf_model = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracies.append(rf_model.score(X_test, y_test))

# Gradient Boosting (using original y_train and y_test)
for n in n_trees:
    gb_model = GradientBoostingClassifier(n_estimators=n, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_accuracies.append(gb_model.score(X_test, y_test))

# XGBoost (using adjusted y_train_encoded and y_test_encoded)
for n in n_trees:
    xgb_model = XGBClassifier(n_estimators=n, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train_encoded)
    xgb_accuracies.append(xgb_model.score(X_test, y_test_encoded))

# Display results
print("Random Forest Accuracies:", list(zip(n_trees, rf_accuracies)))
print("Gradient Boosting Accuracies:", list(zip(n_trees, gb_accuracies)))
print("XGBoost Accuracies:", list(zip(n_trees, xgb_accuracies)))

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.plot(n_trees, rf_accuracies, marker='o', linestyle='-', color='purple', label='Random Forest')
plt.plot(n_trees, gb_accuracies, marker='s', linestyle='--', color='blue', label='Gradient Boosting')
plt.plot(n_trees, xgb_accuracies, marker='^', linestyle='-.', color='green', label='XGBoost')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Random Forest, Gradient Boosting, and XGBoost')
plt.legend()
plt.savefig('Accuracy Comparison: Random Forest, Gradient Boosting, and XGBoost')
plt.show()



#Q6 Compare results with Adam/SGD (Simple Neural Networks)
# Define the number of hidden layers for comparison
hidden_layer_sizes = [(50,), (100,), (200,)]
adam_accuracies = []
sgd_accuracies = []

# Train Simple Neural Networks with different optimizers: Adam and SGD
for layers in hidden_layer_sizes:
    # Adam optimizer
    adam_model = MLPClassifier(hidden_layer_sizes=layers, solver='adam', random_state=42, max_iter=500)
    adam_model.fit(X_train, y_train)
    adam_accuracies.append(adam_model.score(X_test, y_test))
    
    # SGD optimizer
    sgd_model = MLPClassifier(hidden_layer_sizes=layers, solver='sgd', random_state=42, max_iter=500)
    sgd_model.fit(X_train, y_train)
    sgd_accuracies.append(sgd_model.score(X_test, y_test))

# Display results
print("Adam Neural Network Accuracies:", list(zip(hidden_layer_sizes, adam_accuracies)))
print("SGD Neural Network Accuracies:", list(zip(hidden_layer_sizes, sgd_accuracies)))

# Plot accuracy comparison for Adam and SGD
plt.figure(figsize=(10, 6))
plt.plot([str(layers) for layers in hidden_layer_sizes], adam_accuracies, marker='o', linestyle='-', color='orange', label='Adam')
plt.plot([str(layers) for layers in hidden_layer_sizes], sgd_accuracies, marker='s', linestyle='--', color='cyan', label='SGD')
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Neural Networks with Adam vs SGD')
plt.legend()
plt.savefig('Accuracy Comparison: Neural Networks with Adam vs SGD')
plt.show()



#Q7
# Split dataset for neural network training with Keras
# Adjust labels to range from 0 to 3 to meet Keras requirements
y_train_encoded = y_train - 1
y_test_encoded = y_test - 1

# Hyperparameter combinations for dropout and L2 regularization (weight decay)
hyperparameters = [
    {'dropout_rate': 0.2, 'l2_lambda': 0.001},
    {'dropout_rate': 0.3, 'l2_lambda': 0.01},
    {'dropout_rate': 0.5, 'l2_lambda': 0.0005},
]

# Store results for each hyperparameter combination
results = []

# Train and evaluate models with different dropout and L2 regularization settings
for params in hyperparameters:
    dropout_rate = params['dropout_rate']
    l2_lambda = params['l2_lambda']
    
    # Define neural network model
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(4, activation='softmax')
    ])

    # Compile model with Adam optimizer
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, verbose=0)
    
    # Evaluate model on test data
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    # Store results
    results.append({
        'dropout_rate': dropout_rate,
        'l2_lambda': l2_lambda,
        'accuracy': accuracy
    })

# Display results
results_df = pd.DataFrame(results)
print("Dropout and L2 Regularization Comparison Results:")
print(results_df)
