
"""
DATACAMP CERTIFICATION : DATA SCIENCE
"""

####################################################
'''
0. Import the useful packages
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

####################################################
'''
1. For every column in the data:
a. State whether the values match the description given in the table above.
b. State the number of missing values in the column.
c. Describe what you did to make values match the description if they did not
match.
'''
df = pd.read_csv('fitness_class_2212.csv')
df.describe(include='all')
print(df)
df = df.set_index('booking_id')

day = pd.CategoricalDtype(categories=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'unknown'], ordered=False)
time = pd.CategoricalDtype(categories=['AM', 'PM', 'unknown'], ordered=True)
category = pd.CategoricalDtype(categories=['Yoga', 'Aqua', 'Strength', 'HIIT', 'Cycling'], ordered=False)

df['days_before'] = df['days_before'].str.extract(r'(\d+)')

df['day_of_week'] = df['day_of_week'].fillna('unknown')
df['time'] = df['time'].fillna('unknown')
df['category'] = df['category'].fillna('unknown')
df['days_before'] = df['days_before'].fillna(0)
df['weight'] = df['weight'].fillna(np.mean(df['weight']))

df = df.dropna(subset='attended')

df = df.astype({
    'days_before': 'int32',
    'day_of_week': day,
    'time': time,
    'category': category})
print(df)

"""
- booking_id: the identifier is unique. There are 0 missing values. 
- mois_comme_membre: type is already 'int64'. There are no missing values.
- weight : type is already 'float64'.
- days_before : 
- day_of_week :
- hour :
- category :
- assisted : 
"""

####################################################
'''
2. Create a visualization that shows how many bookings attended the class. Use the
visualization to:
a. State which category of the variable attended has the most observations
b. Explain whether the observations are balanced across categories of the
variable attended
'''
# A.
category_colors = ['red', 'green', 'blue', 'purple', 'orange']
category_counts = df.groupby('category')['attended'].count()
plt.bar(category_counts.index, category_counts.values, color=category_colors)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Count of attended')
plt.show()

# B.
# Figure 1 - Attendance Count shows that the majority of members did not attend the
# class (0), outnumbering those who did (1). The count of non-attendees is approximately
# two times higher than the count of attendees.
category_counts = df['attended'].value_counts()
plt.bar(category_counts.index, category_counts.values, color=category_colors)
plt.xlabel('Attended')
plt.ylabel('Count')
plt.title('Count of attended')
plt.xticks(category_counts.index)
plt.show()

####################################################
'''
3. Describe the distribution of the number of months as a member. Your answer must
include a visualization that shows the distribution.
'''
# 1. Without transformation
plt.hist(df['months_as_member'], bins=100, density=True, alpha=0.5)
# Probability Density Function (PDF)
plt.title('Distribution of the number of months as a member')
plt.xlabel('Months as member')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()

# 2. With log transformation
plt.hist(np.log(df['months_as_member']), bins=20, density=True, alpha=0.5)
# Probability Density Function (PDF)
plt.title('Distribution of the number of months as a member (log transformation)')
plt.xlabel('Months as member')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()

# BONUS
plt.hist(df['weight'], bins=100, density=True, alpha=0.5)
# Probability Density Function (PDF)
plt.title('Distribution of the weight of a member')
plt.xlabel('Member weight')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()

# 2. With log transformation
plt.hist(np.log(df['weight']), bins=30, density=True, alpha=0.5)
# Probability Density Function (PDF)
plt.title('Distribution of the weight of a member (log transformation)')
plt.xlabel('Member weight')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()

####################################################
'''
4. Describe the relationship between attendance and number of months as a member.
Your answer must include a visualization to demonstrate the relationship.
'''
sns.boxplot(data=df, x="attended", y="months_as_member")
plt.show()

# BONUS: relationship between attendance and weight of the member.
sns.boxplot(data=df, x="attended", y="weight")
plt.show()

df['months_as_member_log'] = np.log(df['months_as_member'])
sns.boxplot(data=df, x="attended", y="months_as_member_log")
plt.show()

####################################################
'''
5. The business wants to predict whether members will attend using the data provided.
State the type of machine learning problem that this is (regression/ classification/
clustering).
'''
# classification

# 1. import packages
from sklearn.linear_model import LogisticRegression  # Classifier model
from sklearn.ensemble import RandomForestClassifier  # Classifier model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# 2. last preprocessing
df = pd.get_dummies(df, columns=['day_of_week'], prefix=['day_of_week'], dtype=int)
df = pd.get_dummies(df, columns=['category'], prefix=['category'], dtype=int)
df['time_numerical'] = 1
df.loc[df['time'] == 'AM', 'time_numerical'] = 0
df['months_as_member_log'] = np.log(df['months_as_member'])

df['weight_log'] = np.log(df['weight'])

# DROP DUPLICATES
df = df[df['weight_log']<5]
df = df[df['months_as_member'] < 140]

print(df)

df_with_nans = df[df.isna().any(axis=1)]

# 3. Split separate df into X and y
X = df[['months_as_member', 'days_before', # 'weight',
       'day_of_week_Mon', 'day_of_week_Tue', 'day_of_week_Wed',
       'day_of_week_Thu', 'day_of_week_Fri', 'day_of_week_Sat',
       'day_of_week_Sun', 'day_of_week_unknown', 'category_Yoga',
       'category_Aqua', 'category_Strength', 'category_HIIT',
       'category_Cycling', 'time_numerical', 'months_as_member_log' ,'weight_log'
        ]]
print(X)
# Create a StandardScaler instance
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# Fit the scaler on the training data and transform both training and test data
# X_scaled = scaler.fit_transform(X)

y = df['attended']
print(y)

# 4. Split dataset into 70% training set and 30% test set
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Create a StandardScaler instance
scaler = StandardScaler()
# scaler = MinMaxScaler()
# Fit the scaler on the training data and transform both training and test data
X_train_scaled = X_train
X_test_scaled = X_test
"""
# columns_to_scale = ['months_as_member_log', 'weight']
columns_to_scale = ['weight']
X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train_scaled[columns_to_scale])
X_test_scaled[columns_to_scale] = scaler.fit_transform(X_test_scaled[columns_to_scale])
"""
####################################################

'''
6. Fit a baseline model to predict whether members will attend using the data provided.
You must include your code.
'''

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Define the hyperparameter grid you want to search over
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga']  # Algorithm to use for optimization
}
# Initialize the classifier model (e.g., Logistic Regression or Random Forest Classifier)
log_ = LogisticRegression(max_iter=1000)  # You can change this to RandomForestClassifier()
# Create a GridSearchCV object
log = GridSearchCV(estimator=log_, param_grid=param_grid, cv=kfold, scoring='precision')
# Fit (train) the model on the training data
log.fit(X_train, y_train)
## log.fit(X_train, y_train)
# Make predictions on the test data
y_pred_log = log.predict(X_test)
## y_pred_log = log.predict(X_test)

####################################################
'''
7. Fit a comparison model to predict whether members will attend using the data
provided. You must include your code.
'''

# Define the hyperparameter grid you want to search over
param_grid = {
    'n_estimators': [50, 100, 200],          # Number of trees in the forest
    'max_depth': [4, 5, 10, 20],        # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],       # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],         # Minimum number of samples required to be at a leaf node
}
# Initialize the classifier model (e.g., Logistic Regression or Random Forest Classifier)
rf_ = RandomForestClassifier()  # You can change this to RandomForestClassifier()
# Create a GridSearchCV object
rf = GridSearchCV(estimator=rf_, param_grid=param_grid, cv=kfold, scoring='precision')
# Fit (train) the model on the training data
rf.fit(X_train, y_train)
## rf.fit(X_train, y_train)
# Make predictions on the test data
y_pred_rf = rf.predict(X_test)
## y_pred_rf = rf.predict(X_test)

####################################################
'''
8. Explain why you chose the two models used in parts 6 and 7.
'''


####################################################
'''
9. Compare the performance of the two models used in parts 6 and 7, using any method
suitable. You must include your code.
'''

# The choice of the most appropriate metric depends on GoalZone's priorities and
# the consequences of making certain types of errors. For example, if GoalZone wants
# to avoid missing members who would attend (avoid false negatives), they may
# prioritize recall. If they want to minimize the risk of opening up spaces unnecessarily
# (minimize false positives), they may prioritize precision.

# Calculate classification performance metrics
accuracy_log = accuracy_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log) # 0.43, 0.42
precision_log = precision_score(y_test, y_pred_log)
confusion_log = confusion_matrix(y_test, y_pred_log)
classification_rep_log = classification_report(y_test, y_pred_log)
# Print the performance metrics
print("Accuracy:", accuracy_log)
print("Recall:", recall_log)
print("Precision:", precision_log)
print("\nConfusion Matrix:\n", confusion_log)
print("\nClassification Report:\n", classification_rep_log)

# Calculate classification performance metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf) # 0.44, 0.45
precision_rf = precision_score(y_test, y_pred_rf)
confusion_rf = confusion_matrix(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)
# Print the performance metrics
print("Accuracy:", accuracy_rf)
print("Recall:", recall_rf)
print("Precision:", precision_rf)
print("\nConfusion Matrix:\n", confusion_rf)
print("\nClassification Report:\n", classification_rep_rf)
####################################################
'''
10. Explain which model performs better and why.
'''
# the first one

####################################################
