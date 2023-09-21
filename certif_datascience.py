
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

####################################################
'''
1. For every column in the data:
a. State whether the values match the description given in the table above.
b. State the number of missing values in the column.
c. Describe what you did to make values match the description if they did not
match.
'''
df = pd.read_csv('fitness_class_2212.csv')
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

df = df.dropna(subset='attended')

df = df.astype({
    'days_before': 'int32',
    'day_of_week': day,
    'time': time,
    'category': category})
print(df)

####################################################
'''
2. Create a visualization that shows how many bookings attended the class. Use the
visualization to:
a. State which category of the variable attended has the most observations
b. Explain whether the observations are balanced across categories of the
variable attended
'''
# A.
category_counts = df.groupby('category')['attended'].sum()
plt.bar(category_counts.index, category_counts.values)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Count of attended')
plt.show()

# B.
print(1)

####################################################
'''
3. Describe the distribution of the number of months as a member. Your answer must
include a visualization that shows the distribution.
'''


####################################################
'''
4. Describe the relationship between attendance and number of months as a member.
Your answer must include a visualization to demonstrate the relationship.
'''


####################################################
'''
5. The business wants to predict whether members will attend using the data provided.
State the type of machine learning problem that this is (regression/ classification/
clustering).
'''


####################################################
'''
6. Fit a baseline model to predict whether members will attend using the data provided.
You must include your code.
'''


####################################################
'''
7. Fit a comparison model to predict whether members will attend using the data
provided. You must include your code.
'''


####################################################
'''
8. Explain why you chose the two models used in parts 6 and 7.
'''


####################################################
'''
9. Compare the performance of the two models used in parts 6 and 7, using any method
suitable. You must include your code.
'''


####################################################
'''
10. Explain which model performs better and why.
'''


####################################################
