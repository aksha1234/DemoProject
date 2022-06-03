import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

df = sns.load_dataset('penguins')
print(df)

plt.figure(figsize=(12, 9))
sns.scatterplot(data=df, x='bill_length_mm', y='flipper_length_mm', hue='species')
# plt.show()

# We will be working on the EDA of this dataset

print(df.describe(include='object'))

# the above syntax provide the description of the numerical columns in our dataset
# selecting only the numerical values
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()
# let us find whether our dataset has some null figures in it
print(df.isnull().sum())

# Now to impute the value which are missing lets use the sklearn libraries
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df[num_cols+cat_cols] = imputer.fit_transform(df[num_cols+cat_cols])
print(df.isnull().sum())

print(df)




