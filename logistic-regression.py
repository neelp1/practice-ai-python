import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('banking.csv', header=0)
data = data.dropna()

print(data.shape)
print(list(data.columns))
print(data.head())
print(data['education'].unique())

#group all basic into one column?
data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

print(data['education'].unique())

print(data['y'].value_counts())

sns.countplot(x='y', data=data, palette='hls')
plt.show()
plt.savefig('count_plot')

print(data.groupby('y').mean())
print("Grouped by Job:")
print(data.groupby('job').mean())
print("Grouped by Marital:")
print(data.groupby('marital').mean())
print("Grouped by Education:")
print(data.groupby('education').mean())
