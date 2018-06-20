import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#cat data
data = np.array([['','tail_length','height'],
                ['0',30,25],
                ['1',15,20],
                ['2',31,24],
                ['3',17,21],
                ['4',29,24]])

df = pd.DataFrame(data=data[1:,1:],
                index=data[1:,0],
                columns=data[0,1:])

f1 = df['tail_length'].values
f2 = df['height'].values

X=np.matrix(zip(f1,f2))

print(X)

kmeans = KMeans(n_clusters=2).fit(X)
print(kmeans.labels_) # print out which cluster each cat belongs
print(kmeans.predict([[29, 26], [14, 19]])) # predict which cluster new cats belongs
print(kmeans.cluster_centers_) # where are the cluster centers
