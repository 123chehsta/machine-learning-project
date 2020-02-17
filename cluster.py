import pandas as pd
dataset= pd.read_csv("studentclusters.csv")
x=dataset.copy()

x.plot.scatter(x='marks',y='shours')
#visualise the data
#noramailze the data

from sklearn.preprocessing import minmax_scale
x_scaled= minmax_scale(x)

#import kmeans
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=5,random_state=1234)
kmeans.fit(x_scaled)
labels= kmeans.labels_

# visualise the clusters
labels=pd.DataFrame(labels)
df=pd.concat([x,labels],axis=1)
df=df.rename(columns={0:'Labels'})
df.plot.scatter(x='marks',y='shours',c='Labels',colormap='Set1')