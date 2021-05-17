#pip install fuzzy-c-means

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fcmeans import FCM

dataset = pd.read_csv('D://Visual Exercise//Python//New folder//Fuzzy-C-Means Clustering//Fuzzy-C-Means Clustering//Mall_Customers.csv')

#Getting data set
X = dataset.iloc[: , [3,4]].values
print(X)

# fit the fuzzy-c-means
fcm = FCM(n_clusters=3,max_iter=150,random_state=0)
fcm.fit(X)

y_pred = fcm.predict(X)

print(y_pred)

# outputs
#predict and labels are same
fcm_centers = fcm.centers
fcm_labels  = fcm.u.argmax(axis=1)

print(fcm_labels)

# Visualising the clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(fcm_centers[:,0],fcm_centers[:,1],s=300 , c='yellow',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()