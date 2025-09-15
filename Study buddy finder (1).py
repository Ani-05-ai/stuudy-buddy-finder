#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd


# In[49]:


data = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")


# In[51]:


data.head()


# In[53]:


data.isnull()


# In[55]:


data.isnull().sum()


# In[59]:


data.isnull().sum(), data.shape, data.info()


# In[57]:


data.isnull().sum(), data.shape, data.info()


# In[61]:


data = data.dropna()


# In[89]:


print(data.columns.tolist())


# In[91]:


data.columns = data.columns.str.strip().str.replace('\n', ' ', regex=True)


# In[95]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Step 1: Clean column names (remove spaces, newlines, tabs at start/end)
data.columns = data.columns.str.strip().str.replace('\n', ' ', regex=True)

# Step 2: Define useful columns (excluded "Daily Social Media Minutes...")
useful_cols = [
    'Age',
    'Books read past year Provide in integer value between (0-50)',
    'Introversion extraversion',
    'Club top1'
]

# Step 3: Split into numeric and categorical
numeric_cols = [
    'Introversion extraversion',
]

categorical_cols = [
    'Age',
    'Books read past year Provide in integer value between (0-50)',
    'Club top1'
]

# Step 4: Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Step 5: Apply preprocessing
X = preprocessor.fit_transform(data[useful_cols])

print("✅ Preprocessing successful. Final shape:", X.shape)


# In[107]:


X.shape


# In[ ]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

distortions = []
silhouettes = []
K = range(2, 11)  # test 2–10 clusters

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    distortions.append(kmeans.inertia_)  # Elbow method
    silhouettes.append(silhouette_score(X, labels))  # Cluster quality

# Plot elbow method
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Plot silhouette score
plt.plot(K, silhouettes, 'ro-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores')
plt.show()


# In[115]:


from sklearn.cluster import KMeans


# In[121]:


kmeans_model = KMeans(n_clusters=9)
kmeans_model.fit(X)


# In[119]:


import numpy as np
kmeans_model.labels_ , kmeans_model.cluster_centers_ , kmeans_model.inertia_,np.unique(kmeans_model.labels_)


# In[123]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Range of clusters to test
ks = range(2, 15)   # test from 2 to 14 clusters
inertias = []

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot the elbow method
plt.plot(ks, inertias, 'bo-')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method to Find Optimal k")
plt.show()


# In[136]:


from sklearn.metrics import silhouette_score

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    print(f"k={k}, Silhouette Score={sil_score:.4f}")


# In[138]:


final_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = final_kmeans.fit_predict(X)

print("Cluster sizes:", np.bincount(labels))


# In[140]:


print("Final inertia:", final_kmeans.inertia_)


# In[142]:


from sklearn.metrics import silhouette_score
sil = silhouette_score(X, labels)
print("Final silhouette score:", sil)


# In[125]:


import matplotlib.pyplot as plt
plt.plot(ks, inertias)


# In[144]:


import matplotlib.pyplot as plt
plt.bar(range(2), np.bincount(labels))
plt.xlabel("Cluster")
plt.ylabel("Number of students")
plt.title("Cluster distribution")
plt.show()


# In[146]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis")
plt.title("Clusters (PCA projection)")
plt.show()


# In[127]:


np.unique(kmeans_model.labels_)


# In[129]:


from sklearn.decomposition import PCA

# Reduce X to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap='tab20', s=50, alpha=0.6)
plt.title("Clusters Visualization (2D PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()


# In[148]:


np.bincount(labels) # in 0th cluster we we hhave 7 points,1st cluster we have 3 points and so on  


# In[150]:


from sklearn.metrics import silhouette_score

score = silhouette_score(X_pca, labels)
print("Silhouette Score:", score)


# In[154]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)   # reduce to 2D
X_pca = pca.fit_transform(X)



print(X, X.shape)
print(X_pca, X_pca.shape)
# Variance explained by each component
explained_var = pca.explained_variance_ratio_

print("Explained variance ratio for each component:")
for i, var in enumerate(explained_var, start=1):
    print(f"PC{i}: {var*100:.2f}%")

print(f"Total information retained: {explained_var.sum()*100:.2f}%")

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c='blue', alpha=0.6, edgecolor='k')
plt.xlabel(f"PC1 ({explained_var[0]*100:.2f}% variance)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.2f}% variance)")
plt.title("2D PCA Projection with Explained Variance")
plt.grid(True)
plt.show()


# In[156]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Apply PCA
pca = PCA(n_components=3)   # reduce to 3D
X_pca = pca.fit_transform(X)

# Variance explained by each component
explained_var = pca.explained_variance_ratio_

print("Explained variance ratio for each component:")
for i, var in enumerate(explained_var, start=1):
    print(f"PC{i}: {var*100:.2f}%")

print(f"Total information retained: {explained_var.sum()*100:.2f}%")

# 3D Scatter plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c='blue', alpha=0.6, edgecolor='k')

ax.set_xlabel(f"PC1 ({explained_var[0]*100:.2f}% variance)")
ax.set_ylabel(f"PC2 ({explained_var[1]*100:.2f}% variance)")
ax.set_zlabel(f"PC3 ({explained_var[2]*100:.2f}% variance)")
ax.set_title("3D PCA Projection with Explained Variance")

plt.show()


# In[158]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# --- PCA to 2D ---
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X)

# 2D Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca2[:,0], X_pca2[:,1], c=kmeans_model.labels_, cmap="tab20", alpha=0.7, edgecolor='k')
plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("2D PCA Projection with Cluster Coloring")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.show()

# --- PCA to 3D ---
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X)

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
scatter3d = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2],
                       c=kmeans_model.labels_, cmap="tab20", alpha=0.7, edgecolor='k')

ax.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]*100:.2f}%)")
ax.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]*100:.2f}%)")
ax.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2]*100:.2f}%)")
ax.set_title("3D PCA Projection with Cluster Coloring")

fig.colorbar(scatter3d, ax=ax, label="Cluster")
plt.show()

