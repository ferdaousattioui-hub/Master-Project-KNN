#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Chargement et préparation
iris = load_iris()
X = iris.data[:, :2] # On garde 2 features pour la visualisation 2D
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Entraînement
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 3. Affichage des métriques textuelles
print(f"Précision globale : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nRapport de classification :\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 4. Visualisation des frontières de décision
fig, ax = plt.subplots(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(
    knn, X, response_method="predict", ax=ax, 
    xlabel=iris.feature_names[0], ylabel=iris.feature_names[1],
    alpha=0.3, cmap=plt.cm.coolwarm
)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
plt.title("Frontières de décision KNN (k=5)")
plt.show()

# 5. Visualisation de la matrice de confusion
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Matrice de Confusion")
plt.show()


# In[ ]:




