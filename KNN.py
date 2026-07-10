import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Projet ML - k-NN Iris Classification", layout="wide")

# --- BARRE LATÉRALE DE NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez la section :", ["1. Présentation (PDF)", "2. Dashboard Interactive k-NN"])

# --- CONFIGURATION ESTHÉTIQUE ---
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# --- CHARGEMENT DU DATASET IRIS ---
@st.cache_data
def load_iris_data():
    iris = load_iris()
    return iris

# --- SECTION 1 : PRESENTATION PPT ---
if page == "1. Présentation (PDF)":
    st.title("📂 Présentation du Projet - Algorithme k-NN")
    st.write("Voici les diapositives de mon projet sur la classification avec les k plus proches voisins.")
    
   
    file_id = "12nC_LDt4i2wnjCmWoBl-OCCz95jBfT2P"  
    lien_drive_embed = f"https://drive.google.com/file/d/{file_id}/preview"
    
    st.components.v1.html(
        f'<iframe src="{lien_drive_embed}" style="width:100%; height:750px;" frameborder="0" allowfullscreen></iframe>',
        height=750
    )

# --- SECTION 2 : DASHBOARD INTERACTIVE k-NN ---
elif page == "2. Dashboard Interactive k-NN":
    st.title("💻 Classification Interactive d'Iris via k-NN")
    st.write("Modifiez l'hyperparamètre $k$ en direct pour analyser la géométrie et l'ajustement des frontières de décision.")

    # --- BARRE LATÉRALE DES PARAMÈTRES (INTERACTIF) ---
    st.sidebar.subheader("🎛️ Hyperparamètres du k-NN")
    k_value = st.sidebar.slider("Nombre de voisins (k / n_neighbors)", min_value=1, max_value=25, value=5, step=1)
    metric_choice = st.sidebar.selectbox("Métrique de Distance", ["minkowski", "euclidean", "manhattan"])

    # Chargement et préparation (Exactement kima f script dyalk)
    iris = load_iris_data()
    X = iris.data[:, :2]  # On garde 2 features (Sepal Length & Sepal Width) pour la 2D
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement en direct avec les sliders
    knn = KNeighborsClassifier(n_neighbors=k_value, metric=metric_choice)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred) * 100

    # --- PANNEAU DE STATISTIQUES ---
    col1, col2, col3 = st.columns(3)
    col1.metric(label="📊 Dataset", value=f"Iris ({len(X)} échantillons)")
    col2.metric(label="⚙️ Valeur de k sélectionnée", value=f"{k_value} voisins")
    col3.metric(label="🎯 Précision Globale (Accuracy)", value=f"{acc:.2f} %")

    st.markdown("---")

    # --- VISUALISATION 1 & 2 : GRAPHES ---
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("🗺️ Cartographie des Frontières de Décision (2D)")
        st.write("Ce graphe montre comment l'espace vectoriel est découpé selon la valeur de $k$.")
        
        with plt.style.context('default'):
            fig_boundary, ax_boundary = plt.subplots(figsize=(8, 6.5), facecolor='white')
            
            # Affichage des frontières de décision exactes
            DecisionBoundaryDisplay.from_estimator(
                knn, X, response_method="predict", ax=ax_boundary, 
                xlabel=iris.feature_names[0], ylabel=iris.feature_names[1],
                alpha=0.2, cmap=plt.cm.coolwarm
            )
            
            # Scatter plot des points réels
            scatter = ax_boundary.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm, s=50)
            
            # Légende personnalisée pour les classes de fleurs
            handles, _ = scatter.legend_elements()
            ax_boundary.legend(handles, iris.target_names, loc="upper right", title="Espèces d'Iris")
            ax_boundary.set_title(f"Frontières de décision pour k = {k_value}", fontsize=12, fontweight='bold')
            
            st.pyplot(fig_boundary)

    with col_right:
        st.subheader("📊 Matrice de Confusion de l'Iris")
        st.write("Analyse des erreurs de classification sur l'échantillon de test.")
        
        with plt.style.context('default'):
            fig_cm, ax_cm = plt.subplots(figsize=(6.5, 6), facecolor='white')
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=iris.target_names, yticklabels=iris.target_names, 
                        ax=ax_cm, cbar=False, square=True, annot_kws={"size": 14})
            
            ax_cm.set_title("Matrice de Confusion k-NN", fontsize=12, fontweight='bold', pad=10)
            ax_cm.set_ylabel('Vraie Espèce (Réelle)', fontsize=10)
            ax_cm.set_xlabel('Espèce Prédite', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_cm)

    st.markdown("---")
    
    # --- RAPPORT TEXTUEL DÉTAILLÉ ---
    st.subheader("📋 Rapport de Classification Complet")
    report_dict = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    st.dataframe(df_report.style.format(precision=3), use_container_width=True)
