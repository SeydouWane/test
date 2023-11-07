import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Chargement du fichier Excel
file = st.file_uploader("Importer un fichier Excel", type=["xlsx", "xls"])

# Charger le fichier dans un DataFrame si un fichier est sélectionné
if file:
    df = pd.read_excel(file)  

    # Affichage des premières lignes du fichier
    show_first_rows = st.checkbox("Afficher les premières lignes du fichier")
    if show_first_rows:
        st.write("Les premières lignes du fichier :")
        st.write(df.head())

    # Affichage du nombre de lignes et de colonnes du fichier
    show_shape = st.checkbox("Afficher le nombre de lignes et de colonnes du fichier")
    if show_shape:
        st.write("Le nombre de lignes et de colonnes :")
        st.write(df.shape)

    # Affichage des valeurs manquantes
    show_missing_values = st.checkbox("Afficher les valeurs manquantes")
    if show_missing_values:
        st.write("Valeurs manquantes :")
        st.write(df.isna().sum() / df.shape[0])

    # Affichage des statistiques descriptives du fichier
    show_descriptive_stats = st.checkbox("Afficher les statistiques descriptives")
    if show_descriptive_stats:
        st.write("Les statistiques descriptives :")
        st.write(df.describe())

    # Analyse de la matrice de corrélation
    show_correlation_matrix = st.checkbox("Afficher la matrice de corrélation")
    if show_correlation_matrix:
        st.write("La matrice de corrélation :")
        corr = df.corr()
        sns.heatmap(corr, annot=True)
        st.pyplot()

    # Exemple : sns.countplot(data=df, x='Gender', hue='Loan_Status')
    st.write("Analyse des variables catégorielles :")
    categorical_cols = st.multiselect("Sélectionnez les variables catégorielles à analyser", df.select_dtypes(include='object').columns.tolist())
    for col in categorical_cols:
        sns.countplot(data=df, x=col, hue='Loan_Status')
        plt.xticks(rotation=45)
        st.pyplot()
    # Analyse des variables numériques (ajouter vos propres analyses)
    st.write("Analyse des variables numériques :")
    numerical_cols = st.multiselect("Sélectionnez les variables numériques à analyser", df.select_dtypes(include=['int64', 'float64']).columns.tolist())
    for col in numerical_cols:
        sns.histplot(data=df, x=col, kde=True)  # Supprimez 'hue='Loan_Status''
        st.pyplot()

    # Pré-traitement des données
    # Afficher toutes les colonnes du DataFrame dans un multiselect
    selected_cols = st.multiselect("Sélectionnez les colonnes", df.columns)

    # Vérifier si des colonnes ont été sélectionnées
    if selected_cols:
        # Division du Dataset en données d'entraînement et de test
        x = df[selected_cols]
        y = df['Loan_Status']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        # Sélection des meilleures variables
        selector = SelectKBest(chi2, k=4)
        x_train_selected = selector.fit_transform(x_train, y_train)
        x_test_selected = selector.transform(x_test)

        # Entraînement du modèle Logistic Regression
        logistic_regr = LogisticRegression()
        logistic_regr.fit(x_train_selected, y_train)
        y_pred = logistic_regr.predict(x_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Score de précision du modèle : {accuracy}")

        # Matrice de confusion
        st.write("Matrice de confusion :")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
    else:
        st.write("Veuillez sélectionner au moins une colonne.")
