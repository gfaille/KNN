import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

tableau = pd.read_csv("data.csv", sep=",")

# entrainner l'ia a predire 
X = tableau[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
Y = tableau['species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

knn = KNeighborsClassifier(n_neighbors= 10)
knn.fit(X_train, Y_train)
knn.score(X_test, Y_test)


def graph_sepal () :
    sns.scatterplot(tableau, x='sepal_length', y='sepal_width', hue='species')

def graph_petal():
    sns.scatterplot(tableau, x='petal_length', y='petal_width', hue='species')

    
def predic_fleur() :
    prediction_espece = knn.predict([[sepal_largeur, sepal_hauteur, petal_largeur, petal_hauteur]])
    st.success("l'espèce est : "+ prediction_espece[0])

st.title("Les espèce d'iris ")

# afficher slider
sepal_largeur = st.slider("larger du sepal", 0.0, 5.0)
sepal_hauteur = st.slider("hauteur du sepal", 0.0, 8.0)
petal_largeur = st.slider("largeur de la petal", 0.0, 5.0)
petal_hauteur = st.slider("hauteur de la petal", 0.0, 8.0)

if st.button("rechercher") :
    predic_fleur()
    elt_en_cours = {
        "sepal_length" : [sepal_largeur],
        "sepal_width" : [sepal_hauteur],
        "petal_length" : [petal_largeur],
        "petal_width" : [petal_hauteur],
        "species" : "recherche en cours"
    }
    elt_en_cours = pd.DataFrame(elt_en_cours)
    tableau_affiche = pd.concat([tableau, elt_en_cours])

    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.columns(2)
    st.write(tableau_affiche)
    graph1 = sns.FacetGrid(tableau_affiche, hue='species').map(plt.scatter, 'sepal_length', 'sepal_width').add_legend()
    graph2 = sns.FacetGrid(tableau_affiche, hue='species').map(plt.scatter, 'petal_length', 'petal_width').add_legend()
    col1.pyplot(graph1)
    col2.pyplot(graph2)

else :
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.columns(2)
    col1.pyplot(graph_sepal())
    col2.pyplot(graph_petal())
