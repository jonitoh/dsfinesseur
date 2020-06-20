"""Module centré sur le preprocessing d'un jeu de données.
Bien que basé sur un projet d'analyse de données spécifique (OpenFoodFacts),
le contenu a pour vocation d'être générique."""
import math
import time

import numpy as np
import pandas as pd


def est_une_valeur_manquante(variable, liste_personnalisee=None):
    """ Pour savoir si une variable est NaN, il y a la fonction np.nan.
    Toutefois, elle ne fonctionne que sur des valeurs numériques.
    Cette fonction s'utilise sur tout type de fonction.
        
    Arguments d'entrées:
        variable (Python Object)
        
    Arguments de sorties:
        (bool)
    """
    if liste_personnalisee is None:
        liste_personnalisee = []
    return (variable in liste_personnalisee) or (bool(variable) and variable != 0) or (variable != variable)


def calculer_train_test_split(y, train_rate=.70, valid_rate=None, classes=None):
    """ 
    """
    length = len(y)
    if valid_rate is None:
        probas = [train_rate, 1 - train_rate]
        split_into = ["train", "test"]
    else:
        probas = [train_rate, 1 - train_rate - valid_rate, valid_rate]
        split_into = ["train", "valid", "test"]

    
    if classes is None:
        return (np
                .random
                .choice(split_into, length, p=probas)
                )
    
    if len(classes) != len(set(y)):
        raise Error("len(classes) != len(set(y))")

    indexes = np.array(['unknown'] * length)
    for modality in classes:
        indexes_per_modality = [ i for i in range(length) if y[i] == modality ]
        indexes[ indexes_per_modality ] = np.random.choice(split_into, len(indexes_per_modality), p=probas)
    return indexes



    # coding: utf-8


def eliminer_colonne_vide(tableau, taux_de_vide_minimum=0.50):
    """Supprimer les colonnes avec un taux de remplissage inférieur à une référence donnée.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        taux_de_vide_minimum (float): au delà de ce taux, la colonne sera considérée vide
    
    Arguments de sortie:
        tableau_nettoye (pandas.DataFrame)
    """
    classement = pd.DataFrame()
    classement['colonne'] = list(tableau)
    classement['taux_de_vide'] = (
        classement['colonne']
        .apply(lambda colonne: tableau[colonne].isna().mean())
    )
    classement = classement.sort_values('taux_de_vide', ascending=False)
    print("clst df:\n ", classement)
    
    colonnes_a_eliminer = classement.loc[classement['taux_de_vide']>=taux_de_vide_minimum, 'colonne']
    tableau_nettoye = tableau.drop(columns=colonnes_a_eliminer)
    
    return tableau_nettoye


def eliminer_ligne_vide(tableau, taux=None):
    """Supprimer les lignes avec un taux de remplissage inférieur à une référence donnée.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        taux (float)
    
    Arguments de sorties:
        tableau_nettoye (pandas.DataFrame)
    """
    if not taux:
        taux = 0.50
    taux = len(tableau.columns) * taux // 1

    lignes_a_eliminer = tableau.isna().sum(1)
    tableau_nettoye = tableau[ lignes_a_eliminer>=taux ]

    return tableau_nettoye


def trouver_valeur_par_défaut(valeur_par_defaut, clef):
    """Permet de personnaliser une valeur par défaut suivant un axe."""
    if isinstance(valeur_par_defaut, dict):
        return valeur_par_defaut.get(clef, None)
    return valeur_par_defaut


def premiere_occurence(vecteur, valeur_par_defaut=None):
    """Wrapper de la méthode first_valid_index d'un Pandas.Series.
    
    Arguments d'entrée:
        vecteur (pandas.Series)
        valeur_par_defaut
    Arguments de sortie:
        (Python Object)
    """
    index = vecteur.first_valid_index()
    if index is None:
        return valeur_par_defaut
    return vecteur[index]


def plus_frequente_occurence(vecteur, valeur_par_defaut=None):
    """Wrapper de la méthode first_valid_index d'un Pandas.Series.
    
    Arguments d'entrée:
        vecteur (pandas.Series)
        valeur_par_defaut
    Arguments de sortie:
        (Python Object)
    """
    return vecteur.value_counts(ascending=False, dropna=False).iloc[0]


def calculer_imputation(tableau, strategie='première occurence', valeur_par_defaut=None):
    """Imputer suivant la strategie choisie.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        strategie (str):
            'première occurence' retourne la première valeur non nulle,
            'la plus fréquente' retourne la valeur la plus fréquente non nulle
    
    Arguments de sortie:
        tableau_nettoye (pandas.DataFrame)
    """
    fonction = None
    if strategie =="première occurence":
        fonction = premiere_occurence
    elif strategie=="la plus fréquente":
        fonction = plus_frequente_occurence
    else:
        raise ValueError(f"l'argument strategie ne peut prendre que les valeurs suivantes: 'première occurence' et 'la plus fréquente'. La valeur donnée ici est {strategie}")
    if tableau.empty:
        raise ValueError("l'argument tableau est vide.")
    tableau_impute = {index: fonction(tableau[index], trouver_valeur_par_défaut(valeur_par_defaut, index)) for index in tableau.columns}
    tableau_impute = pd.Series(tableau_impute)
    return tableau_impute


def eliminer_doublons(tableau, colonnes_ciblees=None, strategie='première occurence'):
    """Retourne un tableau avec des individus uniques
    basés sur les colonnes ciblées.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        colonnes_ciblees (list)
        strategie (str): méthode pour trouver l'unique individu
    
    Arguments de sortie:
        tableau_nettoye (pandas.DataFrame)
    """
    if colonnes_ciblees is None or not isinstance(colonnes_ciblees, list):
        if not isinstance(colonnes_ciblees, list):
            print(f"l'argument colonnes_ciblees n'est pas de type list mais {type(colonnes_ciblees)}")
            return tableau.drop_duplicates()
    methode = lambda tableau: calculer_imputation(tableau, strategie)
    return (
        tableau
        .groupby(colonnes_ciblees)
        .agg(methode)
        .reset_index(drop=True)
    )



# def encoding():

#     from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder

#     preprocessed_thoracic_data = thoracic_data.copy()
#     encodeurs = None

#     for _,serie in encoding_all.iterrows():
#         print("feature ", serie.features)
#         args = None
#         if serie.type == 'ordered category':
#             args = serie.value.split("<")
#             args = dict(zip(args, range(minimum_value, minimum_value + len(args))))
#         else:
#             args = serie.value.split(",")
#         encoder_args[ serie.features ] = args

#     #def preparer_data(data, binary_features, not_binary_features, numerical_features, encodeurs=None):
#     if encodeurs is None:
#         encodeurs = dict()

#     for feature, arg in encoder_args.items():
#         if isinstance(arg, list):
#             arg = np.array(arg).reshape(-1, 1)
#             encodeur = LabelEncoder() if feature != target else LabelBinarizer()
#             encodeur = encodeur.fit(arg)
#             encodeurs[ feature ] = encodeur
#             preprocessed_thoracic_data[ feature ] = encodeur.transform(preprocessed_thoracic_data[ feature ])    

#             if len(arg) == 2:#binary
#                 encodeur = LabelEncoder().fit(arg)
#                 encodeurs[ feature ] = encodeur
#                 preprocessed_thoracic_data[ feature ] = encodeur.transform(preprocessed_thoracic_data[ feature ])    

#             else:#unordered category
#                 encodeur = OneHotEncoder().fit(arg)
#                 encodeurs[ feature ] = encodeur
#                 encoded_features = [ category + '_' + category for category in encodeur.categories_[0] ]
                
#                 encoded_data = encodeur.transform(preprocessed_thoracic_data[ feature ].values.reshape(-1, 1)).toarray()
#                 encoded_data = pd.DataFrame(encoded_data, columns=encoded_features)
#                 # Etape de reset index nécessaire avant concaténation (pb très couteux en temps)            
#                 preprocessed_thoracic_data = (
#                     pd
#                     .concat([preprocessed_thoracic_data.reset_index(drop=True),
#                             encoded_data.reset_index(drop=True)],
#                             axis=1,
#                             join='outer'
#                         )
#                     .drop(columns=feature)
#                 )

        
#         elif isinstance(arg, dict):#ordered category
#             preprocessed_thoracic_data[ feature ] = preprocessed_thoracic_data[ feature ].apply(arg)
#             encodeurs[ feature ] = arg
            
#         else:
#             print(f"Oups! it should not happend! Why {feature} is not encoded.")
