# coding: utf-8
"""
"""
import math
import statistics as stat
from enum import Enum
from datetime import datetime

from numpy import datetime64
import numpy as np


# def convertir(donnees, instance):
#     """ données peut etre  """
#     resultat = None
#     if instance == 'list':
#         if isinstance(donnees, list):
#             resultat = donnees
#         elif isinstance(donnees, np.ndarray):
#             resultat = 
    
#     elif instance == 'array':

#     elif instance == 'dataframe':


#     else:
#         raise ValueError(f"Instance de conversion non reconnu: {instance}")

#     if isinstance(donnees, np.ndarray):

#         if isinstance(data, np.ndarray):
#             converted = data
#         elif isinstance(data, pd.Series):
#             converted = data.values
#         elif isinstance(data, list):
#             converted = np.array(data)
#         elif isinstance(data, pd.DataFrame):
#             converted = data.as_matrix()



def sigmoide(x, valeur_sup=1.0, valeur_inf=.0, pente=1):
    """Personnalisable sigmoide.

    Arguments d'entrées:
        x (float)
        valeur_sup, valeur_inf, pente (float): paramètres
    
    Arguments de sorties:
        (float)
    """
    return valeur_inf + ( valeur_sup - valeur_inf ) / ( 1 + math.exp(- pente * x) )


def inverse_sigmoide(x, valeur_sup=1.0, valeur_inf=.0, pente=1):
    """Personnalisable inverse sigmoide.

    Arguments d'entrées:
        x (float)
        valeur_sup, valeur_inf, pente (float): paramètres
    
    Arguments de sorties:
        (float)
    """
    return math.log( ( x - valeur_inf ) / ( valeur_sup - valeur_inf) ) / pente