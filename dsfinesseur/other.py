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