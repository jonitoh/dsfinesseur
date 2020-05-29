# coding: utf-8
"""
Fonctions et classes pour le typage des variables:
    - la classe Typage permet de rassembler les différents types possibles d'une variable.
    - la fonction permet de d'associer le type adéquate à une variable
"""
import math
import statistics as stat
from enum import Enum
from datetime import datetime

from numpy import datetime64
import numpy as np


__all__ = [ 'Nature', 'valeur_numpy_nulle', 'donner_type', 'sigmoide', 'inverse_sigmoide' ]


SEUIL_CATEGORIE_PAR_DEFAUT = 5
"""A partir de combien de modalités au maximum peut-on dire qu'une variable est une classe ? """




NUM_FEATURES = 6


DEFAULT_MODEL = 'LogisticRegression'


CATEGORY_LENGTH = 5


class EnumGestionnaire(Enum):
    """Méthodes à rajouter à la classe Enum."""

    @classmethod
    def has_name(cls, name):
        """Check if the name is in the enumeration. """
        return name in cls._member_names_

    @classmethod
    def has_value(cls, value):
        """Check if the value is in the enumeration. """
        return value in cls._value2member_map_

    @classmethod
    def generate_name_from_value(cls, element, has_default_value=True):
        """Ensure coherency in the enumeration """
        # default name
        name = cls._member_names_.keys()[0] if has_default_value else None
        if cls.has_name(element):
            name = element
        elif cls.has_value(element):
            name = cls(element).name
        else:
            # TODO: custom log
            print("Beware invalid role value. How is it possible?")
        return name


class Typage(EnumManager):
    """Available data types enumerations."""
    CATEGORY = "category"
    BINARY = "binary"
    ORDERED_CATEGORY = "ordered category"
    UNORDERED_CATEGORY = "unordered category"
    NUMBER = "number"
    DATETIME = "datetime"


def est_categorique(variable_type):
    """Check if a feature is categorical.
    
    Arguments:
        - feature_type (Attributs)

    Returns:
        - (bool)
   
    """
    return feature_type in (Attributs.CATEGORY, Attributs.BINARY)


### Native Python-based 
def valeur_numpy_nulle(variable):
    """ Pour savoir si une variable est NaN, il y a la fonction np.nan.
    Toutefois, elle ne fonctionne que sur des valeurs numériques.
    Cette fonction s'utilise sur tout type de fonction.
        
    Arguments d'entrées:
        variable (Python Object)
        
    Arguments de sorties:
        (bool)
    """
    return variable in [None, ""] or str(variable).lower() == "nan"


def trouver_type(variable, seuil_categorie):
    """Déterminer le type d'une variable par calcul.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.

    Arguments de sorties:
        (Enum)
    """
    nombre_de_valeurs_distinctes = len(set(variable))

    type_variable_categorie = variable.dtype == 'object'
    type_variable_datetime = variable.dtype == datetime64

    if nombre_de_valeurs_distinctes == 2:
        return Nature.BINARY
    elif nombre_de_valeurs_distinctes < seuil_categorie or type_variable_categorie:
        return Nature.CATEGORY
    elif type_variable_datetime:
        return Nature.DATETIME
    else:
        return Nature.NUMBER


def donner_type(variable, seuil_categorie=None, type_attitre=None):
    """Retourner le type d'une variable.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.
        type_attitre (str): possible type donnée par l'utilisateur à vérifier.

    Arguments de sorties:
        (Enum)
    """
    if seuil_categorie is None:
        seuil_categorie = SEUIL_CATEGORIE_PAR_DEFAUT
    
    type_attitre = Nature.generate_name_from_value(type_attitre, has_default_value=False)
    if type_attitre is None:
        return trouver_type(variable, seuil_categorie)
    return type_attitre


### NumPy-based 
def valeur_numpy_nulle(variable):
    """ Pour savoir si une variable est NaN, il y a la fonction np.nan.
    Toutefois, elle ne fonctionne que sur des valeurs numériques.
    Cette fonction s'utilise sur tout type de fonction.
        
    Arguments d'entrées:
        variable (Python Object)
        
    Arguments de sorties:
        (bool)
    """
    return variable in [None, ""] or str(variable).lower() == "nan"


def trouver_type(variable, seuil_categorie):
    """Déterminer le type d'une variable par calcul.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.

    Arguments de sorties:
        (Enum)
    """
    nombre_de_valeurs_distinctes = len(set(variable))

    type_variable_categorie = variable.dtype == 'object'
    type_variable_datetime = variable.dtype == datetime64

    if nombre_de_valeurs_distinctes == 2:
        return Nature.BINARY
    elif nombre_de_valeurs_distinctes < seuil_categorie or type_variable_categorie:
        return Nature.CATEGORY
    elif type_variable_datetime:
        return Nature.DATETIME
    else:
        return Nature.NUMBER


def donner_type(variable, seuil_categorie=None, type_attitre=None):
    """Retourner le type d'une variable.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.
        type_attitre (str): possible type donnée par l'utilisateur à vérifier.

    Arguments de sorties:
        (Enum)
    """
    if seuil_categorie is None:
        seuil_categorie = SEUIL_CATEGORIE_PAR_DEFAUT
    
    type_attitre = Nature.generate_name_from_value(type_attitre, has_default_value=False)
    if type_attitre is None:
        return trouver_type(variable, seuil_categorie)
    return type_attitre







class EnumManager(Enum):
    """Better handler of an enumeration."""
    
    @classmethod
    def has_name(cls, name):
        """Check if the name is in the enumeration. """
        return name in cls._member_names_
    
    @classmethod
    def has_value(cls, value):
        """Check if the value is in the enumeration. """
        return value in cls._value2member_map_

    @classmethod
    def generate_name_from_value(cls, element):
        """Ensure coherency in the enumeration """
        # default name
        name = cls._member_names_.keys()[0]
        if cls.has_name(element):
            name = element
        elif cls.has_value(element):
            name = cls(element).name
        else:
            print("invalid role value. How possible?")
        return name


class Attributs(EnumManager):
    """Enumerations of allowed data types."""
    BINARY = "Binary"
    CATEGORY = "Category"
    NUMBER = "Number"
    DATETIME = "Datetime"







# coding: utf-8
"""
Fonctions et classes pour le typage des variables:
    - la classe Typage permet de rassembler les différents types possibles d'une variable.
    - la fonction permet de d'associer le type adéquate à une variable
"""
import math
import statistics as stat
from enum import Enum
from datetime import datetime

from numpy import datetime64
import numpy as np


__all__ = [ 'Nature', 'valeur_numpy_nulle', 'donner_type', 'sigmoide', 'inverse_sigmoide' ]


SEUIL_CATEGORIE_PAR_DEFAUT = 5
"""A partir de combien de modalités au maximum peut-on dire qu'une variable est une classe ? """




NUM_FEATURES = 6


DEFAULT_MODEL = 'LogisticRegression'


CATEGORY_LENGTH = 5


class EnumGestionnaire(Enum):
    """Méthodes à rajouter à la classe Enum."""

    @classmethod
    def has_name(cls, name):
        """Check if the name is in the enumeration. """
        return name in cls._member_names_

    @classmethod
    def has_value(cls, value):
        """Check if the value is in the enumeration. """
        return value in cls._value2member_map_

    @classmethod
    def generate_name_from_value(cls, element, has_default_value=True):
        """Ensure coherency in the enumeration """
        # default name
        name = cls._member_names_.keys()[0] if has_default_value else None
        if cls.has_name(element):
            name = element
        elif cls.has_value(element):
            name = cls(element).name
        else:
            # TODO: custom log
            print("Beware invalid role value. How is it possible?")
        return name


class Typage(EnumManager):
    """Available data types enumerations."""
    CATEGORY = "category"
    BINARY = "binary"
    ORDERED_CATEGORY = "ordered category"
    UNORDERED_CATEGORY = "unordered category"
    NUMBER = "number"
    DATETIME = "datetime"


def est_categorique(variable_type):
    """Check if a feature is categorical.
    
    Arguments:
        - feature_type (Attributs)

    Returns:
        - (bool)
   
    """
    return feature_type in (Attributs.CATEGORY, Attributs.BINARY)


### Native Python-based 
def valeur_numpy_nulle(variable):
    """ Pour savoir si une variable est NaN, il y a la fonction np.nan.
    Toutefois, elle ne fonctionne que sur des valeurs numériques.
    Cette fonction s'utilise sur tout type de fonction.
        
    Arguments d'entrées:
        variable (Python Object)
        
    Arguments de sorties:
        (bool)
    """
    return variable in [None, ""] or str(variable).lower() == "nan"


def trouver_type(variable, seuil_categorie):
    """Déterminer le type d'une variable par calcul.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.

    Arguments de sorties:
        (Enum)
    """
    nombre_de_valeurs_distinctes = len(set(variable))

    type_variable_categorie = variable.dtype == 'object'
    type_variable_datetime = variable.dtype == datetime64

    if nombre_de_valeurs_distinctes == 2:
        return Nature.BINARY
    elif nombre_de_valeurs_distinctes < seuil_categorie or type_variable_categorie:
        return Nature.CATEGORY
    elif type_variable_datetime:
        return Nature.DATETIME
    else:
        return Nature.NUMBER


def donner_type(variable, seuil_categorie=None, type_attitre=None):
    """Retourner le type d'une variable.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.
        type_attitre (str): possible type donnée par l'utilisateur à vérifier.

    Arguments de sorties:
        (Enum)
    """
    if seuil_categorie is None:
        seuil_categorie = SEUIL_CATEGORIE_PAR_DEFAUT
    
    type_attitre = Nature.generate_name_from_value(type_attitre, has_default_value=False)
    if type_attitre is None:
        return trouver_type(variable, seuil_categorie)
    return type_attitre


### NumPy-based 
def valeur_numpy_nulle(variable):
    """ Pour savoir si une variable est NaN, il y a la fonction np.nan.
    Toutefois, elle ne fonctionne que sur des valeurs numériques.
    Cette fonction s'utilise sur tout type de fonction.
        
    Arguments d'entrées:
        variable (Python Object)
        
    Arguments de sorties:
        (bool)
    """
    return variable in [None, ""] or str(variable).lower() == "nan"


def trouver_type(variable, seuil_categorie):
    """Déterminer le type d'une variable par calcul.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.

    Arguments de sorties:
        (Enum)
    """
    nombre_de_valeurs_distinctes = len(set(variable))

    type_variable_categorie = variable.dtype == 'object'
    type_variable_datetime = variable.dtype == datetime64

    if nombre_de_valeurs_distinctes == 2:
        return Nature.BINARY
    elif nombre_de_valeurs_distinctes < seuil_categorie or type_variable_categorie:
        return Nature.CATEGORY
    elif type_variable_datetime:
        return Nature.DATETIME
    else:
        return Nature.NUMBER


def donner_type(variable, seuil_categorie=None, type_attitre=None):
    """Retourner le type d'une variable.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.
        type_attitre (str): possible type donnée par l'utilisateur à vérifier.

    Arguments de sorties:
        (Enum)
    """
    if seuil_categorie is None:
        seuil_categorie = SEUIL_CATEGORIE_PAR_DEFAUT
    
    type_attitre = Nature.generate_name_from_value(type_attitre, has_default_value=False)
    if type_attitre is None:
        return trouver_type(variable, seuil_categorie)
    return type_attitre







class EnumManager(Enum):
    """Better handler of an enumeration."""
    
    @classmethod
    def has_name(cls, name):
        """Check if the name is in the enumeration. """
        return name in cls._member_names_
    
    @classmethod
    def has_value(cls, value):
        """Check if the value is in the enumeration. """
        return value in cls._value2member_map_

    @classmethod
    def generate_name_from_value(cls, element):
        """Ensure coherency in the enumeration """
        # default name
        name = cls._member_names_.keys()[0]
        if cls.has_name(element):
            name = element
        elif cls.has_value(element):
            name = cls(element).name
        else:
            print("invalid role value. How possible?")
        return name


class Attributs(EnumManager):
    """Enumerations of allowed data types."""
    BINARY = "Binary"
    CATEGORY = "Category"
    NUMBER = "Number"
    DATETIME = "Datetime"







