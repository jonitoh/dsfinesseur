# coding: utf-8
"""
Fonctions et classes pour le typage des variables:
    - la classe `Typage` permet de rassembler les différents types possibles d'une variable.
    - la fonction `donner_un_type` permet de d'associer le type adéquate à une variable
"""
import dateutil
from dateutil.parser import parse, parserinfo 
from enum import Enum

from numpy import datetime64
import numpy as np


__all__ = [ 'Nature', 'donner_type' ]


SEUIL_CATEGORIE_PAR_DEFAUT = 5
"""A partir de combien de modalités au maximum peut-on dire qu'une variable est une classe ? """


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


class Type(EnumGestionnaire):
    """Enumération de tous les types de données."""
    CATEGORY = "category"
    BINARY = "binary"
    ORDERED_CATEGORY = "ordered category"
    UNORDERED_CATEGORY = "unordered category"
    NUMBER = "number"
    DATETIME = "datetime"

    def est_categorique(self):
        """Check if a feature is categorical.
        
        Arguments:
            - feature_type (Attributs)

        Returns:
            - (bool)
    
        """
        return self in (self.__class__.CATEGORY, self.__class__.BINARY, self.__class__.UNORDERED_CATEGORY, self.__class__.ORDERED_CATEGORY)


DATETIME_DTYPE = [np.dtype('datetime64'), np.dtype('datetime64[s]'), np.dtype('datetime64[ns]') ]


class CustomParserInfo(parserinfo):
    """ TODO: real custom of date please
    # type cas datetime
    # https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format
    # https://docs.python.org/fr/3.7/library/datetime.html
    # https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64  
    """
    
    MONTHS= [('Jan', 'January', 'Janvier'), ('Feb', 'February', 'Février'), ('Mar', 'March', 'Mars'), ('Apr', 'April', 'Avril'), ('May', 'May', 'Mai'), ('Jun', 'June', 'Juin'), ('Jul', 'July', 'Juillet'), ('Aug', 'August', 'Août'), ('Sep', 'Sept', 'September', 'Septembre'), ('Oct', 'October', 'Octobre'), ('Nov', 'November', 'Novembre'), ('Dec', 'December', 'Décembre')]

    WEEKDAYS= [('Mon', 'Monday', 'Lundi'), ('Tue', 'Tuesday', 'Mardi'), ('Wed', 'Wednesday', 'Mercredi'), ('Thu', 'Thursday', 'Jeudi'), ('Fri', 'Friday', 'Vendredi'), ('Sat', 'Saturday', 'Samedi'), ('Sun', 'Sunday', 'Dimanche')]
    
    @classmethod
    def from_language(cls, language=None, kwargs=None):
        return cls

  
def est_numpy_date(element):#, string, fuzzy=False, language='eng', kwargs=None):
    return any( y == date_type for date_type in DATETIME_DTYPE )


def est_string_date(string, fuzzy=False, language='eng', kwargs=None):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    parserinfo_instance = CustomParserInfo.from_language(language, kwargs)()
    try: 
        parse(string, fuzzy=fuzzy, parserinfo=parserinfo_instance)
        return True

    except:
        return False


def est_date(element, string_date_args=None):
    """
    string_date_args: dictionnaire dont les clefs sont: string, fuzzy, language, kwargs

    """
    type_datetime = any(isinstance(element, date_type) for date_type in DATETIME_DTYPE)
    type_series = isinstance(element, pd.Series)
    type_str = isinstance(element, str)
    type_list = isinstance(element, list)
    
    if type_datetime:
        return True
    
    if type_series:
        type_object = element.dtype == np.object
        if type_object:
            return any(map(est_numpy_date, element))
        return est_numpy_date(element)
    
    if type_str:
        if string_date_args is None:
            string_date_args = {}
        return est_string_date(string=element, **string_date_args)

    if type_list:
        return any(map(est_date, element))


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
    type_variable_datetime = variable.dtype == datetime64 or any(map(lambda s: est_date(s, fuzzy=False, language='eng', kwargs=None), variable))

    if nombre_de_valeurs_distinctes == 2:
        return Type.BINARY
    elif nombre_de_valeurs_distinctes < seuil_categorie or type_variable_categorie:
        return Type.CATEGORY
    elif type_variable_datetime:
        return Type.DATETIME
    else:
        return Type.NUMBER


def donner_type(variable, seuil_categorie=None, type_candidat=None):
    """Retourner le type d'une variable.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.
        type_candidat (str): possible type donnée par l'utilisateur à vérifier.

    Arguments de sorties:
        (Enum)
    """
    if seuil_categorie is None:
        seuil_categorie = SEUIL_CATEGORIE_PAR_DEFAUT
    
    type_candidat = Type.generate_name_from_value(type_candidat, has_default_value=False)
    if type_candidat is None:
        return trouver_type(variable, seuil_categorie)
    return type_candidat
