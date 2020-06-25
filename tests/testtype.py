# coding: utf-8
"""
Test pour le module dsfinesseur.type
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime as dt
from dsfinesseur import type as ds_type # code from module you're testing


class TypageDateTestCase(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        datetime_object = dt.date(2020, 8, 12)
        self.datetime_object = datetime_object
        self.datetime_str_fr = "Lundi 13 Mars"
        self.datetime_str_eng = "Tuesday 23 June"
        self.datetime_pandas = pd.to_datetime([datetime_object])
        self.datetime_numpy = np.datetime64('2005-02-25')
        self.fake_date = "hjfbkoiv"

    def tearDown(self):
        """Call after every test case."""
        pass

    def testEst_numpy_date(self):
        """Test est_numpy_date"""
        assert est_numpy_date(self.datetime_numpy) == True, "Not recognized as a date"
        assert est_numpy_date(self.datetime_str_fr) == False, "Recognized as a date"

    def testEst_string_date(self):
        """Test est_string_date"""
        assert est_string_date(self.datetime_str_fr, fuzzy=False, language='eng', kwargs=None) == True, "Not recognized as a date"
        assert est_string_date(self.datetime_str_eng, fuzzy=False, language='eng', kwargs=None) == True, "Not recognized as a date"
        assert est_string_date(self.datetime_numpy, fuzzy=False, language='eng', kwargs=None) == False, "Recognized as a date"

    def testEst_date(self):
        """Test est_date"""
        assert est_date(self.datetime_str_fr) == True, "Not recognized as a date"
        assert est_date(self.datetime_object) == True, "Not recognized as a date"
        assert est_date(self.fake_date) == False, "Recognized as a date"


class TypageDonnerTestCase(unittest.TestCase):

    def setUp(self):
        blah_blah_blah()

    def tearDown(self):
        blah_blah_blah()

    def testBlah(self):
        assert self.blahblah == "blah", "blah isn't blahing blahing correctly"



  


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


if __name__ == "__main__":
    unittest.main() # run all tests