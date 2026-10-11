"""Pipeline ML : prédiction du churn client (Churn_Modelling.csv)."""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "Exited"
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]


def prepare_data(csv_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge le CSV, le prétraite et le sépare en train/test.

    Args:
        csv_path (str): chemin du fichier CSV.
        test_size (float): proportion du jeu de test.
        random_state (int): graine pour la reproductibilité.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    required = set(COLUMNS_TO_DROP + [TARGET, "Gender"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    # Encodage de Gender (Female/Male -> 0/1)
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    # Séparation features / cible, puis suppression des colonnes inutiles
    x = df.drop(columns=[TARGET] + COLUMNS_TO_DROP)
    y = df[TARGET]

    return train_test_split(x, y, test_size=test_size, random_state=random_state)



def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier.

    Args:
        x_train (DataFrame): features d'entraînement.
        y_train (Series): cible d'entraînement.
        n_estimators (int): nombre d'arbres.
        random_state (int): graine pour la reproductibilité.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue le modèle sur le jeu de test et affiche les métriques.

    Args:
        model: modèle entraîné.
        x_test (DataFrame): features de test.
        y_test (Series): cible de test.

    Returns:
        float: accuracy du modèle.
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    return accuracy


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle entraîné avec joblib.

    Args:
        model: modèle entraîné.
        path (str): chemin du fichier de sortie.

    Returns:
        str: chemin du fichier sauvegardé.
    """
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans : {path}")
    return path


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib.

    Args:
        path (str): chemin du fichier du modèle.

    Returns:
        Le modèle chargé.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)
