"""Pipeline Prefect : orchestration des étapes du projet ML (churn)."""

import argparse
import subprocess
import sys

from prefect import flow, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"


# ---------- TASKS ----------
@task(name="install_dependencies")
def install_dependencies_task():
    """Installe les dépendances listées dans requirements.txt."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )


@task(name="prepare_data")
def prepare_data_task(csv_path=DATA_PATH):
    """Charge et prétraite les données."""
    return prepare_data(csv_path)


@task(name="train_model")
def train_model_task(x_train, y_train):
    """Entraîne le modèle."""
    return train_model(x_train, y_train)


@task(name="save_model")
def save_model_task(model, path=MODEL_PATH):
    """Sauvegarde le modèle entraîné."""
    return save_model(model, path)


@task(name="load_model")
def load_model_task(path=MODEL_PATH):
    """Charge le modèle sauvegardé."""
    return load_model(path)


@task(name="evaluate_model")
def evaluate_model_task(model, x_test, y_test):
    """Évalue le modèle sur le jeu de test."""
    return evaluate_model(model, x_test, y_test)


# Le code qualité/sécurité ne cible que les fichiers du projet
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


@task(name="format_code")
def format_code_task():
    """Formate le code avec black."""
    subprocess.run([sys.executable, "-m", "black", *FILES], check=True)


@task(name="code_quality")
def code_quality_task():
    """Vérifie la qualité du code avec flake8."""
    subprocess.run(
        [sys.executable, "-m", "flake8", "--max-line-length", "88", *FILES],
        check=True,
    )


@task(name="code_security")
def code_security_task():
    """Analyse la sécurité du code avec bandit (sévérité moyenne et plus)."""
    subprocess.run([sys.executable, "-m", "bandit", "-ll", *FILES], check=True)


@task(name="unit_tests")
def unit_tests_task():
    """Exécute les tests unitaires avec pytest."""
    subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "test_model_pipeline.py"],
        check=True,
    )


# ---------- FLOWS ----------
@flow(name="code")
def code_flow():
    """Installation, formatage, qualité, sécurité et tests unitaires."""
    install_dependencies_task()
    format_code_task()
    code_quality_task()
    code_security_task()
    unit_tests_task()


@flow(name="all")
def all_flow():
    """Étapes code + prepare, train, save, evaluate."""
    install_dependencies_task()
    format_code_task()
    code_quality_task()
    code_security_task()
    unit_tests_task()
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)
    evaluate_model_task(model, x_test, y_test)


@flow(name="train")
def train_flow():
    """Préparation des données et entraînement du modèle."""
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow():
    """Chargement du modèle et évaluation."""
    _, x_test, _, y_test = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, x_test, y_test)


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect - Churn")
    parser.add_argument(
        "--flow",
        choices=["all", "train", "entrainement", "evaluate", "code"],
        required=True,
        help="flow à lancer",
    )
    args = parser.parse_args()

    if args.flow == "all":
        all_flow()
    elif args.flow in ("train", "entrainement"):
        train_flow()
    elif args.flow == "code":
        code_flow()
    else:
        evaluate_flow()
