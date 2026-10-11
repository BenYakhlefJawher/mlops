"""Point d'entrée : exécute les étapes du pipeline via des arguments CLI."""

import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn")
    parser.add_argument("--data", default="Churn_Modelling.csv",
                        help="chemin du fichier CSV")
    parser.add_argument("--model", default="classifier.joblib",
                        help="chemin du modèle sauvegardé")
    parser.add_argument("--prepare", action="store_true",
                        help="charger et prétraiter les données")
    parser.add_argument("--train", action="store_true",
                        help="entraîner puis sauvegarder le modèle")
    parser.add_argument("--evaluate", action="store_true",
                        help="charger le modèle sauvegardé et l'évaluer")
    args = parser.parse_args()

    if not (args.prepare or args.train or args.evaluate):
        parser.print_help()
        return

    x_train, x_test, y_train, y_test = prepare_data(args.data)
    if args.prepare:
        print(f"Données prêtes : train={x_train.shape}, test={x_test.shape}")

    if args.train:
        model = train_model(x_train, y_train)
        save_model(model, args.model)

    if args.evaluate:
        model = load_model(args.model)
        evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
