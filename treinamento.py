"""
Treinamento dos classificadores para a base Breast Cancer (UCI).

O script compara Decision Tree e Random Forest usando validação cruzada.
O melhor classificador é escolhido pela acurácia média na validação cruzada
e, depois, avaliado em uma parte dos dados que ficou separada desde o início.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_BASE = PASTA_PROJETO / "breast-cancer.csv"
ARQUIVO_MODELO = PASTA_PROJETO / "breast_cancer_modelo.joblib"
ARQUIVO_RESULTADOS = PASTA_PROJETO / "resultados_treinamento.json"

COLUNAS_CATEGORICAS = [
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "breast",
    "breast-quad",
    "irradiat",
]
COLUNAS_NUMERICAS = ["deg-malig"]
COLUNA_ALVO = "Class"


def carregar_dados():
    """Carrega a base e converte '?' em valores ausentes."""
    if not ARQUIVO_BASE.exists():
        raise FileNotFoundError(
            "Não encontrei breast-cancer.csv. "
            "Deixe o arquivo CSV na mesma pasta deste script."
        )

    dados = pd.read_csv(ARQUIVO_BASE)
    dados = dados.replace("?", np.nan)

    colunas_esperadas = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS + [COLUNA_ALVO]
    faltantes = [coluna for coluna in colunas_esperadas if coluna not in dados.columns]
    if faltantes:
        raise ValueError(f"Colunas não encontradas na base: {faltantes}")

    dados = dados.dropna(subset=[COLUNA_ALVO])
    return dados


def criar_preprocessador():
    """Prepara os dados sem aprender informações fora de cada treino."""
    tratamento_categorico = Pipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="most_frequent")),
            ("codificacao", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    tratamento_numerico = Pipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="median")),
            ("normalizacao", MinMaxScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categoricas", tratamento_categorico, COLUNAS_CATEGORICAS),
            ("numericas", tratamento_numerico, COLUNAS_NUMERICAS),
        ]
    )


def main():
    dados = carregar_dados()

    X = dados[COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS]
    y = dados[COLUNA_ALVO]

    # Mantém a proporção das classes na separação treino/teste.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    modelos = {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "classificador__criterion": ["gini", "entropy"],
                "classificador__max_depth": [None, 3, 5, 10],
                "classificador__min_samples_split": [2, 5, 10],
                "classificador__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "classificador__n_estimators": [100, 200],
                "classificador__criterion": ["gini", "entropy"],
                "classificador__max_depth": [None, 5, 10],
                "classificador__min_samples_split": [2, 5],
                "classificador__class_weight": [None, "balanced"],
            },
        ),
    }

    resultados = {}
    melhor_nome = None
    melhor_busca = None
    melhor_acuracia_cv = -1

    print("\nIniciando treinamento e comparação...\n")

    for nome, (classificador, parametros) in modelos.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessador", criar_preprocessador()),
                ("classificador", classificador),
            ]
        )

        busca = GridSearchCV(
            estimator=pipeline,
            param_grid=parametros,
            scoring="accuracy",
            cv=validacao,
            n_jobs=-1,
            refit=True,
        )
        busca.fit(X_treino, y_treino)

        resultados[nome] = {
            "acuracia_media_validacao_cruzada": round(float(busca.best_score_), 4),
            "melhores_parametros": busca.best_params_,
        }

        print(f"{nome}: acurácia média na validação cruzada = {busca.best_score_:.4f}")
        print(f"Melhores parâmetros: {busca.best_params_}\n")

        if busca.best_score_ > melhor_acuracia_cv:
            melhor_acuracia_cv = busca.best_score_
            melhor_nome = nome
            melhor_busca = busca

    melhor_modelo = melhor_busca.best_estimator_
    previsoes = melhor_modelo.predict(X_teste)
    acuracia_teste = accuracy_score(y_teste, previsoes)

    print("=" * 60)
    print(f"Melhor classificador: {melhor_nome}")
    print(f"Acurácia no conjunto de teste: {acuracia_teste:.4f}")
    print("\nRelatório de classificação:")
    print(classification_report(y_teste, previsoes, zero_division=0))
    print("Matriz de confusão:")
    print(confusion_matrix(y_teste, previsoes))

    # O pipeline salvo já contém o pré-processamento e o classificador.
    joblib.dump(melhor_modelo, ARQUIVO_MODELO)

    resultados["melhor_modelo"] = melhor_nome
    resultados["acuracia_teste"] = round(float(acuracia_teste), 4)
    resultados["quantidade_registros"] = int(len(dados))
    resultados["distribuicao_classes"] = {
        str(chave): int(valor) for chave, valor in y.value_counts().items()
    }

    with open(ARQUIVO_RESULTADOS, "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, ensure_ascii=False, indent=4, default=str)

    print(f"\nModelo salvo em: {ARQUIVO_MODELO.name}")
    print(f"Resultados salvos em: {ARQUIVO_RESULTADOS.name}")


if __name__ == "__main__":
    main()
