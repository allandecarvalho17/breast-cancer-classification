"""
Inferência para uma nova paciente.

Edite os valores do dicionário nova_paciente com as características da paciente.
O modelo salvo já contém as etapas de tratamento, codificação e normalização.
"""

from pathlib import Path

import joblib
import pandas as pd
import numpy as np


PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_MODELO = PASTA_PROJETO / "breast_cancer_modelo.joblib"

COLUNAS_ESPERADAS = [
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat",
]


def prever_paciente(dados_paciente):
    """Recebe um dicionário com os dados de uma paciente e retorna a classe prevista."""
    if not ARQUIVO_MODELO.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Execute primeiro o arquivo treinamento.py."
        )

    faltantes = [coluna for coluna in COLUNAS_ESPERADAS if coluna not in dados_paciente]
    if faltantes:
        raise ValueError(f"Informe os seguintes campos: {faltantes}")

    entrada = pd.DataFrame([dados_paciente], columns=COLUNAS_ESPERADAS)
    entrada = entrada.replace("?", np.nan)

    modelo = joblib.load(ARQUIVO_MODELO)
    previsao = modelo.predict(entrada)[0]

    return previsao


if __name__ == "__main__":
    # Exemplo ilustrativo. Altere os valores para testar outra paciente.
    nova_paciente = {
        "age": "50-59",
        "menopause": "ge40",
        "tumor-size": "15-19",
        "inv-nodes": "0-2",
        "node-caps": "no",
        "deg-malig": 1,
        "breast": "right",
        "breast-quad": "central",
        "irradiat": "no",
    }

    diagnostico = prever_paciente(nova_paciente)
    print("Classificação prevista:", diagnostico)

    if diagnostico == "recurrence-events":
        print("Indicação do modelo: recorrência.")
    elif diagnostico == "no-recurrence-events":
        print("Indicação do modelo: sem recorrência.")
