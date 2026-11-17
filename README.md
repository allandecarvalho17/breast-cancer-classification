# Classificação Breast Cancer

Esse projeto foi desenvolvido para a atividade de Data Science. A ideia é usar a base Breast Cancer, disponibilizada pela UCI, para treinar modelos de classificação e depois testar a previsão com os dados de uma nova paciente.

O objetivo é prever o atributo **Class**, que informa se houve recorrência ou não recorrência dos eventos registrados na base. Ou seja, o modelo não está tentando dizer se um tumor é benigno ou maligno; ele trabalha com as duas classes que existem nesse conjunto de dados.

## Arquivos do projeto

- `breast-cancer.csv`: base de dados utilizada no treinamento.
- `treinamento.py`: faz o carregamento, preparação dos dados, treinamento e comparação dos classificadores.
- `inferencia.py`: carrega o modelo salvo e faz a previsão para uma nova paciente.
- `requirements.txt`: bibliotecas necessárias para executar o projeto.

Depois de executar o treinamento, também serão criados:
- `breast_cancer_modelo.joblib`: modelo escolhido, junto com as etapas de preparação dos dados.
- `resultados_treinamento.json`: resumo das métricas e dos parâmetros encontrados.

## Como foi feito

Primeiro, a base é carregada e os valores `?` são tratados como dados ausentes. Depois, as colunas categóricas passam por preenchimento dos valores ausentes e codificação. A coluna `deg-malig` é tratada como numérica, com preenchimento pela mediana e normalização usando MinMaxScaler.

Foram utilizados dois classificadores para comparação: **Decision Tree** e **Random Forest**. A busca dos parâmetros é feita com validação cruzada estratificada. O melhor modelo é escolhido pela acurácia média na validação cruzada.

Também foi separado um conjunto de teste, que não participa do treinamento nem da escolha dos parâmetros. No final, o modelo escolhido é avaliado nesse conjunto, e são exibidos a acurácia, o relatório de classificação e a matriz de confusão.

O pré-processamento fica dentro do próprio pipeline do modelo. Assim, na inferência, os dados novos passam pelas mesmas etapas usadas durante o treinamento, sem precisar montar manualmente uma lista de colunas codificadas.

## Como executar

É necessário ter Python instalado. Abra o terminal na pasta do projeto e instale as bibliotecas:

```bash
pip install -r requirements.txt
```

Depois, execute o treinamento:

```bash
python treinamento.py
```

Quando o treinamento terminar, o arquivo `breast_cancer_modelo.joblib` será criado na pasta. Em seguida, execute:

```bash
python inferencia.py
```

Para testar outra paciente, basta alterar os valores do dicionário `nova_paciente`, no final do arquivo `inferencia.py`, respeitando os valores e formatos das categorias da base.

## Sobre a base

A base utilizada é a Breast Cancer, do UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/14/breast+cancer

Ela possui 286 registros, nove atributos de entrada e a coluna `Class`, que é o atributo que queremos prever.

## Observação

Esse projeto tem finalidade acadêmica e de estudo de classificação. O resultado do modelo não deve ser usado para diagnóstico, decisão clínica ou tratamento de pacientes.
