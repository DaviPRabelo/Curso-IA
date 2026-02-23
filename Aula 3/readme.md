# Aula 3: Fundamentos de Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvarosamp/Curso-IA/blob/main/Aula%203/Aula_3.ipynb)
![Licença](https://img.shields.io/badge/licença-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-blue)

Bem-vindo à **Aula 3** do curso de Inteligência Artificial! Esta aula é dedicada aos fundamentos do Aprendizado de Máquina (Machine Learning) e foi desenvolvida para alunos que já possuem conhecimentos básicos de Python e das bibliotecas NumPy, Pandas e Matplotlib (conteúdo das Aulas 1 e 2).

> 📚 **Carga horária estimada:** 8 horas  
> 🎯 **Objetivo:** Capacitar o aluno a entender, implementar e avaliar modelos clássicos de Machine Learning utilizando a biblioteca scikit-learn.

---

## 📌 Conteúdo Programático

O material aborda os tópicos essenciais de Machine Learning de forma prática e didática:

- **Introdução ao Machine Learning**
  - Tipos de aprendizado (supervisionado, não supervisionado, reforço)
  - Conceitos fundamentais e o trade-off viés-variância
  
- **Preparação de Dados**
  - Análise exploratória de dados (EDA)
  - Tratamento de valores ausentes
  - Codificação de variáveis categóricas
  - Feature engineering e seleção de features
  - Escalonamento de dados
  
- **Modelagem e Validação**
  - Divisão treino/validação/teste
  - Validação cruzada (K-Fold, StratifiedKFold)
  - Modelos clássicos: Regressão Linear, KNN, Árvores de Decisão, SVM, Naive Bayes, Random Forest
  - Avaliação de modelos com métricas apropriadas
  
- **Otimização e Interpretabilidade**
  - Grid Search e Random Search para otimização de hiperparâmetros
  - Importância de features
  - Interpretação dos resultados

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

- **Python 3.8+**
- **NumPy** – operações matemáticas e arrays
- **Pandas** – manipulação e análise de dados
- **Matplotlib** e **Seaborn** – visualização de dados
- **Scikit-learn** – modelos, pré-processamento, métricas, validação
- **Jupyter Notebook** – ambiente interativo

---

## 🚀 Como Utilizar Este Material

### Opção 1: Executar no Google Colab (recomendado)

1. Clique no botão **"Open in Colab"** no início deste README.
2. Faça uma cópia para o seu Drive (Arquivo > Salvar uma cópia no Drive).
3. Execute as células sequencialmente (Shift+Enter).
4. Não há necessidade de instalar dependências adicionais no Colab (numpy, pandas, scikit-learn já vêm pré-instalados).

### Opção 2: Executar localmente (Jupyter Notebook)

1. Clone este repositório:
   ```bash
   git clone https://github.com/alvarosamp/Curso-IA.git
   ```

2. Acesse a pasta da aula:
   ```bash
   cd Curso-IA/Aula\ 3
   ```

3. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

4. Instale as dependências:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

5. Inicie o Jupyter Notebook:
   ```bash
   jupyter notebook Aula_3.ipynb
   ```

---

## 📊 Datasets Utilizados

Durante a aula, trabalhamos com datasets clássicos:

- **Iris** – classificação de espécies de flores (scikit-learn)
- **California Housing** – regressão de preços de imóveis (scikit-learn)
- **Digits** – reconhecimento de dígitos manuscritos (scikit-learn)

Todos os datasets são carregados diretamente via código, sem necessidade de download manual.

---

## 📝 Exercícios Propostos

Ao término da aula, pratique com os seguintes exercícios:

1. **EDA Aprofundada** – Explore um novo dataset (ex: Wine, Breast Cancer)
2. **Pipeline Completo** – Pré-processamento + modelagem + avaliação
3. **Comparação de Modelos** – Treine múltiplos modelos e compare desempenho
4. **Otimização** – Use Grid Search para melhorar seu melhor modelo

---

## 📚 Referências Úteis

- [Documentação oficial do scikit-learn](https://scikit-learn.org/)
- [Documentação do Pandas](https://pandas.pydata.org/docs/)
- [Documentação do NumPy](https://numpy.org/doc/)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)
- GÉRON, Aurélien. *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn & TensorFlow*. O'Reilly, 2019.
- JAMES, Gareth et al. *An Introduction to Statistical Learning*. Springer, 2013.

---

## 👨‍🏫 Autor

**Alvaro Sampaio**  
[![GitHub](https://img.shields.io/badge/GitHub-alvarosamp-181717?style=flat&logo=github)](https://github.com/alvarosamp)


---

## 📄 Licença

Este projeto está licenciado sob a licença MIT.

---

**Bons estudos!** 🚀
