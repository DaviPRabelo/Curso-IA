# Aula 4 — Introdução ao Deep Learning

Este material apresenta os fundamentos de Deep Learning com foco em redes neurais artificiais, desde o perceptron até os conceitos que sustentam modelos mais modernos.

## Objetivos da aula

Ao final desta aula, você deve ser capaz de:

- Entender o que é Deep Learning e por que ele se tornou tão relevante.
- Explicar o funcionamento do perceptron e sua limitação no problema XOR.
- Compreender a arquitetura e o treinamento de uma MLP (Multilayer Perceptron).
- Diferenciar funções de ativação, funções de perda e otimizadores.
- Identificar quando usar TensorFlow/Keras ou PyTorch em projetos práticos.

## Pré-requisitos

- Lógica de programação e Python básico.
- Noções de álgebra linear (vetores e matrizes).
- Conceitos introdutórios de Machine Learning.

## Conteúdo programático (ementa)

### 1. Introdução ao Deep Learning

- 1.1 O que é Deep Learning?
- 1.2 Por que Deep Learning agora?
- 1.3 Aplicações reais

### 2. Perceptron

- 2.1 Origem e inspiração biológica
- 2.2 Modelo matemático do Perceptron
- 2.3 Algoritmo de treinamento
- 2.4 Limitação do Perceptron: o problema XOR

### 3. Redes Neurais Artificiais (Multilayer Perceptron)

- 3.1 Arquitetura de uma MLP
- 3.2 Forward propagation
- 3.3 Backpropagation e o gradiente descendente
- 3.4 Inicialização de pesos

### 4. Funções de Ativação

- 4.1 Por que precisamos de não-linearidade?
- 4.2 Função Sigmoide
- 4.3 Tangente Hiperbólica (tanh)
- 4.4 ReLU (Rectified Linear Unit)
- 4.5 Softmax
- 4.6 Comparação e escolha

### 5. Funções de Perda (Loss Functions)

- 5.1 O papel da função de perda
- 5.2 Erro Quadrático Médio (MSE)
- 5.3 Entropia Cruzada (Cross-Entropy)
- 5.4 Escolha da função de perda

### 6. Otimizadores

- 6.1 Gradiente Descendente
- 6.2 Stochastic Gradient Descent (SGD)
- 6.3 Momentum
- 6.4 Adam (Adaptive Moment Estimation)
- 6.5 Comparação de otimizadores

### 7. Introdução ao TensorFlow/Keras

- 7.1 O que é TensorFlow?
- 7.2 Keras: a API de alto nível
- 7.3 Primeiro modelo: regressão com TensorFlow
- 7.4 Compilação e treinamento

### 8. Visão Geral do PyTorch

- 8.1 O que é PyTorch?
- 8.2 Principais diferenças entre TensorFlow e PyTorch
- 8.3 Exemplo conceitual em PyTorch
- 8.4 Qual escolher?

### 9. Classificação de Imagens com MNIST

- 9.1 O dataset MNIST
- 9.2 Preparação dos dados
- 9.3 Construindo uma MLP para MNIST
- 9.4 Treinamento e avaliação
- 9.5 Visualização dos resultados

### 10. Atividade Prática

- 10.1 Objetivo
- 10.2 Roteiro
- 10.3 Entrega

## Material da aula

- Notebook principal: `script.ipynb`

## Como acompanhar a aula

1. Abra o notebook da aula no VS Code.
2. Execute as células na ordem para acompanhar o fluxo didático.
3. Faça anotações sobre decisões de arquitetura, ativação, perda e otimizador.
4. Ao final, realize a atividade prática proposta.

## Conclusão e próximos passos

Após esta aula, o próximo passo recomendado é implementar e comparar modelos em datasets reais, avaliando impacto de hiperparâmetros, arquitetura e otimizador no desempenho.

## Referências

- Ian Goodfellow, Yoshua Bengio, Aaron Courville — *Deep Learning*.
- Aurélien Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*.
- Documentação oficial do TensorFlow/Keras e PyTorch.