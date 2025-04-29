NLP Text Prediction
=================

This repository contains the code and results for the Natural Language Processing (NLP), as part of a course. The project involves building probabilistic language models using n-grams and neural networks to predict the next word in a given text based on various model architectures and techniques.

Project Overview
----------------

The goal of this project is to apply Natural Language Processing techniques to a dataset containing text data, particularly focusing on:

1.  Building probabilistic language models (unigram, bigram, trigram) and evaluating them.

2.  Implementing a Feed Forward Neural Network for text prediction.

3.  Evaluating the models using the perplexity metric.

4.  Generating predictions for missing words in incomplete sentences.

### Dataset

The dataset used for this project is the Reuters News dataset, which is split into three parts:

-   **Train.csv**: 8550 records (for training)

-   **Val.csv**: 1069 records (for validation)

-   **Test.csv**: 1069 records (for testing)

-   **Sample.csv**: 100 records (to predict the next word in incomplete sentences)

The dataset consists of a single column of text data used for training and testing the models.

### Task Breakdown

The project is divided into the following sections:

#### 1\. Probabilistic Language Models

-   Implement a **Unigram**, **Bigram**, and **Trigram** language model.

-   Apply **Kneser-Ney smoothing** for better handling of zero probabilities.

-   Evaluate the models on the **validation dataset**.

-   Calculate and compare the **Perplexity** for each model on the **test dataset**.

#### 2\. Feed Forward Neural Network (FFNN)

-   Convert text into **n-gram features** and use them as input to the FFNN.

-   Implement a **feed-forward neural network** with at least one hidden layer to predict the next word in a sequence.

-   Train the neural network using the validation dataset.

-   Evaluate the neural network on the test dataset and calculate **Perplexity**.

#### 3\. Text Prediction

-   Use the trained models to predict missing words in incomplete sentences from the **sample.csv** file.

-   Output the predicted words and compare the models' performances based on accuracy and perplexity.

### Dependencies

-   Python 3.x

-   Libraries:

    -   `numpy`

    -   `pandas`

    -   `sklearn`

    -   `tensorflow` (for neural network)

    -   `matplotlib` (for visualizations)

    -   `nltk` (for n-gram generation and text processing)


### Results

The models' performance can be evaluated using the **Perplexity** metric, with lower values indicating better predictions. The best model in terms of perplexity and accuracy will be selected as the final model.