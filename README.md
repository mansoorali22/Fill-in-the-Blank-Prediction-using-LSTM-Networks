# Fill-in-the-Blank Prediction using LSTM Networks

This repository contains the implementation of a project that applies LSTM networks to predict missing words in sentences. Using the RACE dataset, fill-in-the-blank questions are constructed, and two LSTM models (Forward and Backward) are trained to predict the missing word based on contextual information.

---

## Objective

The goal of this project is to:
1. Develop two LSTM-based models:
   - **Forward LSTM**: Predicts the missing word using the first half of the sentence (before the blank).
   - **Backward LSTM**: Predicts the missing word using the reversed second half of the sentence (after the blank).
2. Compare the predictions from both models and determine the better answer for filling in the blank.
3. Fine-tune the models and document the decision-making process in a detailed report.

---

## Dataset

- **Dataset**: [RACE Dataset](https://huggingface.co/datasets/race)
- **Description**: The RACE dataset consists of reading comprehension passages and questions. Sentences from this dataset are used to create fill-in-the-blank questions by removing a random word from the latter half of each sentence.
- **Loading the Dataset**:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("race", "all")