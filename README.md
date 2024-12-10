# Controlled Evaluation of Syntactic Knowledge in Multilingual Language Models
This repository contains the code and datasets for "[Controlled Evaluation of Syntactic Knowledge in Multilingual Language Models](https://arxiv.org/abs/2411.07474)", accepted to appear at [LoResLM @ COLING 2025](https://loreslm.github.io).

## Repository Structure

- **`data_generation/`**:
     - Vocabulary files and code used to generate synthetic test suites.

- **`results_analysis/`**:  
     - Code for the processing and statistical analysis of evaluation results (including the 'performance versus size' and 'robustness to intervening content' analyses).
     - Code for figures.
     - Code for the Hindi PUD treebank examination.

- **`samples/`**:  
     - Code for sampling sentences from test suites for the human validation experiment.
     - Sampled sentences presented to speakers.

- **`suites/`**:  
     - Test suites (generated using the materials from **`data_generation/`**).

- **`evaluate.py`**:  
     - Script for evaluating LMs on test suites.
