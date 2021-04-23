# Evaluation of Nonparametric Bayesian Classification Methods for Predicting Fake News from Article Titles

Nathaniel Hawkins   

A49510775   

CSE802, Spring Semester 2021   

## Overview of Repository

* `src/` - Source code used in this project. Scripts are labeled in order they should be executed to recreate the results shown in this report.
    * `00_exploratory_analysis.ipynb` - Exploratory data analysis notebook. Despite numbering, the data needs to be downloaded and processed first (can be run at any time after `02`)
    * `01_data-download-and-exploration.ipynb` - Download and process dataset with some initial exploration
    * `02_text-preprocessing.ipynb` - Preprocess the text into feature and label matrices
    * `03_feature-construction.ipynb` - Create one-hot encodings of titles
    * `04_baselines.ipynb` - Baseline models prototyping in jupyter notebook
    * `04_baselines.py` - Script version of above notebook
    * `04_baselines.sb` - Job script to run above script version of baselines on HPCC
    * `05_compile-baselines.ipynb` - Compile baseline results and explore in table
    * `06_nonparametrics.py` - Implement nonparametric models
    * `06_nonparametrics.sb` - Job script to run above script on HPCC
    * `07_compile_results.ipynb` - Compile all results into tables for exploration
* `doc/` - Documents for this project. This includes report and initial proposal.
* `results/` - Resulting data files from executing code in `src/` folder. Plots saved to this directory as well.
* `data/` - Houses dataset used in this project. Note: due to the size of the dataset, this directory will be empty until dataset is downloaded and processed.