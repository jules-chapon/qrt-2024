# qrt-2024

*Owner :* Jules CHAPON - jules.b.chapon@gmail.com | jules.chapon.pro@gmail.com

*Version :* 1.0.0

This repo is not the final version of my project. I still have some refinements to make, especially regarding feature-engineering and fine-tuning. However, the pipeline can be considered as complete, and the project can already be used.

## Description

This repo contains my solution to the QRT "Challenge Data" 2024 : https://challengedata.ens.fr/challenges/143.

The goal of this challenge is to predict the result of football games.

Both training and testing data are given by QRT. It is forbidden to use external data.

On the branch "main", you will find the "whole" projects, where all functions and methods are defined.

On the branch "best", you will find the best version of the model, the one that gives the best score on the test dataset.

Other branches contain different versions, where I fine-tune the parameters or try new methods.

## How to use this repo

To install all dependencies, you can do as follow :

- Create an environment with Python 3.10 and install Poetry > 1.7 :

```bash

conda create -n my_env python=3.10
conda activate my_env
pip install poetry>1.7

```

Then, you can install all dependencies with the following command :

```bash

poetry install

```

To run the model, you have to download the different databases on the challenge website : https://challengedata.ens.fr/challenges/143.

You just have to copy and paste training and testing files in the folder : data/input.

You are now able to run the pipeline by running the following command in your terminal :

```bash

python -m predict_foot_result

```

## My solution

### Description of the pipeline

- Preprocessing
- Model
    - Feature Selection
    - Data Sampling
    - Model Selection
    - Fine-Tuning
- Postprocessing

### Preprocessing

For now, preprocessing has been quite simple.
Indeed, values have already been normalized. Hence, all numerical values are between 0 and 10.
All statistics have different modalities (sum, average and standard deviation) and time horizons (season or 5 last games).

For this first version, I decided to use only average stats, as it allows to put all games at the same level. The problem with absolute values is that it depends on the moment of the season, and we do not have this onformation.
On next versions, I will try to add standard deviation to see if it can improve the model.

I also decided to use both season and last games stats, as it gives a better understanding of both global and current dynamics of the team/player.

For now, I decided to replace all missing numerical values by 0 because it seemed that missing values mainly affect players with few played minutes. Hence, I considered that it was relevant to say a player who played 5 minutes is likely not to have many scored goals.

Regarding team features, I decided to keep all statistics. Irrelevant features will be dismissed by the feature selection.

Regarding player features, I had to make a decision :
- For each team, I only kept the most important (based on played minutes) players for each role (1 goal, 5 defenders, 5 midfielders and 5 attackers).
- For each role, I only kept the most important features (I chose the features that I considered to be relevant).
I might consider adding more players and/or more features for each role.
Irrelevant features will be dismissed by the feature selection.

Labels are just gathered in a unique column: 0 if home wins, 1 if draw and 2 if away wins.

### Model

#### Feature Selection

I have 2 methods to select features based on their importance:
- First one is to create random features, make a quick random forest, and to compare their importance with other features. Hence, if a feature is less important than those 5 random columns, I do not choose it.
- Second one is to use Boruta, that will automatically select relevant features. Boruta is used after the "random-column" method.

#### Data Sampling

As our data is imbalanced (less draws), I added a function that allows to resample data in order to balance training data.

This is an option, and it seems not to improve the quality of the model.

#### Model Selection

I have two classifiers :
- a dummy one, that always predict "home wins". It is used as a baseline.
- a LightGBM one. It has immediately been better than the dummy model, and that is why I decided to keep using it.

#### Fine-tuning

The fine-tuning of the hyperparameters of the LGBM model are done with Optuna.

It gives an extra precision to the model.

### Postprocessing

For now, postprocessing is just about formatting the results, to upload them in the expected format.
