# qrt-2024

*Owner :* Jules CHAPON - jules.b.chapon@gmail.com | jules.chapon.pro@gmail.com
*Version :* 1.0.0

## Description

This repo contains my solution to the QRT "Challenge Data" 2024 : https://challengedata.ens.fr/challenges/143.

The goal of this challenge is to predict the result of football games.

Both training and testing data are given by QRT. It is forbidden to use external data.

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

python

```

## My solution
