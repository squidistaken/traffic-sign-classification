# Traffic Sign Classification

A machine learning project for classifying German traffic signs using neural networks.

Authors:

* David van Wuijkhuijse (s5592968)

* Marcus Harald Olof Persson (s5343798)

* Richard Frank Harnisch (s5238366)

## Project Overview

This project aims to classify traffic signs from the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_about.html) using a custom (convolutional) neural network implementation. The project includes data loading, preprocessing, model training, evaluation, and visualisation tools.

Secondly, the project serves as an educational exploration on developing a neural network from scratch. All code is original, and there is no reliance on PyTorch or other ML frameworks (excluding Jupyter Notebook explorations).

We recommend exploring the codebase through Jupyter Notebooks. Be wary the production quality does not match PyTorch or other frameworks, so things may break.

## Installation

The project can be quickly set up with the [`uv` package manager](https://github.com/astral-sh/uv). All dependencies, environments, etc. will be installed.

```bash
uv sync
```
## Downloading the Data

Before running the project, you need to download the dataset. The dataset used in this project is the German Traffic Sign Recognition Benchmark (GTSRB).

```bash
python gtsrb_download.py
```
