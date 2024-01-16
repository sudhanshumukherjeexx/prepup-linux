# 💻 Prepup

![Static Badge](https://img.shields.io/badge/Built_with_%E2%99%A5%EF%B8%8F-Sudhanshu_Mukherjee-black?link=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fsudhanshumukherjeexx%2F)

[![image](https://img.shields.io/pypi/v/prepup-linux.svg)](https://pypi.python.org/pypi/prepup-linux)
![Static Badge](https://img.shields.io/badge/Linux-Supported-green)
![Static Badge](https://img.shields.io/badge/macOS-Supported-blue)
![Static Badge](https://img.shields.io/badge/Ubuntu-Supported-red)
![Static Badge](https://img.shields.io/badge/License-MIT-purple)

  
  
  

### Prepup is a free open-source package that lets you inspect, explore, visualize, and perform pre-processing tasks on datasets in your linux/ubuntu/macOS terminal or any command line interface.

  

## Installation

- Prepup can be installed using the Pip package manager.

-  **!pip install `prepup-linux`**

  

## Motivation

- Developing an efficient and user-friendly command line tool for data pre-processing to handle various tasks such as missing data, data formatting, and cleaning, with a simple interface and scalability for large datasets.

  

## File Format Supported

- CSV

- EXCEL

- PARQUET

  

## Why you should use Prepup?

  

### It's Superfast

- Prepup is built on Polars which is an alternative to pandas and helps you load and manipulate the DataFrames faster.

  

### Analytical

- Prepup handles tasks ranging from the shape of data to the Standardizing of the feature before training the model. It does it right on the terminal.

  

### Compatible

- Prepup supports CSV, EXCEL, and PARQUET formats making it compatible to go with different file formats.

  

### Non-Destructive

- Prepup doesn't alter your raw data, It saves pre-processed data only when the user specifies the path.

  

### Lives in your Terminal

- Prepup is terminal-based and has specific entry points designed for using it instantly.

  

# Command Line Arguments available in PREPUP

## 🕵️ Prepup "File name or File path" `-inspect`


https://github.com/sudhanshumukherjeexx/prepup-linux/assets/64360018/a15889a6-13a0-447b-885b-f112d87ef846



- inspect flag takes the dataframe and returns the Features available, Features datatype, and missing values present in the Dataset.

- File Name: If the current working directory is same as the file location or FILE PATH

  

## 🧭 Prepup "File name or File path" `-explore`


https://github.com/sudhanshumukherjeexx/prepup-linux/assets/64360018/5753c054-fe5c-46fd-90c2-bfc2cf831c3e

- explore flag takes the dataframe and returns the Features available, Features datatype, Correlation between features, Detects Outliers, Checks Normal Distribution, Checks Skewness, Checks Kurtosis, and also allows the option to check if the dataset is Imbalanced.

- File Name: If the current working directory is same as the file location or FILE PATH

  




## 📊 Prepup "File name or File path" `-visualize`


https://github.com/sudhanshumukherjeexx/prepup-linux/assets/64360018/81bdbf0f-0e63-4360-954d-4d2854eec751

- visualize flag plots of the feature distribution directly on the terminal.

- File Name: If the current working directory is same as the file location or FILE PATH

  

## 🔥 Prepup "File name or File path" `-impute`



https://github.com/sudhanshumukherjeexx/prepup-linux/assets/64360018/7f70f26a-063a-4af8-8f5e-47c7192fd898




- There are 8 different strategies available to impute missing data using Prepup

- File Name: If the current working directory is same as the file location or FILE PATH

  

- Option 1 - Drops the Missing Data

- Option 2 - Impute Missing values with a Specific value

- Option 3 - Impute Missing values with Mean.

- Option 4 - Impute Missing values with Median.

- Option 5 - Impute Missing value based on the distribution of existing columns.

- Option 6 - Impute Missing values based on Forward Fill Strategy where missing values are imputed based on the previous data points.

- Option 7 - Impute Missing values based on Backward Strategy where missing values are imputed based on the next data points.

- Option 8 - Impute missing values based on K-Nearest Neighbors.

  

## 🌐 Prepup "File name or File path" `-standardize`



https://github.com/sudhanshumukherjeexx/prepup-linux/assets/64360018/984b7179-a3f5-4c3d-ab22-c3454593435e





- Standardize allows you to standardize the dataset using two different methods:

1. Robust Scaler

2. Standard Scaler

- Robust Scaler is recommended if there are outliers present and you feel they can have an influence on the Machine Learning model.
  
- Standard Scaler is the go-to function if you want to standardize the dataset before training the model on it.

- File Name: If the current working directory is same as the file location or FILE PATH

# License

- Free software: MIT license

# Package Link

- Github: https://github.com/sudhanshumukherjeexx/prepup-linux

- Documentation: https://sudhanshumukherjeexx.github.io/prepup-linux
