#!/usr/bin/env python3

"""This module contains functions and classes used by the other modules.

    author: "Neokai"
"""

import pandas as pd
import plotext as tpl
import numpy as np
import io
import nbformat as nbf
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from scipy.stats import shapiro
from scipy.stats import skew
from scipy.stats import kurtosis
from termcolor import colored
from pyfiglet import Figlet



term_font = Figlet(font="term")

class Prepup:
    
    def __init__(self, dataframe):
        """
        The __init__ function is called when the class is instantiated.
        It takes a dataframe as an argument and assigns it to self.dataframe, which makes it available to other functions in the class.
        
        :param self: Represent the instance of the class
        :param dataframe: Pass the dataframe to the class
        :return: An instance of the class
        :author: Neokai
        """
        if dataframe is None:
            dataframe = pd.DataFrame()
        
        self.dataframe = dataframe
    
    def features_available(self):
        """
        The features_available function returns a list of the features available in the dataframe.
                
        
        :param self: Represent the instance of the class
        :return: A list of the column names in the dataframe
        :author: Neokai
        """
        return self.dataframe.columns
        

    def dtype_features(self):
        """
        The dtype_features function returns the data types of each feature in the dataset.
            This is useful for determining which features are categorical and which are numerical.
        
        :param self: Represent the instance of the class
        :return: A series with the data type of each feature (column) in a pandas dataframe
        :author: Neokai
        """
        return self.dataframe.dtypes

    def missing_values_count(self):
        """
        The missing_values function returns the number of missing values in each column.
            Args:
                self (DataFrame): The DataFrame object to be analyzed.
            Returns:
                A dictionary with the columns as keys and their respective null counts as values.
        
        :param self: Represent the instance of the class
        :return: The number of missing values in each column
        :author: Neokai
        """
        if self.dataframe.empty() == True:
            print("No Missing Value Found")
        else:
            missing_value = self.dataframe.isnull().sum()
            return missing_value
    
    def shape_data(self):
        """
        The shape_data function returns the shape of the dataframe.
                :return: The shape of the dataframe as a tuple (rows, columns)
        
        :param self: Represent the instance of the class
        :return: The shape of the dataframe
        :author: Neokai
        """
        return self.dataframe.shape
    
    def missing_plot(self):
        """
        The missing_plot function takes in a dataframe and plots the missing values for each column.
            It also prints out the number of missing values for each column.
        
        :param self: Represent the instance of the class
        :return: The number of missing values for each column in the dataframe
        :author: Neokai
        """
        empty_count = 0
        non_empty_count = 0

        for column in self.dataframe.columns:
            if self.dataframe[column].isnull().any():
                empty_count += 1
            else:
                non_empty_count += 1

        if empty_count == 0:
            print(colored(term_font.renderText("No Missing Value Found"), 'green'))
        else:
            missing_counts = self.dataframe.isnull().sum()
            missing_counts = missing_counts[missing_counts != 0]

            df_new = pd.DataFrame({
                'Features': missing_counts.index,
                'Missing_Value_Count': missing_counts.values
            })

            print(df_new, "\n")

            tpl.simple_bar(df_new['Features'], df_new['Missing_Value_Count'],width=100, title='Missing Value Count in Each Feature', color='red+')
            tpl.theme('matrix')
            tpl.show()
            tpl.clear_data()
            
    def plot_histogram(self):
        """
        The plot_histogram function plots a histogram for each numerical column in the dataframe.
            The function takes no arguments and returns nothing.
        
        :param self: Represent the instance of the class
        :return: A histogram for each of the columns in the dataframe
        :author: Neokai
        """
        dataframe = self.dataframe.fillna(0)
        numerical_columns = dataframe.select_dtypes(include=['number'], exclude=['datetime', 'object'])
        for column in numerical_columns:
            tpl.clear_data()
            tpl.theme('dark')
            tpl.plotsize(80,20)
            print("\n")
            tpl.hist(dataframe[column], bins=20,color='light-blue', marker='sd') #color=46)
            tpl.title(column)
            tpl.show()
            tpl.clear_data()
    
    def correlation_n(self):
        """
        The correlation_n function takes in a dataframe and returns the correlation between all numerical features.
            The function first selects only the numerical columns from the dataframe, then it creates two lists: one for 
            feature pairs and another for their corresponding correlation values. It then uses simple_bar to plot these 
            values as a bar graph.
        
        :param self: Represent the instance of the class
        :return: A bar graph of the correlation between all numerical features in the dataset
        :author: Neokai
        """
        numerical_columns = self.dataframe.select_dtypes(include=['number'], exclude=['datetime', 'object'])
        features = []
        correlation_val = []
        for i in numerical_columns:
            for j in numerical_columns:
                feature_pair = i,j
                features.append(feature_pair)
                correlation_val.append(round(self.dataframe[i].corr(self.dataframe[j]),2))
        tpl.simple_bar(features, correlation_val,width=100, title='Correlation Between these Features', color=92,marker='*')
        tpl.show()
        tpl.clear_data()

    def scatter_plot(self):
        """
        The scatter_plot function takes the dataframe and selects all columns that are numeric.
        It then creates a scatter plot for each pair of numeric columns in the dataframe.
        
        :param self: Represent the instance of the class
        :return: A scatter plot for each column in the dataframe
        :author: Neokai
        """
        numerical_columns = self.dataframe.select_dtypes(include=['number'], exclude=['datetime', 'object'])
        for i in numerical_columns:
            for j in numerical_columns:
                scatter_p = pd.DataFrame()
                scatter_p["value1"] = self.dataframe[[i]] 
                scatter_p["value2"] = self.dataframe[[j]]
                tpl.theme('matrix')
                tpl.plotsize(80,20)
                tpl.title("\nDistribution of {0} vs {1}".format(i,j))
                tpl.scatter(scatter_p["value1"], scatter_p["value2"], color='white')
                tpl.show()
                tpl.clear_data()
    
    def find_outliers(self, k=1.5):
        """
        The find_outliers function takes a dataframe and returns the outliers in each column.
            The function uses the interquartile range to determine if a value is an outlier or not.
            The default k value is 1.5, but can be changed by passing in another float as an argument.
        
        :param self: Represent the instance of the class
        :param k: Calculate the iqr (interquartile range)
        :return: A print statement of the outliers detected in each column
        :author: Neokai
        """
        numerical_columns = self.dataframe.select_dtypes(include=['number'], exclude=['datetime', 'object'])

        for i in numerical_columns:
            outliers = []
            q1 = np.percentile(self.dataframe[i].values, 25)
            q3 = np.percentile(self.dataframe[i].values, 75)
            iqr = q3 - q1
            lower_bound = q1 - k*iqr
            upper_bound = q3 + k*iqr
            for j in self.dataframe[i].values:
                if j < lower_bound or j > upper_bound:
                    outliers.append(j)
            print(f"\tOutliers detected in {i}\n")
    
    
    def feature_scaling(self):
        """
        The feature_scaling function normalizes the dataframe by subtracting the mean and dividing by standard deviation.
            The function takes in a dataframe as input, drops the target variable if it is specified, 
            converts to pandas dataframe and then performs feature scaling on all columns that are not categorical or boolean. 
            It then saves this normalized dataset as a csv file at user-specified path.
        
        :param self: Represent the instance of the class
        :return: A dataframe with normalized values
        :author: Neokai
        """
        isExist = os.path.exists("missing_data.parquet")
        if isExist == True:
            dataframe = pd.read_parquet("missing_data.parquet")
        else:
            dataframe = self.dataframe
        target_col = input("\nEnter the Target variable to drop (or 'None'): ")
        
        if target_col != "None":
            df = dataframe
            df = df.drop(target_col, axis=1)
        else:
            df = dataframe
        
        
        print("Choice Available for Standardization: \n")
        print("\t1. [Press 1] Robust Scaler [Recommended if outliers are present.].\n")
        print("\t2. [Press 2] Standard Scaler \n")
        choice = int(input("\nEnter your choice: "))
        numerical_columns = df.select_dtypes(include=['number'], exclude=['datetime', 'object'])
        if choice==1:
            for column in numerical_columns:
                robust = RobustScaler()
                df[[column]] = robust.fit_transform(df[[column]])
        elif choice==2:    
            for column in numerical_columns:
                scaler = StandardScaler()
                df[[column]] = scaler.fit_transform(df[[column]])
        
        data_path = input("\nEnter path to save normalized data : ")
        path = data_path  
        s = "\\"
        if s in path:
            path = path.replace(os.sep, '/')
            path = path + "/NormalizedData.csv" 
            path = str(path)
            print(path)
        else:
            path = path + "/NormalizedData.csv"
        df.to_csv(path)
        print("\nFeature Normalized and saved successfully")
    
    def check_nomral_distrubution(self):
        """
        The check_nomral_distrubution function checks if the dataframe is normally distributed.
            It does this by using the Shapiro-Wilk test to check for normality. 
            The function will print out a message stating whether or not each column in the dataframe is normally distributed.
        
        :param self: Represent the instance of the class
        :return: The name of the column, whether it is normally distributed or not and its p-value
        :author: Neokai
        """
        dataframe = self.dataframe
        numerical_columns = dataframe.select_dtypes(include=['number'], exclude=['datetime', 'object'])
        for column in numerical_columns:
            stats, p_value = shapiro(dataframe[column])
            if p_value > 0.05:
                h_8 = Figlet(font='term') 
                print(colored(h_8.renderText(f"* {column} is Normally Distributed with a p-value of {p_value:.2f}\n"),'green'))
            else:
                h_8 = Figlet(font='term')
                print(colored(h_8.renderText(f"* {column} doesn't have a Normal Distribution with a p-value of {p_value} \n"), 'red'))


    def imbalanced_dataset(self):    
        """
        The imbalanced_dataset function takes in a dataframe and the target variable as input.
        It then plots a bar graph of the distribution of the target variable.
        
        :param self: Represent the instance of the class
        :return: A bar plot of the target variable distribution
        :author: Neokai
        """
        dataframe = self.dataframe
        val = input("Enter the Target Variable or ('None'): ")
        if val != "None":
            target_dist = dataframe[val].value_counts()
            tpl.simple_bar(target_dist.index, target_dist.values, width=100,title='Target Variable Distribution',color=92)
            tpl.show()
            tpl.clear_data()
        else:
            print("This Function is Terminated.")
    
    def skewness(self):
        dataframe = self.dataframe
        numeric_data = dataframe.select_dtypes(['number'])
        print("\t\nSkewness present in the data: ",skew(numeric_data, axis=0, bias=True))
    
    def kurtosis(self):
        dataframe = self.dataframe
        numeric_data = dataframe.select_dtypes(['number'])
        print("\t\nKurtosis present in the data: ",kurtosis(numeric_data, axis=0, bias=True))


    def handle_missing_values(self):
        """
        The handle_missing_values function is used to handle missing values in the dataset.
        The user can choose from a variety of options to impute missing data, or drop it altogether.
        
        
        :param self: Represent the instance of the class
        :return: The dataframe with missing values imputed
        :author: Neokai
        """
        dataframe = self.dataframe
        print("Choice Available to Impute Missing Data: \n")
        print("\t1. [Press 1] Drop Missing Data.\n")
        print("\t2. [Press 2] Impute Missing Data with Specific Value.\n")
        print("\t3. [Press 3] Impute Missing Data with Mean.\n")
        print("\t4. [Press 4] Impute Missing Data with Median.\n")
        print("\t5. [Press 5] Impute Missing Data based on Distribution of each Feature..\n")
        print("\t6. [Press 6] Impute Missing Data with Fill Forward Strategy.\n")
        print("\t7. [Press 7] Impute Missing Data with Backward Fill Strategy.\n")
        print("\t8. [Press 8] Impute Missing Data with Nearest Neighbours (Advisable if dataset has missing values randomly).\n")
        choice = int(input("\nEnter your choice: "))
        if choice == 1:
            dataframe = dataframe.dropna()
            dataframe.to_parquet("missing_data.parquet")
            data_path = input("\nEnter path to save Imputed data : ")
            path = data_path  
            s = "\\"
            if s in path:
                path = path.replace(os.sep, '/')
                path = path + "/MissingDataImputed.csv" 
                path = str(path)
                print(path)
            else:
                path = path + "/MissingDataImputed.csv"
        
            dataframe.to_csv(path)
            print("\nMissing Data Imputed and saved succesfully")
            print(colored(term_font.renderText("\nDone...")))
        elif choice == 2:
            mv = int(input("Enter the value to replace missing data: "))
            numerical_columns = dataframe.select_dtypes(include='number', exclude=['datetime', 'object'])
            for column in numerical_columns:
                dataframe[column] = dataframe[column].fillna(mv)
            dataframe.to_parquet("missing_data.parquet")
            data_path = input("\nEnter path to save Imputed data : ")
            path = data_path  
            s = "\\"
            if s in path:
                path = path.replace(os.sep, '/')
                path = path + "/MissingDataImputed.csv" 
                path = str(path)
                print(path)
            else:
                path = path + "/MissingDataImputed.csv"
        
            dataframe.to_csv(path)
            print("\nMissing Data Imputed and saved succesfully")
            print(colored(term_font.renderText("\nDone...")))
        elif choice == 3:
            numerical_columns = dataframe.select_dtypes(include='number', exclude=['datetime', 'object'])
            for column in numerical_columns:
                dataframe[column] = dataframe[column].fillna(dataframe[column].mean())
            dataframe.to_parquet("missing_data.parquet")
            data_path = input("\nEnter path to save Imputed data : ")
            path = data_path  
            s = "\\"
            if s in path:
                path = path.replace(os.sep, '/')
                path = path + "/MissingDataImputed.csv" 
                path = str(path)
                print(path)
            else:
                path = path + "/MissingDataImputed.csv"
        
            dataframe.to_csv(path)
            print("\nMissing Data Imputed and saved succesfully")
            print(colored(term_font.renderText("\nDone...")))
        elif choice == 4:
            numerical_columns = dataframe.select_dtypes(include='number', exclude=['datetime', 'object'])
            for column in numerical_columns:
                dataframe[column] = dataframe[column].fillna(dataframe[column].median())
            dataframe.to_parquet("missing_data.parquet")
            data_path = input("\nEnter path to save Imputed data : ")
            path = data_path  
            s = "\\"
            if s in path:
                path = path.replace(os.sep, '/')
                path = path + "/MissingDataImputed.csv" 
                path = str(path)
                print(path)
            else:
                path = path + "/MissingDataImputed.csv"
        
            dataframe.to_csv(path)
            print("\nMissing Data Imputed and saved succesfully")
            print(colored(term_font.renderText("\nDone...")))
        elif choice == 5:
            numerical_columns = dataframe.select_dtypes(include='number', exclude=['datetime', 'object'])
            for column in numerical_columns:
                mean = dataframe[column].mean()
                std = dataframe[column].std()
                random_values = np.random.normal(loc=mean, scale=std, size=dataframe[column].isnull().sum())
                dataframe[column] = dataframe[column].fillna(pd.Series(random_values,index=dataframe[column][dataframe[column].isnull()].index))
            dataframe.to_parquet("missing_data.parquet")
            data_path = input("\nEnter path to save Imputed data : ")
            path = data_path  
            s = "\\"
            if s in path:
                path = path.replace(os.sep, '/')
                path = path + "/MissingDataImputed.csv" 
                path = str(path)
                print(path)
            else:
                path = path + "/MissingDataImputed.csv"
        
            dataframe.to_csv(path)
            print("\nMissing Data Imputed and saved succesfully")
            print(colored(term_font.renderText("\nDone...")))
        elif choice == 6:
            dataframe = dataframe.ffill()
            dataframe.to_parquet("missing_data.parquet")
            data_path = input("\nEnter path to save Imputed data : ")
            path = data_path  
            s = "\\"
            if s in path:
                path = path.replace(os.sep, '/')
                path = path + "/MissingDataImputed.csv" 
                path = str(path)
                print(path)
            else:
                path = path + "/MissingDataImputed.csv"
        
            dataframe.to_csv(path)
            print("\nMissing Data Imputed and saved succesfully")
            print(colored(term_font.renderText("\nDone...")))
        elif choice == 7: 
            dataframe = dataframe.bfill()
            dataframe.to_parquet("missing_data.parquet")
            data_path = input("\nEnter path to save Imputed data : ")
            path = data_path  
            s = "\\"
            if s in path:
                path = path.replace(os.sep, '/')
                path = path + "/MissingDataImputed.csv" 
                path = str(path)
                print(path)
            else:
                path = path + "/MissingDataImputed.csv"
        
            dataframe.to_csv(path)
            print("\nMissing Data Imputed and saved succesfully")
            print(colored(term_font.renderText("\nDone...")))
        elif choice == 8: 
            numerical_columns = dataframe.select_dtypes(include='number', exclude=['datetime', 'object'])
            for column in numerical_columns:
                missing_inds = dataframe[column].isnull()
                non_missing_inds = ~missing_inds
                non_missing_vals = dataframe[column][non_missing_inds]
                closest_inds = np.abs(dataframe[column][missing_inds].values - non_missing_vals.values.reshape(-1,1)).argmin(axis=0)
                dataframe.loc[missing_inds, column] = non_missing_vals.iloc[closest_inds].values
            dataframe.to_parquet("missing_data.parquet")
            data_path = input("\nEnter path to save Imputed data : ")
            path = data_path  
            s = "\\"
            if s in path:
                path = path.replace(os.sep, '/')
                path = path + "/MissingDataImputed.csv" 
                path = str(path)
                print(path)
            else:
                path = path + "/MissingDataImputed.csv"
        
            dataframe.to_csv(path)
            print("\nMissing Data Imputed and saved succesfully")
            print(colored(term_font.renderText("\nDone...")))
    



