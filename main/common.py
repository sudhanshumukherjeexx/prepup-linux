#!/usr/bin/env python3

import os
import io
import pandas as pd
import plotext as tpl
import numpy as np
import nbformat as nbf
from termcolor import colored
from pyfiglet import Figlet
from scipy.stats import shapiro
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler


                
term_font = Figlet(font="term")

class Prepup:
    
    def __init__(self, dataframe):
        """
        The __init__ function is called when the class is instantiated.
        It takes a dataframe as an argument and assigns it to self.dataframe, which makes it available to other functions in the class.
        """

        if dataframe is None:
            dataframe = pd.DataFrame()
        
        self.dataframe = dataframe
    
    def features_available(self):
        """
        The features_available function returns a list of the features available in the dataframe.
        """
        return self.dataframe.columns
        

    def dtype_features(self):
        """
        The dtype_features function returns the data types of each feature in the dataset.
        This is useful for determining which features are categorical and which are numerical.
        """
        return self.dataframe.dtypes

    def missing_values_count(self):
        """
        The missing_values function returns the number of missing values in each column.
            Args:
                self (DataFrame): The DataFrame object to be analyzed.
            Returns:
                A dictionary with the columns as keys and their respective null counts as values.
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
        """
        return self.dataframe.shape
    
    def missing_plot(self):
        """
        The missing_plot function takes in a dataframe and plots the missing values for each column.
            It also prints out the number of missing values for each column.
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
        Plot histogram for a single selected column
        """
        # numerical columns only
        numerical_columns = self.dataframe.select_dtypes(include=['number']).columns.tolist()
        
        if not numerical_columns:
            print(colored("No numerical columns available for histogram.", 'red'))
            return
        
        # available numerical columns
        print("\nAvailable numerical columns for histogram:")
        print("-" * 50)
        for i, col in enumerate(numerical_columns, 1):
            print(f"{i}. {col}")
        print("-" * 50)
        
        # user selection
        try:
            col_idx = int(input("\nEnter the column number to plot histogram: ")) - 1
            
            if 0 <= col_idx < len(numerical_columns):
                selected_column = numerical_columns[col_idx]
                
                # plot histogram for selected column 
                tpl.clear_data()
                tpl.theme('dark')
                tpl.plotsize(80, 20)
                
                # fill with 0 or drop
                column_data = self.dataframe[selected_column].fillna(0)
                
                print(f"\nPlotting histogram for: {selected_column}")
                tpl.hist(column_data, bins=20, color='light-blue', marker='sd')
                tpl.title(f"Histogram: {selected_column}")
                tpl.show()
                tpl.clear_data()
                
                # basic statistics
                print(f"\nStatistics for {selected_column}:")
                print(f"Mean: {self.dataframe[selected_column].mean():.2f}")
                print(f"Median: {self.dataframe[selected_column].median():.2f}")
                print(f"Std Dev: {self.dataframe[selected_column].std():.2f}")
                print(f"Min: {self.dataframe[selected_column].min():.2f}")
                print(f"Max: {self.dataframe[selected_column].max():.2f}")
                
            else:
                print(colored("Invalid column number.", 'red'))
                
        except ValueError:
            print(colored("Invalid input. Please enter a number.", 'red'))
    
    # def correlation_n(self):
    #     """
    #     The correlation_n function takes in a dataframe and returns the correlation between all numerical features.
    #         The function first selects only the numerical columns from the dataframe, then it creates two lists: one for 
    #         feature pairs and another for their corresponding correlation values. It then uses simple_bar to plot these 
    #         values as a bar graph.
    #     """
    #     numerical_columns = self.dataframe.select_dtypes(include=['number'], exclude=['datetime', 'object'])
    #     features = []
    #     correlation_val = []
    #     for i in numerical_columns:
    #         for j in numerical_columns:
    #             feature_pair = i,j
    #             features.append(feature_pair)
    #             correlation_val.append(round(self.dataframe[i].corr(self.dataframe[j]),2))
    #     tpl.simple_bar(features, correlation_val,width=100, title='Correlation Between these Features', color=92,marker='*')
    #     tpl.show()
    #     tpl.clear_data()

    def correlation_n(self):
        """
        The correlation_n function takes in a dataframe and returns the correlation between all numerical features.
        It displays correlations in a more readable format and optionally creates visualizations.
        """
        numerical_columns = self.dataframe.select_dtypes(include=['number'], exclude=['datetime', 'object'])
        
        if len(numerical_columns.columns) < 2:
            print(colored("Not enough numerical columns for correlation analysis (need at least 2).", 'yellow'))
            return
        
        # Calculate correlation matrix
        corr_matrix = numerical_columns.corr()
        
        # Option 1: Display correlation matrix as a table
        print("\nCorrelation Matrix:")
        print("-" * 50)
        print(corr_matrix.round(3))
        print("-" * 50)
        
        # Option 2: Display only unique correlations (excluding self-correlations)
        features = []
        correlation_val = []
        correlation_pairs = []
        
        # Get unique pairs (avoiding duplicates and self-correlations)
        for i in range(len(numerical_columns.columns)):
            for j in range(i+1, len(numerical_columns.columns)):
                col1 = numerical_columns.columns[i]
                col2 = numerical_columns.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                features.append(f"{col1}-{col2}")
                correlation_val.append(round(corr_value, 3))
                correlation_pairs.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value
        sorted_pairs = sorted(correlation_pairs, key=lambda x: abs(x[2]), reverse=True)
        
        print("\nTop Correlations (excluding self-correlations):")
        print("-" * 60)
        print(f"{'Feature Pair':<30} {'Correlation':<15} {'Strength'}")
        print("-" * 60)
        
        for col1, col2, corr in sorted_pairs:
            # Determine correlation strength
            abs_corr = abs(corr)
            if abs_corr >= 0.7:
                strength = colored("Strong", 'green')
            elif abs_corr >= 0.3:
                strength = colored("Moderate", 'yellow')
            else:
                strength = colored("Weak", 'red')
            
            # Show correlation with color coding
            if corr > 0:
                corr_str = colored(f"{corr:>8.3f}", 'blue')
            else:
                corr_str = colored(f"{corr:>8.3f}", 'red')
                
            print(f"{col1:<15}-{col2:<15} {corr_str:<15} {strength}")
        
        print("-" * 60)
        
        # Plot if there are not too many features
        if len(features) <= 20:
            # Show bar plot
            print("\nCorrelation Bar Plot:")
            tpl.simple_bar(
                features, 
                correlation_val,
                width=100, 
                title='Correlation Between Features (excluding self-correlations)', 
                color='cyan',
                marker='■'
            )
            tpl.show()
            tpl.clear_data()
            
            # Show horizontal bar plot for better readability
            print("\nHorizontal Correlation Plot:")
            tpl.simple_stacked_bar(
                features, 
                [correlation_val],
                width=100,
                title='Feature Correlations',
                labels=['Correlation'],
                colors=['blue'],
                orientation='horizontal'
            )
            tpl.show()
            tpl.clear_data()
        else:
            print(f"\nNote: Too many feature pairs ({len(features)}) for clear visualization.")
            print("Showing top 10 positive and negative correlations only...")
            
            # Get top 10 positive and negative correlations
            sorted_indices = sorted(range(len(correlation_val)), key=lambda k: correlation_val[k])
            top_negative = sorted_indices[:5]
            top_positive = sorted_indices[-5:]
            
            selected_features = []
            selected_values = []
            
            for idx in top_negative + top_positive:
                selected_features.append(features[idx])
                selected_values.append(correlation_val[idx])
            
            tpl.simple_bar(
                selected_features, 
                selected_values,
                width=100, 
                title='Top 5 Positive and Negative Correlations', 
                color='cyan',
                marker='■'
            )
            tpl.show()
            tpl.clear_data()
        
        # Ask if user wants to see heatmap-style visualization
        show_heatmap = input("\nShow correlation heatmap? (y/n): ")
        if show_heatmap.lower() == 'y':
            self._plot_correlation_heatmap(corr_matrix)

    
    def _plot_correlation_heatmap(self, corr_matrix):
        """
        Create an improved text-based heatmap for correlation matrix with proper table formatting
        """
        print("\nCorrelation Heatmap:")
        
        # Limit columns/rows for readability
        max_features = 10
        cols_to_show = corr_matrix.columns[:max_features]
        
        # Format column names to fixed width
        col_width = 12
        shortened_cols = [col[:10] for col in cols_to_show]
        
        # Create table header
        header_line = "│ " + " " * 10 + " │"
        for col in shortened_cols:
            header_line += f" {col:^{col_width-2}} │"
        
        border_line = "├" + "─" * 12 + "┼" + ("─" * col_width + "┼") * len(shortened_cols)
        border_line = border_line[:-1] + "┤"
        
        top_border = "┌" + "─" * 12 + "┬" + ("─" * col_width + "┬") * len(shortened_cols)
        top_border = top_border[:-1] + "┐"
        
        bottom_border = "└" + "─" * 12 + "┴" + ("─" * col_width + "┴") * len(shortened_cols)
        bottom_border = bottom_border[:-1] + "┘"
        
        # Print table
        print(top_border)
        print(header_line)
        print(border_line)
        
        # Print correlation values with colors
        for i, row_name in enumerate(corr_matrix.index[:max_features]):
            row_name_short = row_name[:10]
            row_line = f"│ {row_name_short:<10} │"
            
            for j, col_name in enumerate(cols_to_show):
                value = corr_matrix.iloc[i, j]
                value_str = f"{value:.3f}"
                
                # Color coding
                if i == j:
                    colored_value = colored(value_str, 'white', attrs=['bold'])
                elif value >= 0.7:
                    colored_value = colored(value_str, 'green', attrs=['bold'])
                elif value >= 0.3:
                    colored_value = colored(value_str, 'green')
                elif value >= -0.3:
                    colored_value = colored(value_str, 'yellow')
                elif value >= -0.7:
                    colored_value = colored(value_str, 'red')
                else:
                    colored_value = colored(value_str, 'red', attrs=['bold'])
                
                # Add padding to maintain alignment with ANSI codes
                padding = col_width - 2 - len(value_str)
                row_line += f" {colored_value}{' ' * padding} │"
            
            print(row_line)
        
        print(bottom_border)
        
        # Legend
        print("\nColor Legend:")
        print("  " + colored("■", 'green', attrs=['bold']) + " Strong positive  (0.70 to 1.00)")
        print("  " + colored("■", 'green') + " Moderate positive (0.30 to 0.69)")
        print("  " + colored("■", 'yellow') + " Weak correlation  (-0.29 to 0.29)")
        print("  " + colored("■", 'red') + " Moderate negative (-0.69 to -0.30)")
        print("  " + colored("■", 'red', attrs=['bold']) + " Strong negative  (-1.00 to -0.70)")
        print("  " + colored("■", 'white', attrs=['bold']) + " Self correlation  (1.00)")
        
        # Ask if user wants visual representation
        show_visual = input("\nShow visual correlation map? (y/n): ")
        if show_visual.lower() == 'y':
            self._plot_visual_correlation_map(corr_matrix)

    
    def _plot_visual_correlation_map(self, corr_matrix):
        """Create a visual correlation map with bars"""
        print("\nVisual Correlation Map:")
        
        # Limit features for readability
        max_features = 10
        cols_to_show = corr_matrix.columns[:max_features]
        
        # Column setup
        col_width = 12
        shortened_cols = [col[:10] for col in cols_to_show]
        
        # Create borders
        top_border = "┌" + "─" * 12 + "┬" + ("─" * col_width + "┬") * len(shortened_cols)
        top_border = top_border[:-1] + "┐"
        
        bottom_border = "└" + "─" * 12 + "┴" + ("─" * col_width + "┴") * len(shortened_cols)
        bottom_border = bottom_border[:-1] + "┘"
        
        border_line = "├" + "─" * 12 + "┼" + ("─" * col_width + "┼") * len(shortened_cols)
        border_line = border_line[:-1] + "┤"
        
        # Header
        header_line = "│ " + " " * 10 + " │"
        for col in shortened_cols:
            header_line += f" {col:^{col_width-2}} │"
        
        print(top_border)
        print(header_line)
        print(border_line)
        
        # Visual bars
        for i, row_name in enumerate(corr_matrix.index[:max_features]):
            row_name_short = row_name[:10]
            row_line = f"│ {row_name_short:<10} │"
            
            for j, col_name in enumerate(cols_to_show):
                value = corr_matrix.iloc[i, j]
                
                if i == j:
                    cell_content = colored("█████", 'white', attrs=['bold']) + " 1.00"
                else:
                    # Create visual bar
                    abs_value = abs(value)
                    if abs_value >= 0.8:
                        bar = "█████"
                    elif abs_value >= 0.6:
                        bar = "████ "
                    elif abs_value >= 0.4:
                        bar = "███  "
                    elif abs_value >= 0.2:
                        bar = "██   "
                    elif abs_value >= 0.1:
                        bar = "█    "
                    else:
                        bar = "·    "
                    
                    # Color based on positive/negative
                    if value > 0:
                        colored_bar = colored(bar, 'blue')
                    else:
                        colored_bar = colored(bar, 'red')
                    
                    value_str = f"{value:5.2f}"
                    cell_content = f"{colored_bar} {value_str}"
                
                row_line += f" {cell_content:<{col_width-2}} │"
            
            print(row_line)
        
        print(bottom_border)
        
        # Visual legend
        print("\nVisual Legend:")
        print(f"  {colored('█████', 'blue')} Strong positive   |  {colored('█████', 'red')} Strong negative")
        print(f"  {colored('████ ', 'blue')} Moderate positive |  {colored('████ ', 'red')} Moderate negative")
        print(f"  {colored('███  ', 'blue')} Weak positive     |  {colored('███  ', 'red')} Weak negative")
        print(f"  {colored('██   ', 'blue')} Very weak positive|  {colored('██   ', 'red')} Very weak negative")
        print(f"  {colored('█    ', 'blue')} Negligible positive| {colored('█    ', 'red')} Negligible negative")
        print(f"  {colored('·    ', 'white')} No correlation")

    
    def plot_scatter(self):
        """
        Plot scatter plot for two selected columns
        """
        
        # numerical columns only
        numerical_columns = self.dataframe.select_dtypes(include=['number']).columns.tolist()
        
        if len(numerical_columns) < 2:
            print(colored("Need at least 2 numerical columns for scatter plot.", 'red'))
            return
        
        # available numerical columns
        print("\nAvailable numerical columns for scatter plot:")
        print("-" * 50)
        for i, col in enumerate(numerical_columns, 1):
            print(f"{i}. {col}")
        print("-" * 50)
        
        try:
            # first column
            x_idx = int(input("\nEnter the column number for X-axis: ")) - 1
            
            if not (0 <= x_idx < len(numerical_columns)):
                print(colored("Invalid column number for X-axis.", 'red'))
                return
                
            # second column
            y_idx = int(input("Enter the column number for Y-axis: ")) - 1
            
            if not (0 <= y_idx < len(numerical_columns)):
                print(colored("Invalid column number for Y-axis.", 'red'))
                return
                
            if x_idx == y_idx:
                print(colored("Please select different columns for X and Y axes.", 'yellow'))
                return
                
            x_column = numerical_columns[x_idx]
            y_column = numerical_columns[y_idx]
            

            # prepare data (handle missing values)
            scatter_df = pd.DataFrame({
                'x': self.dataframe[x_column].fillna(0),
                'y': self.dataframe[y_column].fillna(0)
            })
            
            tpl.clear_data()
            tpl.theme('matrix')
            tpl.plotsize(80, 20)
            
            print(f"\nPlotting scatter plot: {x_column} vs {y_column}")
            tpl.scatter(scatter_df['x'], scatter_df['y'], color='white')
            tpl.title(f"Scatter Plot: {x_column} vs {y_column}")
            tpl.xlabel(x_column)
            tpl.ylabel(y_column)
            tpl.show()
            tpl.clear_data()
            
            # show correlation
            correlation = self.dataframe[x_column].corr(self.dataframe[y_column])
            print(f"\nCorrelation between {x_column} and {y_column}: {correlation:.4f}")
            
            if abs(correlation) < 0.3:
                print(colored("Weak correlation", 'yellow'))
            elif abs(correlation) < 0.7:
                print(colored("Moderate correlation", 'blue'))
            else:
                print(colored("Strong correlation", 'green'))
                
        except ValueError:
            print(colored("Invalid input. Please enter a number.", 'red'))


    def find_outliers(self, k=1.5):
        """
        The find_outliers function takes a dataframe and returns the outliers in each column.
            The function uses the interquartile range to determine if a value is an outlier or not.
            The default k value is 1.5, but can be changed by passing in another float as an argument.
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
    
    
    def feature_encoding(self):
        """
        The feature_encoding function provides various encoding options for categorical variables.
        
        Options include:
        - Label Encoding: Converts categorical values to numerical labels (e.g., "red", "green", "blue" → 0, 1, 2)
        - One Hot Encoding: Creates binary columns for each category (e.g., "color_red", "color_green", "color_blue")
        - Ordinal Encoding: Similar to label encoding but allows specifying an order (e.g., "small", "medium", "large" → 0, 1, 2)
        """
        try:
            # load dataframe
            isExist = os.path.exists("missing_data.parquet")
            if isExist == True:
                dataframe = pd.read_parquet("missing_data.parquet")
            else:
                dataframe = self.dataframe
            
            # categorical columns
            categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_columns:
                print(colored("No categorical columns found in the dataset.", 'yellow'))
                return
            
            print("\nCategorical columns available in the dataset:")
            print("-" * 80)
            print(f"{'#':<3} {'Column Name':<30} {'Data Type':<10} {'Unique Values':<15} {'Sample Values'}")
            print("-" * 80)
            
            for i, col in enumerate(categorical_columns):
                # column info
                col_type = str(dataframe[col].dtype)
                unique_count = dataframe[col].nunique()
                
                # sample values (up to 3)
                samples = dataframe[col].dropna().unique()[:5]
                sample_str = ', '.join([str(x) for x in samples])
                if len(sample_str) > 20:
                    sample_str = sample_str[:20] + "..."
                
                print(f"{i+1:<3} {col[:30]:<30} {col_type:<10} {unique_count:<15} {sample_str}")
            
            print("-" * 80)
            
            # allow user to select columns
            print("\nSelect categorical columns to encode:")
            print("(Enter column numbers separated by commas, or 'all' to select all columns)")
            
            cols_input = input("\nEnter columns to encode: ")
            
            selected_columns = []
            if cols_input.lower() == 'all':
                selected_columns = categorical_columns
                print(f"\nSelected all {len(selected_columns)} categorical columns.")
            else:
                try:
                    # parse the input and get column indices
                    col_indices = [int(idx.strip()) - 1 for idx in cols_input.split(',')]
                    # get column names
                    selected_columns = [categorical_columns[idx] for idx in col_indices if 0 <= idx < len(categorical_columns)]
                    
                    if not selected_columns:
                        print(colored("\nNo valid columns selected. Exiting function.", 'red'))
                        return
                    
                    print(f"\nSelected columns: {', '.join(selected_columns)}")
                except ValueError:
                    print(colored("\nInvalid input. Exiting function.", 'red'))
                    #input("\nPress Enter to continue...")
                    return
            
            # dataframe copy
            df = dataframe.copy()
            
            # encoding options
            print("\nEncoding Methods:")
            print("1. Label Encoding - Maps each unique value to a number")
            print("2. One Hot Encoding - Creates binary columns for each category")
            print("3. Ordinal Encoding - Maps values to ordered integers based on specified order")
            print("4. Exit and return to main menu")
            
            encoding_choice = input("\nSelect encoding method (1-4): ")
            
            if encoding_choice == '4':
                print(colored("Exiting feature encoding function.", 'yellow'))
                return
            elif encoding_choice not in ['1', '2', '3']:
                print(colored("Invalid choice. Exiting function.", 'red'))
                return
            
            # label Encoding
            if encoding_choice == '1':
                
                for col in selected_columns:
                    # handle NaN values
                    df[col] = df[col].fillna('Unknown')
                    
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    
                    # mapping for reference
                    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    print(f"\nMapping for column '{col}':")
                    for orig, encoded in mapping.items():
                        print(f"  {orig} → {encoded}")
                
                print(colored("\nLabel encoding completed successfully!", 'green'))

            # One Hot Encoding    
            elif encoding_choice == '2':
                            
                # ask user to drop the original columns
                drop_orig = input("\nDrop original columns after encoding? (y/n): ").lower() == 'y'
                
                for col in selected_columns:
                    # handle NaN values
                    df[col] = df[col].fillna('Unknown')
                    
                    # apply one-hot encoding
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df[[col]])
                    
                    # DataFrame with encoded data
                    encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
                    
                    # add encoded columns to original dataframe
                    df = pd.concat([df, encoded_df], axis=1)
                    
                    # if requested, drop original column 
                    if drop_orig:
                        df = df.drop(col, axis=1)
                    
                    print(f"\nEncoded column '{col}' into {len(encoded_cols)} new columns.")
                
                print(colored("\nOne Hot encoding completed successfully!", 'green'))

            # Ordinal Encoding    
            elif encoding_choice == '3':
            
                for col in selected_columns:
                    # Handle NaN values
                    df[col] = df[col].fillna('Unknown')
                    
                    # get unique values
                    unique_vals = df[col].unique().tolist()
                    
                    print(f"\nColumn: {col}")
                    print(f"Unique values: {', '.join(str(x) for x in unique_vals)}")
                    
                    # ask user to provide order
                    print("\nSpecify the order for ordinal encoding:")
                    print("Enter values separated by commas, from lowest to highest rank")
                    print("Press Enter to use alphabetical order")
                    
                    order_input = input("> ")
                    
                    if order_input.strip():
                        try:
                            # parse user-provided order
                            ordered_values = [x.strip() for x in order_input.split(',')]
                            
                            # validate all unique values are included
                            missing_vals = set(unique_vals) - set(ordered_values)
                            if missing_vals:
                                print(f"Warning: Values not included in order: {', '.join(str(x) for x in missing_vals)}")
                                print("These will be assigned to NaN. Continue? (y/n)")
                                if input().lower() != 'y':
                                    continue
                            
                            # mapping
                            mapping = {val: i for i, val in enumerate(ordered_values)}
                            
                            # apply mapping
                            df[col] = df[col].map(mapping)
                            
                            # Handle values not in mapping
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                        except Exception as e:
                            print(colored(f"Error creating ordinal mapping: {str(e)}", 'red'))
                            print("Using default alphabetical order instead.")
                            # apply default ordinal encoding (alphabetical)
                            mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
                            df[col] = df[col].map(mapping)
                    else:
                        # apply default ordinal encoding (alphabetical)
                        mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
                        df[col] = df[col].map(mapping)
                    
                    # print mapping for reference
                    print(f"\nMapping for column '{col}':")
                    for orig, encoded in sorted(mapping.items(), key=lambda x: x[1]):
                        print(f"  {orig} → {encoded}")
                
                print(colored("\nOrdinal encoding completed successfully!", 'green'))
            
            # save the encoded data
            data_path = input("\nEnter path to save encoded data (or type 'exit' to cancel): ")
            
            if data_path.lower() == 'exit':
                print(colored("Operation canceled. Data was encoded but not saved.", 'yellow'))
                return
            
            try:
                # path
                path = data_path
                if '\\' in path:
                    path = path.replace(os.sep, '/')
                
                if not path.endswith('.csv'):
                    path = path + "/EncodedData.csv"
                
                # create directory if it doesn't exist
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                # save to CSV
                df.to_csv(path, index=False)
                print(colored(f"\nEncoded data saved successfully to {path}", 'green'))
                
                # update the class dataframe with the encoded version
                self.dataframe = df
                
            except Exception as e:
                print(colored(f"Error saving file: {str(e)}", 'red'))
                print(colored("Data was encoded but could not be saved to file.", 'yellow'))
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(colored(f"An unexpected error occurred: {str(e)}", 'red'))
            traceback.print_exc()  
            input("\nPress Enter to continue...")
            return
    
    
    def data_type_conversion(self):
        """
        The data_type_conversion function allows users to change the data types of columns
        using pandas' native type conversion functionality.
        
        Supported conversions:
        - String types: str, object
        - Integer types: int8, int16, int32, int64
        - Float types: float16, float32, float64
        - DateTime types: datetime64
        - Boolean type: bool
        
        """
        try:
            import pandas as pd
            import numpy as np
            
            # load dataframe
            isExist = os.path.exists("missing_data.parquet")
            if isExist:
                dataframe = pd.read_parquet("missing_data.parquet")
            else:
                dataframe = self.dataframe
                
            if dataframe is None or dataframe.empty:
                print(colored("No data available for conversion.", 'red'))
                input("\nPress Enter to continue...")
                return
                
                
            # column preview with current data types
            print("\nColumns available for data type conversion:")
            print("-" * 90)
            print(f"{'#':<3} {'Column Name':<30} {'Current Type':<15} {'Sample Values':<40}")
            print("-" * 90)
            
            # mapping of column index to column name for later reference
            column_map = {}
            
            for i, col in enumerate(dataframe.columns):
                # get current data type
                col_type = str(dataframe[col].dtype)
                
                # sample values (up to 3)
                samples = dataframe[col].dropna().head(3).values
                sample_str = ', '.join([str(x) for x in samples])
                if len(sample_str) > 40:
                    sample_str = sample_str[:37] + "..."
                
                # print row
                print(f"{i+1:<3} {col[:30]:<30} {col_type:<15} {sample_str:<40}")
                
                # store mapping
                column_map[i+1] = col
                
            print("-" * 90)
            
            # select columns to convert
            print("\nSelect columns to convert:")
            print("(Enter column numbers separated by commas)")
            
            cols_input = input("\nEnter column numbers: ")
            
            try:
                # parse column indices
                selected_indices = [int(idx.strip()) for idx in cols_input.split(',')]
                
                # validate indices and get column names
                selected_columns = []
                for idx in selected_indices:
                    if idx in column_map:
                        selected_columns.append(column_map[idx])
                    else:
                        print(colored(f"Warning: Column number {idx} is not valid and will be skipped.", 'yellow'))
                
                if not selected_columns:
                    print(colored("No valid columns selected. Exiting function.", 'red'))
                    return
                    
                print(f"\nSelected columns: {', '.join(selected_columns)}")
                
            except ValueError:
                print(colored("Invalid input. Exiting function.", 'red'))
                return
            
            # Create a copy of the dataframe to work with
            df_converted = dataframe.copy()
            
            # Data type conversion options
            dtype_options = {
                "1": {"name": "String (object)", "dtype": "object"},
                "2": {"name": "Integer (int8)", "dtype": "int8"},
                "3": {"name": "Integer (int16)", "dtype": "int16"},
                "4": {"name": "Integer (int32)", "dtype": "int32"},
                "5": {"name": "Integer (int64)", "dtype": "int64"},
                "6": {"name": "Float (float16)", "dtype": "float16"},
                "7": {"name": "Float (float32)", "dtype": "float32"},
                "8": {"name": "Float (float64)", "dtype": "float64"},
                "9": {"name": "DateTime", "dtype": "datetime64[ns]"},
                "10": {"name": "Boolean", "dtype": "bool"},
                "11": {"name": "Skip this column", "dtype": None}
            }
            
            # process each selected column
            for col in selected_columns:
                # get current data type
                current_type = str(df_converted[col].dtype)
                
                print(f"\nColumn: {col} (Current type: {current_type})")
                print("Choose target data type:")
                
                # display data type options
                for key, option in dtype_options.items():
                    print(f"{key}. {option['name']}")
                
                type_choice = input(f"\nEnter your choice (1-{len(dtype_options)}): ")
                
                if type_choice not in dtype_options:
                    print(colored(f"Invalid choice. Skipping column '{col}'.", 'yellow'))
                    continue
                    
                # skip if chosen
                if type_choice == "11":
                    print(f"Skipping column '{col}'")
                    continue
                    
                try:
                    target_dtype = dtype_options[type_choice]["dtype"]
                    
                    # handle conversion based on target type
                    if target_dtype == "object":
                        # convert to string
                        df_converted[col] = df_converted[col].astype(str)
                        
                    elif target_dtype in ["int8", "int16", "int32", "int64"]:
                        # convert to integer with proper error handling
                        # first convert to float (handles NaN better) then to integer
                        df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                        # use pandas nullable integer type (Int64, Int32, etc) to handle NaN values
                        pandas_nullable_type = f"Int{target_dtype[-2:]}"  # Extract 64, 32, etc.
                        df_converted[col] = df_converted[col].astype(pandas_nullable_type)
                        
                    elif target_dtype in ["float16", "float32", "float64"]:
                        # convert to float
                        df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype(target_dtype)
                        
                    elif target_dtype == "datetime64[ns]":
                        # convert to datetime
                        df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                        
                    elif target_dtype == "bool":
                        # for strings, convert common boolean terms
                        if pd.api.types.is_string_dtype(df_converted[col]):
                            df_converted[col] = df_converted[col].str.lower().isin(['true', '1', 'yes', 'y', 't', 'on'])
                        else:
                            # for numeric, handle NaN properly
                            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').fillna(0).astype(bool)
                    
                    print(colored(f"✅ Successfully converted '{col}' to {dtype_options[type_choice]['name']}", 'green'))
                    
                    # Show sample of converted data
                    print("\nSample of converted data:")
                    print(df_converted[col].head(3))
                    
                except Exception as e:
                    print(colored(f"❌ Error converting '{col}': {str(e)}", 'red'))
                    print("Column will remain unchanged.")
            
            # Save the converted data
            save_choice = input("\nSave the converted data? (y/n): ")
            
            if save_choice.lower() == 'y':
                data_path = input("\nEnter path to save converted data (or type 'exit' to cancel): ")
                
                if data_path.lower() == 'exit':
                    print(colored("Operation canceled. Data was converted but not saved.", 'yellow'))
                else:
                    try:
                        # path
                        path = data_path
                        if '\\' in path:
                            path = path.replace(os.sep, '/')
                        
                        if not path.endswith('.csv'):
                            path = path + "/ConvertedData.csv"
                        
                        # create directory if it doesn't exist
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        
                        # Save to CSV
                        df_converted.to_csv(path, index=False)
                        print(colored(f"\nConverted data saved successfully to {path}", 'green'))
                        
                    except Exception as e:
                        print(colored(f"Error saving file: {str(e)}", 'red'))
                        print(colored("Data was converted but could not be saved to file.", 'yellow'))
            
            # update the class dataframe
            update_df = input("\nUpdate the current working dataframe with these conversions? (y/n): ")
            
            if update_df.lower() == 'y':
                self.dataframe = df_converted
                print(colored("Working dataframe updated with converted data types.", 'green'))
            
            
        except Exception as e:
            print(colored(f"An unexpected error occurred: {str(e)}", 'red'))
            import traceback
            traceback.print_exc()  # print detailed error for debugging
            return


    def feature_scaling(self):
        """
        The feature_scaling function provides various scaling and transformation options for the dataframe.
        
        Options include:
        - Min-Max Scaling: Scales features to a range, typically [0,1]
        - Standardization (Z-score): Scales to mean=0, std=1
        - Robust Scaler: Scales using statistics robust to outliers
        - Max Abs Scaler: Scales by dividing by the maximum absolute value
        
        And transformations:
        - Quantile Transformer: Maps to uniform or normal distribution
        - Log Transformer: Natural logarithm transformation
        - Reciprocal Transformation: 1/x transformation
        - Square Root Transformation: √x transformation
        
        """
        try:
            isExist = os.path.exists("missing_data.parquet")
            if isExist == True:
                dataframe = pd.read_parquet("missing_data.parquet")
            else:
                dataframe = self.dataframe
            
            print("\nFeature Scaling and Transformation Options:")
            
            # display available columns and allow user to select columns to drop
            print("\nAvailable columns in the dataset:")
            for i, col in enumerate(dataframe.columns):
                print(f"{i+1}. {col}")
            
            print("\nSelect columns to drop before feature scaling or transformation:")
            print("(Enter column numbers separated by commas, or 'None' to keep all columns)")
            
            drop_input = input("\nEnter columns to drop: ")
            
            if drop_input.lower() != 'none':
                try:
                    # parse the input and get column indices
                    drop_indices = [int(idx.strip()) - 1 for idx in drop_input.split(',')]
                    columns_to_drop = [dataframe.columns[idx] for idx in drop_indices if 0 <= idx < len(dataframe.columns)]
                    
                    if columns_to_drop:
                        # drop selected columns
                        df = dataframe.drop(columns=columns_to_drop)
                        print(f"\nDropped columns: {', '.join(columns_to_drop)}")
                    else:
                        df = dataframe
                        print(colored("\nNo valid columns selected to drop. Using all columns.", 'red'))
                except ValueError:
                    df = dataframe
                    print(colored("\nInvalid input. Using all columns.", 'red'))
            else:
                df = dataframe
                print(colored("\nUsing all columns.", 'green'))
            
            print("\nAvailable Options:")
            print("\n=== SCALING OPTIONS ===")
            print("1. Min-Max Scaler [Scales features to a range of [0,1]]")
            print("2. Standard Scaler (Z-score) [Scales to mean=0, std=1]")
            print("3. Robust Scaler [Recommended if outliers are present]")
            print("4. Max Abs Scaler [Scales by dividing by the maximum absolute value]")
            
            print("\n=== TRANSFORMATION OPTIONS ===")
            print("5. Quantile Transformer [Maps to uniform or normal distribution]")
            print("6. Log Transformer [Natural logarithm transformation]")
            print("7. Reciprocal Transformation [1/x transformation]")
            print("8. Square Root Transformation [√x transformation]")
            print("="*97)
            print("9. Exit and return to main menu")
            
            choice_str = input("\nEnter your choice (1-9): ")
            
            # check if user wants to exit
            if choice_str == '9':
                print(colored("Exiting feature scaling function.", 'yellow'))
                #input("\nPress Enter to continue...")
                return
            
            try:
                choice = int(choice_str)
                if choice < 1 or choice > 8:
                    print(colored("Invalid choice. Please enter a number between 1 and 8.", 'red'))
                    return
            except ValueError:
                print(colored("Invalid input. Please enter a number.", 'red'))
                return
            
            numerical_columns = df.select_dtypes(include=['number'], exclude=['datetime', 'object'])
            
            # if no numerical columns are found
            if len(numerical_columns.columns) == 0:
                print(colored("No numerical columns found for scaling/transformation.", 'red'))
                input("\nPress Enter to continue...")
                return
            
            # apply the selected scaling/transformation method
            if choice == 1:
                # Min-Max Scaling
                for column in numerical_columns:
                    scaler = MinMaxScaler()
                    df[[column]] = scaler.fit_transform(df[[column]])
                
            elif choice == 2:
                # Standard Scaler (Z-score)
                for column in numerical_columns:
                    scaler = StandardScaler()
                    df[[column]] = scaler.fit_transform(df[[column]])
                    
            elif choice == 3:
                # Robust Scaler
                for column in numerical_columns:
                    robust = RobustScaler()
                    df[[column]] = robust.fit_transform(df[[column]])
                    
            elif choice == 4:
                # Max Abs Scaler
                for column in numerical_columns:
                    scaler = MaxAbsScaler()
                    df[[column]] = scaler.fit_transform(df[[column]])
                    
            elif choice == 5:
                # Quantile Transformer
                distribution = input("\nSelect distribution type ('uniform' or 'normal'): ").lower()
                if distribution not in ['uniform', 'normal']:
                    distribution = 'uniform'  # Default if invalid input
                    print(colored(f"Invalid distribution type. Using '{distribution}' as default.", 'yellow'))
                
                try:
                    n_quantiles = min(100, len(df))  # ensure n_quantiles doesn't exceed data length
                    
                    for column in numerical_columns:
                        transformer = QuantileTransformer(output_distribution=distribution, n_quantiles=n_quantiles)
                        # handle potential NaN values
                        mask = df[column].notna()
                        if mask.all():
                            df[[column]] = transformer.fit_transform(df[[column]])
                        else:
                            temp = df.loc[mask, column].values.reshape(-1, 1)
                            temp = transformer.fit_transform(temp)
                            df.loc[mask, column] = temp.flatten()
                except Exception as e:
                    print(colored(f"Error during quantile transformation: {str(e)}", 'red'))
                    input("\nPress Enter to continue...")
                    return
                    
            elif choice == 6:
                # log Transformer
                try:
                    for column in numerical_columns:
                        # handle zeros and negative values before log transform
                        min_val = df[column].min()
                        if min_val <= 0:
                            offset = abs(min_val) + 1  # Add offset to make all values positive
                            print(colored(f"Column '{column}' contains values ≤ 0. Adding offset: {offset}", 'yellow'))
                            df[column] = df[column] + offset
                        
                        df[column] = np.log(df[column])
                except Exception as e:
                    print(colored(f"Error during log transformation: {str(e)}", 'red'))
                    input("\nPress Enter to continue...")
                    return
                    
            elif choice == 7:
                # Reciprocal Transformation (1/x)
                try:
                    for column in numerical_columns:
                        # handle zeros before reciprocal transform
                        zeros_mask = df[column] == 0
                        if zeros_mask.any():
                            print(colored(f"Warning: Column '{column}' contains zeros which cannot be reciprocally transformed.", 'yellow'))
                            print("Adding small epsilon value to zeros.")
                            min_non_zero = df.loc[df[column] != 0, column].min()
                            epsilon = min_non_zero / 1000 if min_non_zero > 0 else 1e-10
                            df.loc[zeros_mask, column] = epsilon
                        
                        df[column] = 1 / df[column]
                except Exception as e:
                    print(colored(f"Error during reciprocal transformation: {str(e)}", 'red'))
                    input("\nPress Enter to continue...")
                    return
                    
            elif choice == 8:
                # Square Root Transformation
                try:
                    for column in numerical_columns:
                        # handle negative values before sqrt transform
                        neg_mask = df[column] < 0
                        if neg_mask.any():
                            print(colored(f"Warning: Column '{column}' contains negative values which cannot be square root transformed.", 'yellow'))
                            print("Taking absolute value before transformation.")
                            df[column] = df[column].abs()
                        
                        df[column] = np.sqrt(df[column])
                except Exception as e:
                    print(colored(f"Error during square root transformation: {str(e)}", 'red'))
                    input("\nPress Enter to continue...")
                    return
            
            # Save the transformed data
            data_path = input("\nEnter path to save normalized/transformed data (or type 'exit' to cancel): ")
            
            if data_path.lower() == 'exit':
                print(colored("Operation canceled. Data was transformed but not saved.", 'yellow'))
                input("\nPress Enter to continue...")
                return
            
            try:
                path = data_path  
                s = "\\"
                if s in path:
                    path = path.replace(os.sep, '/')
                    path = path + "/TransformedData.csv" 
                    path = str(path)
                    print(path)
                else:
                    path = path + "/TransformedData.csv"
                
                df.to_csv(path)
                
                operation_type = "scaled" if choice <= 4 else "transformed"
                print(colored(f"\nFeatures {operation_type} and saved successfully", 'green'))
                
            except Exception as e:
                print(colored(f"Error saving file: {str(e)}", 'red'))
                print(colored("Data was transformed but could not be saved to file.", 'yellow'))
                input("\nPress Enter to continue...")
        
        except Exception as e:
            print(colored(f"An unexpected error occurred: {str(e)}", 'red'))
            input("\nPress Enter to continue...")
            return
    
    def check_nomral_distrubution(self):
        """
        The check_nomral_distrubution function checks if the dataframe is normally distributed.
            It does this by using the Shapiro-Wilk test to check for normality. 
            The function will print out a message stating whether or not each column in the dataframe is normally distributed.
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
        """
        try:
            dataframe = self.dataframe
            
            if dataframe is None or dataframe.empty:
                print(colored("No data available for analysis.", 'red'))
                input("\nPress Enter to continue...")
                return
            
            
            # Display column list with numbering
            print("\nAvailable columns for target variable analysis:")
            print("-" * 80)
            print(f"{'#':<3} {'Column Name':<30} {'Data Type':<15} {'Unique Values':<10}")
            print("-" * 80)
            
            # Create a mapping of column index to column name for selection
            column_map = {}
            
            for i, col in enumerate(dataframe.columns):
                # Get column info
                col_type = str(dataframe[col].dtype)
                unique_count = dataframe[col].nunique()
                
                # Print row
                print(f"{i+1:<3} {col[:30]:<30} {col_type:<15} {unique_count:<10}")
                
                # Store mapping
                column_map[i+1] = col
                
            print("-" * 80)
            
            # Get user selection
            print("\nSelect target variable by number (or enter 0 to exit):")
            target_idx = input("\nEnter column number: ")
            
            try:
                target_idx = int(target_idx)
                
                if target_idx == 0:
                    print(colored("Exiting imbalanced dataset function.", 'yellow'))
                    #input("\nPress Enter to continue...")
                    return
                    
                if target_idx not in column_map:
                    print(colored(f"Invalid column number {target_idx}. Exiting function.", 'red'))
                    #input("\nPress Enter to continue...")
                    return
                    
                # Get selected column name
                target_col = column_map[target_idx]
                print(f"\nSelected target variable: {target_col}")
                
                # Calculate and plot distribution
                target_dist = dataframe[target_col].value_counts()
                
                # Check number of unique values
                if len(target_dist) > 30:
                    print(colored(f"Warning: Target variable has {len(target_dist)} unique values, which may be too many for a clear visualization.", 'yellow'))
                    proceed = input("Continue with visualization anyway? (y/n): ")
                    if proceed.lower() != 'y':
                        print(colored("Visualization canceled.", 'yellow'))
                        #input("\nPress Enter to continue...")
                        return
                
                # Plot the distribution
                tpl.simple_bar(
                    target_dist.index, 
                    target_dist.values, 
                    width=100,
                    title=f'Distribution of {target_col}',
                    color=92
                )
                tpl.show()
                tpl.clear_data()
                
                # Show class distribution statistics
                print("\nClass Distribution Statistics:")
                print("-" * 50)
                total = target_dist.sum()
                dist_df = pd.DataFrame({
                    'Value': target_dist.index,
                    'Count': target_dist.values,
                    'Percentage': (target_dist.values / total * 100).round(2)
                })
                print(dist_df.sort_values('Count', ascending=False))
                
                # Check for class imbalance
                max_class = target_dist.max()
                min_class = target_dist.min()
                imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
                
                print("\nImbalance Analysis:")
                if imbalance_ratio > 10:
                    print(colored(f"Severe class imbalance detected! Ratio of {imbalance_ratio:.2f}:1 (majority:minority)", 'red'))
                elif imbalance_ratio > 3:
                    print(colored(f"Moderate class imbalance detected. Ratio of {imbalance_ratio:.2f}:1 (majority:minority)", 'yellow'))
                else:
                    print(colored(f"Classes are relatively balanced. Ratio of {imbalance_ratio:.2f}:1 (majority:minority)", 'green'))
                    
            except ValueError:
                print(colored("Invalid input. Please enter a valid column number.", 'red'))
                
            #input("\nPress Enter to continue...")
            
        except Exception as e:
            print(colored(f"An unexpected error occurred: {str(e)}", 'red'))
            input("\nPress Enter to continue...")
            return
    
    def skewness(self):
        dataframe = self.dataframe
        # Get numeric columns
        numeric_data = dataframe.select_dtypes(include=['number', 'Int64', 'Int32', 'Int16', 'Int8'])
        
        if numeric_data.empty:
            print(colored("No numeric columns found for skewness calculation", 'yellow'))
            return
        
        # Convert nullable integers to regular integers for scipy functions
        numeric_data_converted = numeric_data.copy()
        for col in numeric_data_converted.columns:
            # Check if column is a nullable integer type
            if numeric_data_converted[col].dtype.name in ['Int64', 'Int32', 'Int16', 'Int8']:
                # Convert to regular numpy int, filling NaN with 0 or dropping them
                # Option 1: Fill NaN with 0 (you can choose the appropriate strategy)
                numeric_data_converted[col] = numeric_data_converted[col].fillna(0).astype('int64')
                # Option 2: Or drop NaN values for this calculation
                # numeric_data_converted[col] = numeric_data_converted[col].dropna().astype('int64')
        
        try:
            skew_results = skew(numeric_data_converted, axis=0, bias=True)
            print("\t\nSkewness present in the data:")
            for col, skew_val in zip(numeric_data_converted.columns, skew_results):
                print(f"{col}: {skew_val:.4f}")
        except Exception as e:
            print(colored(f"Error calculating skewness: {str(e)}", 'red'))
            print("Trying column-by-column calculation...")
            
            # Fallback: calculate skewness for each column individually
            for col in numeric_data_converted.columns:
                try:
                    # Drop NaN values for this column
                    col_data = numeric_data_converted[col].dropna()
                    if len(col_data) > 0:
                        skew_val = skew(col_data)
                        print(f"{col}: {skew_val:.4f}")
                    else:
                        print(f"{col}: No valid data for skewness calculation")
                except Exception as col_e:
                    print(f"{col}: Error - {str(col_e)}")

    def kurtosis(self):
        dataframe = self.dataframe
        # Get numeric columns
        numeric_data = dataframe.select_dtypes(include=['number', 'Int64', 'Int32', 'Int16', 'Int8'])
        
        if numeric_data.empty:
            print(colored("No numeric columns found for kurtosis calculation", 'yellow'))
            return
        
        # Convert nullable integers to regular integers for scipy functions
        numeric_data_converted = numeric_data.copy()
        for col in numeric_data_converted.columns:
            # Check if column is a nullable integer type
            if numeric_data_converted[col].dtype.name in ['Int64', 'Int32', 'Int16', 'Int8']:
                # Convert to regular numpy int, filling NaN with 0 or dropping them
                # Option 1: Fill NaN with 0 (you can choose the appropriate strategy)
                numeric_data_converted[col] = numeric_data_converted[col].fillna(0).astype('int64')
                # Option 2: Or drop NaN values for this calculation
                # numeric_data_converted[col] = numeric_data_converted[col].dropna().astype('int64')
        
        try:
            kurtosis_results = kurtosis(numeric_data_converted, axis=0, bias=True)
            print("\t\nKurtosis present in the data:")
            for col, kurt_val in zip(numeric_data_converted.columns, kurtosis_results):
                print(f"{col}: {kurt_val:.4f}")
        except Exception as e:
            print(colored(f"Error calculating kurtosis: {str(e)}", 'red'))
            print("Trying column-by-column calculation...")
            
            # Fallback: calculate kurtosis for each column individually
            for col in numeric_data_converted.columns:
                try:
                    # Drop NaN values for this column
                    col_data = numeric_data_converted[col].dropna()
                    if len(col_data) > 0:
                        kurt_val = kurtosis(col_data)
                        print(f"{col}: {kurt_val:.4f}")
                    else:
                        print(f"{col}: No valid data for kurtosis calculation")
                except Exception as col_e:
                    print(f"{col}: Error - {str(col_e)}")


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
    



