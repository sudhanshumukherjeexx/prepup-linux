#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import os
import sys
from main.common import Prepup
from termcolor import colored
from pyfiglet import Figlet
import time
from sklearn.model_selection import train_test_split
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import shutil
import textwrap
import warnings


term_font = Figlet(font="term")


def load_file(file_path):
    # Check file extension
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Invalid file format. Only CSV, Excel, and Parquet files are supported.")


class PrepupInteractive:
    def __init__(self):
        """Initialize the interactive interface"""
        self.dataframe = None
        self.file_path = None
        self.data_processor = None
        self.term_font = Figlet(font="term")
        self.big_font = Figlet(font="big")
        
    def display_section_header(self, title, subtitle=None):
        """Display a formatted section header that adjusts to screen size"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # get terminal size
        terminal_width = shutil.get_terminal_size().columns
        
        # create decorative border for section
        border_char = "═"
        top_border = "╔" + border_char * (terminal_width - 2) + "╗"
        bottom_border = "╚" + border_char * (terminal_width - 2) + "╝"
        
        print(colored(top_border, 'cyan'))
        
        # display section title with ASCII art
        section_font = Figlet(font='digital')  # You can change the font
        title_ascii = section_font.renderText(title)
        
        # center and display the ASCII title
        for line in title_ascii.split('\n'):
            if line.strip():
                # ensure the line fits within terminal width
                if len(line) > terminal_width - 4:
                    line = line[:terminal_width - 7] + "..."
                centered_line = line.center(terminal_width - 4)
                print(colored("║ " + centered_line + " ║", 'yellow'))
        
        # add subtitle if provided
        if subtitle:
            separator = "─" * (terminal_width - 4)
            print(colored("║ " + separator + " ║", 'cyan'))
            
            # wrap and center subtitle
            wrapped_lines = textwrap.wrap(subtitle, width=terminal_width - 8)
            for line in wrapped_lines:
                centered_line = line.center(terminal_width - 4)
                print(colored("║ " + centered_line + " ║", 'light_blue'))
        
        print(colored(bottom_border, 'cyan'))
        
        # current dataset info if available
        if self.file_path:
            print(f"\nCurrent dataset: {os.path.basename(self.file_path)}")
            if self.dataframe is not None:
                print(f"Shape: {self.dataframe.shape[0]} rows × {self.dataframe.shape[1]} columns")
        
        print("-" * terminal_width)
        print()


    def _display_minimalist_header(self):
        """Display a clean, minimalist header using figlet"""
        terminal_width = shutil.get_terminal_size().columns
        
        # Top border
        print()
        print(colored("═" * terminal_width, 'green'))
        print()
        
        # Use figlet for the logo
        if terminal_width >= 80:
            # Use a font that renders well
            f = Figlet(font='doh', width=terminal_width)
            logo_text = f.renderText('PREPUP!')
            
            # Print each line centered
            for line in logo_text.rstrip().split('\n'):
                print(colored(line.center(terminal_width), 'green'))
        else:
            # Fallback to simple text for narrow terminals
            f = Figlet(font='small', width=terminal_width)
            logo_text = f.renderText('PREPUP!')
            
            for line in logo_text.rstrip().split('\n'):
                print(colored(line.center(terminal_width), 'green'))
        
    
        print()
        
        # Bottom border
        print(colored("═" * terminal_width, 'green'))
        print()

    
    def display_header(self):
        """Display the application header"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Display deprecation warning
        print(colored("\n⚠️ DEPRECATION NOTICE: prepup-linux has been renamed to ride-cli.", 'yellow'))
        print(colored("Please uninstall prepup-linux and install ride-cli instead:", 'yellow'))
        print(colored("  pip uninstall prepup-linux", 'red'))
        print(colored("  pip install ride-cli", 'green'))
        print(colored("This package will be deprecated soon.\n", 'yellow'))
        
        # Display the minimalist header
        self._display_minimalist_header()
        
        # Display the formatted description
        self.display_formatted_description()
        
        # Display current dataset info if available
        if self.file_path:
            print(f"\nCurrent dataset: {os.path.basename(self.file_path)}")
            if self.dataframe is not None:
                print(f"Shape: {self.dataframe.shape[0]} rows × {self.dataframe.shape[1]} columns")
        
        # Dynamic separator based on terminal width
        terminal_width = shutil.get_terminal_size().columns
        print("-" * terminal_width)


    # def display_header(self):
    #     """Display the application header"""
    #     os.system('cls' if os.name == 'nt' else 'clear')
        
    #     # Display the main title
    #     print(colored(self.big_font.renderText("PREPUP !"), 'green'))
        
    #     # Display the formatted description
    #     self.display_formatted_description()
        
    #     # Display current dataset info if available
    #     if self.file_path:
    #         print(f"\nCurrent dataset: {os.path.basename(self.file_path)}")
    #         if self.dataframe is not None:
    #             print(f"Shape: {self.dataframe.shape[0]} rows × {self.dataframe.shape[1]} columns")
        
    #     # Dynamic separator based on terminal width
    #     terminal_width = shutil.get_terminal_size().columns
    #     print("-" * terminal_width)
    
    
    def display_formatted_description(self):
        """Display a properly formatted description that adjusts to screen size"""
        # Get terminal size
        terminal_width = shutil.get_terminal_size().columns
        
        # Create a decorative border
        border_char = "═"
        top_border = "╔" + border_char * (terminal_width - 2) + "╗"
        bottom_border = "╚" + border_char * (terminal_width - 2) + "╝"
        
        # Description text
        description = """Prepup: Interactive Data Analysis Tool

            Prepup is a free open-source package that lets you perform data pre-processing tasks on datasets without writing a single line of code and minimal intervention."""
        
        # Calculate padding for centering
        padding_width = terminal_width - 4  # Account for borders
        
        print(colored(top_border, 'cyan'))
        
        # Split the description into title and body
        lines = description.strip().split('\n')
        title = lines[0]
        body = '\n'.join(lines[1:]).strip()
        
        # Display the title centered (without ASCII art for cleaner look)
        title_padding = " " * ((padding_width - len(title)) // 2)
        print(colored("║ " + title_padding + title + title_padding + " " * (padding_width - len(title_padding) * 2 - len(title)) + " ║", 'yellow', attrs=['bold']))
        
        # Add a separator
        separator = "─" * (terminal_width - 4)
        print(colored("║ " + separator + " ║", 'cyan'))
        
        # Wrap and display the body text
        wrapped_lines = textwrap.wrap(body, width=padding_width-2)
        for line in wrapped_lines:
            line_padding = " " * ((padding_width - len(line)) // 2)
            padded_line = line_padding + line + line_padding + " " * (padding_width - len(line_padding) * 2 - len(line))
            print(colored("║ " + padded_line + " ║", 'light_blue'))
        
        print(colored(bottom_border, 'cyan'))  
        
    def load_data(self):
        """Load data from a file"""

        self.display_section_header(
        "Load Dataset", 
        "Import your data from CSV, Excel, or Parquet files"
    )
        
        file_path = input("Enter the path to your dataset file (CSV, Excel, Parquet): ")
        
        try:
            if file_path.endswith('.csv'):
                self.dataframe = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.dataframe = pd.read_excel(file_path)
            elif file_path.endswith('.parquet'):
                self.dataframe = pd.read_parquet(file_path)
            else:
                print(colored("Error: Unsupported file format. Please use CSV, Excel, or Parquet files.", 'red'))
                input("Press Enter to continue...")
                return
                
            self.file_path = file_path
            self.data_processor = Prepup(self.dataframe)
            
            print(colored(f"\nSuccess! Dataset loaded with {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns.", 'green'))
            
            print("\nPreview of the first 5 rows:")
            print(self.dataframe.head())
            
            input("\nPress Enter to continue to the main menu...")
            
        except Exception as e:
            print(colored(f"Error loading file: {str(e)}", 'red'))
            input("Press Enter to continue...")
            
    def inspect_data(self):
        """Inspect the loaded dataset"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_section_header(
        "Data Inspection",
        "Explore your dataset structure, types, and basic information"
    )
        
        while True:
            print("\nInspection Options:")
            print("1. View features and data types")
            print("2. View dataset shape")
            print("3. Check missing values")
            print("4. View data sample")
            print("5. View data summary statistics")
            print("6. Back to main menu")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == '1':
                print("\nFeatures available in the dataset:")
                print(self.data_processor.features_available())
                print("\nData types of features:")
                print(self.data_processor.dtype_features())
                
            elif choice == '2':
                print("\nShape of the dataset:")
                print(self.data_processor.shape_data())
                
            elif choice == '3':
                print("\nMissing values in the dataset:")
                self.data_processor.missing_plot()
                
            elif choice == '4':
                n_rows = input("\nEnter number of rows to view (default=5): ") or "5"
                try:
                    n = int(n_rows)
                    print(f"\nFirst {n} rows of the dataset:")
                    
                    column_map = {}
            
                    for i, col in enumerate(self.dataframe.columns):
                        # get current data type
                        col_type = str(self.dataframe[col].dtype)
                        
                        # sample values (up to 5)
                        samples = self.dataframe[col].dropna().head(n).values
                        sample_str = ', '.join([str(x) for x in samples])
                        if len(sample_str) > 40:
                            sample_str = sample_str[:37] + "..."
                        
                        # print row
                        print(f"{i+1:<3} {col[:30]:<30} {col_type:<15} {sample_str:<40}")
                        
                        # store mapping
                        column_map[i+1] = col
                except ValueError:
                    print(colored("Invalid input. Please enter a number.", 'red'))
                    
            elif choice == '5':
                print("\nSummary statistics:")
                print(self.dataframe.describe().T) 
                
            elif choice == '6':
                break
                
            else:
                print(colored("Invalid choice. Please try again.", 'red'))
                
            input("\nPress Enter to continue...")
            self.display_section_header(
        "Data Inspection",
        "Explore your dataset structure, types, and basic information"
    )
            
    def explore_data(self):
        """Explore the dataset with statistical analysis"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_section_header(
        "Data Exploration",
        "Perform statistical analysis and discover patterns in your data"
    )
        
        # exploration options
        while True:
            print("\nExploration Options:")
            print("1. Feature correlation analysis")
            print("2. Check for normal distribution")
            print("3. Detect outliers")
            print("4. View skewness")
            print("5. View kurtosis")
            print("6. Check for imbalanced target variable")
            print("7. Back to main menu")
            
            choice = input("\nEnter your choice (1-7): ")
            
            if choice == '1':
                self.data_processor.correlation_n()
                
            elif choice == '2':
                self.data_processor.check_nomral_distrubution()
                
            elif choice == '3':
                self.data_processor.find_outliers()
                
            elif choice == '4':
                self.data_processor.skewness()
                
            elif choice == '5':
                self.data_processor.kurtosis()
                
            elif choice == '6':
                self.data_processor.imbalanced_dataset()
                
            elif choice == '7':
                break
                
            else:
                print(colored("Invalid choice. Please try again.", 'red'))
                
            input("\nPress Enter to continue...")
            self.display_section_header(
        "Data Exploration",
        "Perform statistical analysis and discover patterns in your data"
    )
            
    def visualize_data(self):
        """Visualize the dataset"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_section_header(
            "Data Visualization",
            "Create visual representations of your data"
        )
        
        while True:
            print("\nVisualization Options:")
            print("1. Plot histogram")
            print("2. Plot scatter plot")
            print("3. Back to main menu")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                self.data_processor.plot_histogram()
                
            elif choice == '2':
                self.data_processor.plot_scatter()
                
            elif choice == '3':
                break
                
            else:
                print(colored("Invalid choice. Please try again.", 'red'))
                
            input("\nPress Enter to continue...")
            self.display_section_header(
                "Data Visualization",
                "Create visual representations of your data"
            )

    
            
    def impute_missing_values(self):
        """Handle missing values in the dataset"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_section_header(
        "Missing Value Imputation",
        "Handle missing data using various imputation strategies"
    )
        
        # Check if there are any missing values
        if self.dataframe.isnull().sum().sum() == 0:
            print(colored("\nNo missing values found in the dataset.", 'green'))
            input("\nPress Enter to continue...")
            return
            
        # Process missing values in memory
        try:
            # Copy dataframe
            original_df = self.dataframe.copy()
            
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
            
            df_imputed = None
            
            if choice == 1:
                df_imputed = self.dataframe.dropna()
            elif choice == 2:
                mv = int(input("Enter the value to replace missing data: "))
                df_imputed = self.dataframe.copy()
                numerical_columns = df_imputed.select_dtypes(include='number', exclude=['datetime', 'object'])
                for column in numerical_columns:
                    df_imputed[column] = df_imputed[column].fillna(mv)
            elif choice == 3:
                df_imputed = self.dataframe.copy()
                numerical_columns = df_imputed.select_dtypes(include='number', exclude=['datetime', 'object'])
                for column in numerical_columns:
                    df_imputed[column] = df_imputed[column].fillna(df_imputed[column].mean())
            elif choice == 4:
                df_imputed = self.dataframe.copy()
                numerical_columns = df_imputed.select_dtypes(include='number', exclude=['datetime', 'object'])
                for column in numerical_columns:
                    df_imputed[column] = df_imputed[column].fillna(df_imputed[column].median())
            elif choice == 5:
                df_imputed = self.dataframe.copy()
                numerical_columns = df_imputed.select_dtypes(include='number', exclude=['datetime', 'object'])
                for column in numerical_columns:
                    mean = df_imputed[column].mean()
                    std = df_imputed[column].std()
                    random_values = np.random.normal(loc=mean, scale=std, size=df_imputed[column].isnull().sum())
                    df_imputed[column] = df_imputed[column].fillna(pd.Series(random_values, index=df_imputed[column][df_imputed[column].isnull()].index))
            elif choice == 6:
                df_imputed = self.dataframe.ffill()
            elif choice == 7: 
                df_imputed = self.dataframe.bfill()
            elif choice == 8: 
                df_imputed = self.dataframe.copy()
                numerical_columns = df_imputed.select_dtypes(include='number', exclude=['datetime', 'object'])
                for column in numerical_columns:
                    missing_inds = df_imputed[column].isnull()
                    if missing_inds.sum() > 0:  # Only process if there are missing values
                        non_missing_inds = ~missing_inds
                        non_missing_vals = df_imputed[column][non_missing_inds]
                        if len(non_missing_vals) > 0:  # Only process if there are non-missing values
                            closest_inds = np.abs(np.subtract.outer(
                                np.zeros(missing_inds.sum()), 
                                non_missing_vals.values
                            )).argmin(axis=1)
                            df_imputed.loc[missing_inds, column] = non_missing_vals.iloc[closest_inds].values
            
            if df_imputed is not None:
                # Update dataframe
                self.dataframe = df_imputed
                self.data_processor = Prepup(self.dataframe)
                
                # Export the data
                data_path = input("\nEnter path to save Imputed data: ")
                if data_path:
                    path = data_path.replace(os.sep, '/') if '\\' in data_path else data_path
                    path = path + "/MissingDataImputed.csv" if not path.endswith('.csv') else path
                    
                    try:
                        self.dataframe.to_csv(path, index=False)
                        print(colored(f"\nMissing Data Imputed and saved successfully to {path}", 'green'))
                    except Exception as e:
                        print(colored(f"Error saving CSV: {str(e)}", 'red'))
                        print(colored("Data was imputed in memory but could not be saved to file.", 'yellow'))
                
                print(colored(term_font.renderText("\nDone..."), 'green'))
        except Exception as e:
            print(colored(f"\nError during imputation: {str(e)}", 'red'))
            print(colored("Operation canceled. No changes were made to the dataset.", 'yellow'))
        
        input("\nPress Enter to continue...")
            
    def standardize_features(self):
        """Standardize feature columns"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_section_header(
        "Feature Scaling",
        "Scale and transform your features for better model performance"
    )

        
        # Call feature_scaling method
        self.data_processor.feature_scaling()
        
        input("\nPress Enter to continue...")

    def encode_features(self):
        """Standardize feature columns"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_section_header(
        "Feature Encoding",
        "Convert categorical variables into numerical format"
    )
        
        # Call feature_scaling method
        self.data_processor.feature_encoding()
        
        input("\nPress Enter to continue...")
    
    def change_datatype(self):
        """Standardize feature columns"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_section_header(
        "Data Type Conversion",
        "Convert columns to appropriate data types"
    )
        
        # Call feature_scaling method
        self.data_processor.data_type_conversion()
        
        input("\nPress Enter to continue...")
            
    def display_no_data_message(self):
        """Display a message when no data is loaded"""
        self.display_header()
        print(colored("\nNo dataset loaded. Please load a dataset first.", 'yellow'))
        input("\nPress Enter to continue...")
        
    def export_data(self):
        """Export the current dataframe to a file"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_section_header(
        "Export Data",
        "Save your processed data in various formats"
    )
        
        # export path and format
        export_path = input("\nEnter the path to save the file: ")
        
        print("\nExport Format Options:")
        print("1. CSV (.csv)")
        print("2. Excel (.xlsx)")
        print("3. Parquet (.parquet)")
        
        format_choice = input("\nChoose export format (1-3): ")
        
        file_path = export_path
        if not os.path.isdir(export_path):
            # If export_path is not a directory, extract the directory part
            export_dir = os.path.dirname(export_path)
            if export_dir and not os.path.exists(export_dir):
                print(colored(f"Directory {export_dir} does not exist.", 'red'))
                create_dir = input("Create directory? (y/n): ")
                if create_dir.lower() == 'y':
                    os.makedirs(export_dir)
                else:
                    input("\nExport canceled. Press Enter to continue...")
                    return
        
        try:
            if format_choice == '1':
                if not export_path.endswith('.csv'):
                    file_path = export_path + ".csv"
                self.dataframe.to_csv(file_path, index=False)
                print(colored(f"\nData exported successfully to {file_path}", 'green'))
                
            elif format_choice == '2':
                if not export_path.endswith('.xlsx'):
                    file_path = export_path + ".xlsx"
                self.dataframe.to_excel(file_path, index=False)
                print(colored(f"\nData exported successfully to {file_path}", 'green'))
                
            elif format_choice == '3':
                if not export_path.endswith('.parquet'):
                    file_path = export_path + ".parquet"
                self.dataframe.to_parquet(file_path)
                print(colored(f"\nData exported successfully to {file_path}", 'green'))
                
            else:
                print(colored("Invalid choice. Export canceled.", 'red'))
                
        except Exception as e:
            print(colored(f"Error exporting data: {str(e)}", 'red'))
            
        input("\nPress Enter to continue...")

    
    def automl(self):
        """
        Run AutoML to find the best model for regression or classification tasks
        """
        if self.dataframe is None:
            self.display_no_data_message()
            return
        
        self.display_section_header(
        "AutoML",
        "Automatically find the best machine learning model for your data"
    )
        
        # Get the target variable
        print("\nAvailable columns:")
        for i, col in enumerate(self.dataframe.columns, 1):
            print(f"{i}. {col}")
        
        target_idx = input("\nEnter the number of the target column: ")
        try:
            target_idx = int(target_idx) - 1
            if target_idx < 0 or target_idx >= len(self.dataframe.columns):
                print(colored("Invalid column index. Returning to main menu.", 'red'))
                input("\nPress Enter to continue...")
                return
            
            target_column = self.dataframe.columns[target_idx]
        except ValueError:
            print(colored("Invalid input. Please enter a number.", 'red'))
            input("\nPress Enter to continue...")
            return
        
        # Determine classification or regression
        task_type = ""
        while task_type not in ["1", "2"]:
            print("\nSelect task type:")
            print("1. Classification (target variable has discrete classes)")
            print("2. Regression (target variable has continuous values)")
            task_type = input("\nEnter your choice (1-2): ")
        
        model_type = "classification" if task_type == "1" else "regression"
        
        # Ask user for save path
        while True:
            results_path = input("\nEnter the full path to save the AutoML results (including filename, e.g., /path/to/results.csv): ")
            
            try:
                # Ensure the directory exists
                import os
                directory = os.path.dirname(results_path)
                
                # Create directory if it doesn't exist
                if directory and not os.path.exists(directory):
                    create_dir = input(f"Directory {directory} does not exist. Create it? (y/n): ")
                    if create_dir.lower() == 'y':
                        os.makedirs(directory)
                    else:
                        print("Save canceled.")
                        continue
                
                # Ensure file has .csv extension
                if not results_path.lower().endswith('.csv'):
                    results_path += '.csv'
                
                # Create an instance of AutoML Processor
                try:
                    from main.automl_processor import AutoMLProcessor  # Adjust import path as needed
                    
                    # Initialize AutoML Processor with the current dataframe
                    automl_processor = AutoMLProcessor(self.dataframe)
                    
                    # Run AutoML
                    results = automl_processor.run_automl(target_column, model_type)
                    
                    # Save results to the specified path
                    results.to_csv(results_path)
                    print(f"\n💾 Results saved to {results_path}")
                    input("\nPress Enter to return to the main menu...")
                    break
                
                except Exception as e:
                    print(colored(f"Error running AutoML: {str(e)}", 'red'))
                    retry = input("Do you want to try saving again? (y/n): ")
                    if retry.lower() != 'y':
                        break
            
            except Exception as e:
                print(colored(f"Error saving results: {str(e)}", 'red'))
                retry = input("Do you want to try saving again? (y/n): ")
                if retry.lower() != 'y':
                    break


    def run(self):
        """Run the interactive interface"""
        while True:
            self.display_header()
            
            # main menu
            print("\nMain Menu:")
            print("\n")
            print("1. Load Dataset")
            print("2. Inspect Data")
            print("3. Changing Data Type")
            print("4. Explore Data")
            print("5. Visualize Data")
            print("6. Impute Missing Values")
            print("7. Feature Encoding")
            print("8. Feature Scaling and Tranformation")
            print("9. Export Data")
            print("10. AutoML (Train & Evaluate Models)")
            print("\n'$' Export Data (saves current state)")
            print("\n'exit': Exit Prepup")
            
            
            choice = input("\nEnter your choice (1-10, $, exit): ")
            
            if choice == '1':
                self.load_data()
            elif choice == '2':
                self.inspect_data()
            elif choice == '3':
                self.change_datatype()
            elif choice == '4':
                self.explore_data()
            elif choice == '5':
                self.visualize_data()
            elif choice == '6':
                self.impute_missing_values()
            elif choice == '7':
                self.encode_features()
            elif choice == '8':
                self.standardize_features()
            elif choice == '9':
                self.export_data()
            elif choice == '10':
                self.automl()
            elif choice == '$':
                self.export_data()
            
            elif choice == 'exit':
                print(colored("\nThank you for using Prepup!", 'green'))
                print(colored("Exiting...", 'light_blue'))
                sys.exit(0)
            else: 
                print(colored("\nInvalid Choice. Thank you for using Prepup!", 'green'))
                print(colored("Exiting...", 'light_blue'))
                sys.exit(0)
            


                   

def main():    
    # argument parsing
    parser = argparse.ArgumentParser(description="""
        Prepup: Interactive Data Preprocessing Tool
        Prepup is a free open-source package that lets you perform data pre-processing tasks on datasets without writing a single line of code and minimal intervention.
        """)

    parser.add_argument('file', type=str, nargs='?', help='Optional dataset file to load')
    args = parser.parse_args()
    
    # Create the interactive application
    app = PrepupInteractive()
    
    # If a file was provided, load it
    if args.file:
        try:
            app.file_path = args.file
            app.dataframe = load_file(args.file)
            app.data_processor = Prepup(app.dataframe)
            print(colored(f"Dataset {os.path.basename(args.file)} loaded successfully.", 'green'))
        except Exception as e:
            print(colored(f"Error loading file: {str(e)}", 'red'))
    
    # Run the interactive application
    app.run()


if __name__ == '__main__':
    main()