#!/usr/bin/env python3

"""Main module.

   author : "Neokai"
"""
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
        
    def display_header(self):
        """Display the application header"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(colored(self.big_font.renderText("PREPUP !"), 'green'))
        print(colored("Interactive Data Preprocessing & Analysis Tool", 'light_blue'))
        print(colored("Prepup is a free open-source package that lets you perform data pre-processing tasks on datasets without writing a single line of code and minimal intervention.", 'light_blue'))
        print("-" * 80)
        
        if self.file_path:
            print(f"Current dataset: {os.path.basename(self.file_path)}")
            if self.dataframe is not None:
                print(f"Shape: {self.dataframe.shape[0]} rows × {self.dataframe.shape[1]} columns")
        print("-" * 80)
        
    def load_data(self):
        """Load data from a file"""
        self.display_header()
        print(colored(self.term_font.renderText("Load Dataset"), 'light_blue'))
        
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
            
            # Preview data
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
            
        self.display_header()
        print(colored(self.term_font.renderText("Data Inspection"), 'light_blue'))
        
        # submenu for inspection options
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
                    print(self.dataframe.head(n))
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
            self.display_header()
            print(colored(self.term_font.renderText("Data Inspection"), 'light_blue'))
            
    def explore_data(self):
        """Explore the dataset with statistical analysis"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_header()
        print(colored(self.term_font.renderText("Data Exploration"), 'light_blue'))
        
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
            self.display_header()
            print(colored(self.term_font.renderText("Data Exploration"), 'light_blue'))
            
    def visualize_data(self):
        """Visualize the dataset"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_header()
        print(colored(self.term_font.renderText("Data Visualization"), 'light_blue'))
        
        # visualization options
        while True:
            print("\nVisualization Options:")
            print("1. Plot histograms for numerical features")
            print("2. Plot scatter plots")
            print("3. Back to main menu")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                self.data_processor.plot_histogram()
                
            elif choice == '2':
                print("\nNote: Scatter plots may generate many plots for datasets with many numerical features.")
                confirm = input("Do you want to continue? (y/n): ")
                if confirm.lower() == 'y':
                    self.data_processor.scatter_plot()
                
            elif choice == '3':
                break
                
            else:
                print(colored("Invalid choice. Please try again.", 'red'))
                
            input("\nPress Enter to continue...")
            self.display_header()
            print(colored(self.term_font.renderText("Data Visualization"), 'light_blue'))
            
    def impute_missing_values(self):
        """Handle missing values in the dataset"""
        if self.dataframe is None:
            self.display_no_data_message()
            return
            
        self.display_header()
        print(colored(self.term_font.renderText("Missing Value Imputation"), 'light_blue'))
        
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
            
        self.display_header()
        print(colored(self.term_font.renderText("Feature Standardization"), 'light_blue'))
        
        # Call feature_scaling method
        self.data_processor.feature_scaling()
        
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
            
        self.display_header()
        print(colored(self.term_font.renderText("Export Data"), 'light_blue'))
        
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
        
        self.display_header()
        print(colored(self.term_font.renderText("AutoML Model Selection"), 'light_blue'))
        
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
            #self.capture_console_output(self)
            self.display_header()
            
            # Main menu
            print("\nMain Menu:")
            print("1. Load Dataset")
            print("2. Inspect Data")
            print("3. Explore Data")
            print("4. Visualize Data")
            print("5. Impute Missing Values")
            print("6. Standardize Features")
            print("7. Export Data")
            print("8. AutoML (Train & Evaluate Models)")
            print("9. Exit Prepup")
            
            choice = input("\nEnter your choice (1-9): ")
            
            if choice == '1':
                self.load_data()
            elif choice == '2':
                self.inspect_data()
            elif choice == '3':
                self.explore_data()
            elif choice == '4':
                self.visualize_data()
            elif choice == '5':
                self.impute_missing_values()
            elif choice == '6':
                self.standardize_features()
            elif choice == '7':
                self.export_data()
            elif choice == '8':
                self.automl()
            else: 
                print(colored("\nThank you for using Prepup!", 'green'))
                print(colored("Exiting...", 'light_blue'))
                sys.exit(0)
            


def main():
    args = parse_args()
    
    # Check if running in interactive mode
    if args.interactive:
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
    else:
        # Traditional CLI mode with flags
        if args.file:
            try:
                df = load_file(args.file)
                run_cli_mode(args, df)
            except Exception as e:
                print(colored(f"Error: {str(e)}", 'red'))
                sys.exit(1)
        else:
            # No file provided for CLI mode
            print(colored("Error: No dataset file provided. Please specify a file path or use -interactive mode.", 'red'))
            sys.exit(1)
                   

def main():
    # Simplified argument parsing
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