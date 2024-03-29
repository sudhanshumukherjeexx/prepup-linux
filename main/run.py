#!/usr/bin/env python3


"""Main module.

   author : "Neokai"
"""
import argparse
import pandas as pd
from main.common import Prepup
from termcolor import colored
from pyfiglet import Figlet
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Prepup is a free open-source package that lets you inspect, expdore, visualize, and perform pre-processing tasks on datasets in your computers terminal.")
    parser.add_argument('file',type=str, help='Dataset file')
    parser.add_argument('-inspect', action='store_true', help='Observe the dataset and its Features.')
    parser.add_argument('-explore', action='store_true', help='Explore Dataset.')
    parser.add_argument('-visualize', action='store_true', help='Visualize Feature Distribution.')
    parser.add_argument('-impute', action='store_true', help='Impute Missing values.')
    parser.add_argument('-standardize', action='store_true', help='Standardize Feature Columns.')
    return parser.parse_args()

def load_file(file_path):
    # Check file extension
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Invalid file format. Only CSV and Excel files are supported.")


def main():
    args = parse_args()
    intro = Figlet(font='big')
    print(colored(intro.renderText("PREPUP !"), 'green'))
    time.sleep(0.5)
    df = load_file(args.file)
    
    if args.standardize:
        start = time.time() 
        crafter = Prepup(df)
        
        h_6 = Figlet(font='term')
        print(colored(h_6.renderText("\nFeature Scaling in Progress..."),'light_blue'))
        time.sleep(0.2)
        crafter.feature_scaling()
        
        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for -normalize flag..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
    elif args.explore:
        start = time.time() 

        crafter = Prepup(df)
        h_1 = Figlet(font='term')
        print(colored(h_1.renderText("Features present in the Dataset..."), 'light_blue')) 
        time.sleep(0.5)
        print(crafter.features_available())
        
        h_2 = Figlet(font='term')
        print(colored(h_2.renderText("\nDatatype of each Feature..."), 'light_blue'))
        time.sleep(0.5)
        print(crafter.dtype_features())

        h_5 = Figlet(font='term')
        print(colored(h_5.renderText("\nFeature Correlation..."),'light_blue'))
        time.sleep(0.5)
        crafter.correlation_n()

        h_6 = Figlet(font='term')
        print(colored(h_6.renderText("\nCheck for Normal Distribution..."),'light_blue'))
        time.sleep(0.2)
        crafter.check_nomral_distrubution()

        h_9 = Figlet(font='term')
        print(colored(h_9.renderText("\nDetecting Outliers..."),'light_blue'))
        crafter.find_outliers()

        h_4 = Figlet(font='term')
        print(colored(h_4.renderText("\nFeatures: Skewness Present in the Dataset..."),'light_blue'))
        time.sleep(0.5)
        crafter.skewness()

        h_4 = Figlet(font='term')
        print(colored(h_4.renderText("\nFeatures: Kurtosis Present in the Dataset..."),'light_blue'))
        time.sleep(0.5)
        crafter.kurtosis()

        h_6 = Figlet(font='term')
        print(colored(h_6.renderText("\nCheck for Imabalanced Dataset..."),'light_blue'))
        time.sleep(0.2)
        crafter.imbalanced_dataset()

        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for -explore flag..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
    elif args.visualize:
        start = time.time() 
        crafter = Prepup(df)
        
        h_6 = Figlet(font='term')
        print(colored(h_6.renderText("\nFeature Distribution..."),'light_blue'))
        time.sleep(0.2)
        crafter.plot_histogram()

        # h_6 = Figlet(font='term')
        # print(colored(h_6.renderText("\nRelationship visualized between two variables (Pair-Plot)..."),'light_blue'))
        # crafter =  Prepup(df)
        # crafter.scatter_plot()

        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for -visualize flag..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
    elif args.impute:
        start = time.time()
        crafter =  Prepup(df)

        h_6 = Figlet(font='term')
        print(colored(h_6.renderText("\nHandle Missing Data..."),'light_blue'))
        time.sleep(0.2)
        crafter.handle_missing_values()

        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for -impute flag..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
    elif args.inspect:
        start = time.time() 
        crafter = Prepup(df)
        h_1 = Figlet(font='term')
        print(colored(h_1.renderText("Features Present in the Dataset..."), 'light_blue')) 
        time.sleep(0.5)
        print(crafter.features_available())
        
        h_2 = Figlet(font='term')
        print(colored(h_2.renderText("\nFeature's Datatype..."), 'light_blue'))
        time.sleep(0.5)
        print(crafter.dtype_features())

        h_3 = Figlet(font='term')
        print(colored(h_3.renderText("\nShape of Data..."),'light_blue'))
        time.sleep(0.5)
        print(crafter.shape_data())
                
        h_4 = Figlet(font='term')
        print(colored(h_4.renderText("\nFeatures: Missing values count..."),'light_blue'))
        time.sleep(0.5)
        crafter.missing_plot()
        
        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for loading and inspecting data..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
    else:
        start = time.time() 
        crafter = Prepup(df)
        h_1 = Figlet(font='term')
        print(colored(h_1.renderText("Features Present in the Dataset..."), 'light_blue')) 
        time.sleep(0.5)
        print(crafter.features_available())
        
        h_2 = Figlet(font='term')
        print(colored(h_2.renderText("\nFeature's Datatype..."), 'light_blue'))
        time.sleep(0.5)
        print(crafter.dtype_features())

        h_3 = Figlet(font='term')
        print(colored(h_3.renderText("\nShape of Data..."),'light_blue'))
        time.sleep(0.5)
        print(crafter.shape_data())
                
        h_4 = Figlet(font='term')
        print(colored(h_4.renderText("\nFeatures: Missing values count..."),'light_blue'))
        time.sleep(0.5)
        crafter.missing_plot()
        
        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for loading and inspecting data..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
                   

if __name__ == '__main__':
    main()