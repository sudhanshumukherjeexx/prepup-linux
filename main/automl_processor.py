import pandas as pd
import numpy as np
import time
from termcolor import colored
from typing import Tuple, Optional

class AutoMLProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize AutoML Processor with input dataframe
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            Input dataframe to be processed and analyzed
        """
        self.original_df = dataframe.copy()
        self.preprocessed_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def _preprocess_data(self, target_column: str, task_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Comprehensive data preprocessing method with robust NaN handling
        """
        # Create a copy of the original dataframe
        df = self.original_df.copy()
        
        # Validate target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        # Step 1: Handle missing values
        print("\n🔍 Preprocessing: Missing Value Analysis")
        
        # Check initial missing values
        initial_missing = df.isnull().sum()
        print("Initial Missing Values:")
        print(initial_missing[initial_missing > 0])
        
        # For regression, ensure target is numeric
        if task_type == 'regression':
            try:
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
            except Exception as e:
                raise ValueError(f"Cannot convert target column to numeric: {e}")
        
        # Remove rows with missing target values
        initial_rows = len(df)
        df = df.dropna(subset=[target_column])
        print(f"Removed {initial_rows - len(df)} rows with missing target values")
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(exclude=['number']).columns
        
        # Impute missing values using sklearn's SimpleImputer
        from sklearn.impute import SimpleImputer
        import numpy as np
        
        # Impute numeric columns
        numeric_cols_to_impute = [col for col in numeric_cols if col != target_column]
        if numeric_cols_to_impute:
            numeric_imputer = SimpleImputer(strategy='median')
            df[numeric_cols_to_impute] = numeric_imputer.fit_transform(df[numeric_cols_to_impute])
        
        # Impute categorical columns
        cat_cols_to_impute = [col for col in cat_cols if col != target_column]
        if cat_cols_to_impute:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[cat_cols_to_impute] = cat_imputer.fit_transform(df[cat_cols_to_impute])
        
        # Encode categorical variables
        print("\n🔍 Preprocessing: Categorical Variable Encoding")
        cat_cols_to_process = [col for col in cat_cols if col != target_column]
        
        for col in cat_cols_to_process:
            # Get unique values
            unique_vals = df[col].nunique()
            
            if unique_vals <= 10:
                # One-hot encode for few unique values
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                print(f"One-hot encoded: {col} (unique values: {unique_vals})")
            elif unique_vals <= 50:
                # Label encode for moderate number of unique values
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                print(f"Label encoded: {col} (unique values: {unique_vals})")
            else:
                # Drop columns with too many unique values
                df = df.drop(columns=[col])
                print(f"Dropped: {col} (too many unique values: {unique_vals})")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Feature scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Verify no missing values remain
        print("\n🔍 Missing Values After Preprocessing:")
        print("Features:", X_scaled.isnull().sum().sum())
        print("Target:", y.isnull().sum())
        
        # Print final preprocessing summary
        print("\n📊 Preprocessing Summary:")
        print(f"Final dataset shape: {X_scaled.shape[0]} rows, {X_scaled.shape[1]} features")
        
        return X_scaled, y


    def _split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> None:
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        test_size : float, optional
            Proportion of data to use for testing (default 0.2)
        """
        from sklearn.model_selection import train_test_split
        
        # Stratify only for classification with few classes
        stratify = y if len(y.unique()) < 10 else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        
        print("\n🔀 Data Split:")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
    
    def run_automl(self, target_column: str, task_type: str = 'classification') -> pd.DataFrame:
        """
        Main AutoML method to run machine learning pipeline
        
        Parameters:
        -----------
        target_column : str
            Name of the target column
        task_type : str, optional
            Type of machine learning task ('classification' or 'regression')
        
        Returns:
        --------
        pd.DataFrame
            Results of model evaluations
        """
        # Validate task type
        if task_type not in ['classification', 'regression']:
            raise ValueError("Task type must be 'classification' or 'regression'")
        
        # Start timing
        start_time = time.time()
        
        # Print header
        print(colored("\n🤖 AutoML Model Selection 🤖", 'light_blue'))
        print(f"Task Type: {task_type.capitalize()}")
        print(f"Target Column: {target_column}")
        
        # Preprocess data
        X, y = self._preprocess_data(target_column, task_type)
        
        # Split data
        self._split_data(X, y)
        
        # Model selection based on task type
        if task_type == 'classification':
            results = self._run_classification_models()
        else:
            results = self._run_regression_models()
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        print(colored(f"\n✅ Analysis completed in {execution_time:.2f} seconds!", 'green'))
        
        return results
    
    def _run_classification_models(self) -> pd.DataFrame:
        """
        Run multiple classification models and evaluate
        
        Returns:
        --------
        pd.DataFrame
            Comparative results of classification models
        """
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                                     f1_score, roc_auc_score, precision_score, recall_score)
        
        # List of classifiers to test
        classifiers = [
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('Extra Trees', ExtraTreesClassifier(n_estimators=100, random_state=42)),
            ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
            ('Decision Tree', DecisionTreeClassifier(random_state=42)),
            ('KNN', KNeighborsClassifier()),
            ('SVM', SVC(probability=True, random_state=42)),
            ('Naive Bayes', GaussianNB()),
            ('AdaBoost', AdaBoostClassifier(random_state=42))
        ]
        
        # Try to import optional libraries
        try:
            import xgboost as xgb
            classifiers.append(('XGBoost', xgb.XGBClassifier(random_state=42, verbosity=0)))
        except ImportError:
            print("XGBoost not available")
        
        try:
            import lightgbm as lgb
            classifiers.append(('LightGBM', lgb.LGBMClassifier(random_state=42, verbose=-1)))
        except ImportError:
            print("LightGBM not available")
        
        # Results storage
        results = []
        
        # Evaluate each classifier
        for name, model in classifiers:
            try:
                print(f"\n🔍 Evaluating {name} Classifier...")
                
                # Train the model
                model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_pred = model.predict(self.X_test)
                
                # Probability predictions (for AUC)
                try:
                    y_proba = model.predict_proba(self.X_test)
                except:
                    y_proba = None
                
                # Compute metrics
                metrics = {
                    'Accuracy': accuracy_score(self.y_test, y_pred),
                    'Balanced Accuracy': balanced_accuracy_score(self.y_test, y_pred),
                    'F1 Score': f1_score(self.y_test, y_pred, average='weighted'),
                    'Precision': precision_score(self.y_test, y_pred, average='weighted'),
                    'Recall': recall_score(self.y_test, y_pred, average='weighted')
                }
                
                # AUC score (if possible)
                if y_proba is not None:
                    try:
                        # Multiclass or binary classification
                        if y_proba.shape[1] == 2:
                            metrics['ROC AUC'] = roc_auc_score(self.y_test, y_proba[:, 1])
                        else:
                            metrics['ROC AUC'] = roc_auc_score(self.y_test, y_proba, multi_class='ovr', average='weighted')
                    except Exception:
                        metrics['ROC AUC'] = np.nan
                
                # Store results
                result_entry = {'Model': name, **metrics}
                results.append(result_entry)
                
                print(f"✓ Completed - Balanced Accuracy: {metrics['Balanced Accuracy']:.4f}")
            
            except Exception as e:
                print(f"❌ Error with {name}: {str(e)}")
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Balanced Accuracy', ascending=False).set_index('Model')
        
        # Display results
        print("\n📊 Classification Model Comparison:")
        print(results_df)
        
        return results_df
    
    def _run_regression_models(self) -> pd.DataFrame:
        """
        Run multiple regression models and evaluate
        
        Returns:
        --------
        pd.DataFrame
            Comparative results of regression models
        """
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR
        from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                                     r2_score, mean_absolute_percentage_error)
        
        # List of regressors to test
        regressors = [
            ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
            ('Linear Regression', LinearRegression()),
            ('Ridge Regression', Ridge(random_state=42)),
            ('Lasso Regression', Lasso(random_state=42)),
            ('Elastic Net', ElasticNet(random_state=42)),
            ('Decision Tree', DecisionTreeRegressor(random_state=42)),
            ('KNN', KNeighborsRegressor()),
            ('SVR', SVR()),
            ('AdaBoost', AdaBoostRegressor(random_state=42))
        ]
        
        # Try to import optional libraries
        try:
            import xgboost as xgb
            regressors.append(('XGBoost', xgb.XGBRegressor(random_state=42, verbosity=0)))
        except ImportError:
            print("XGBoost not available")
        
        try:
            import lightgbm as lgb
            regressors.append(('LightGBM', lgb.LGBMRegressor(random_state=42, verbose=-1)))
        except ImportError:
            print("LightGBM not available")
        
        # Results storage
        results = []
        
        # Evaluate each regressor
        for name, model in regressors:
            try:
                print(f"\n🔍 Evaluating {name} Regressor...")
                
                # Train the model
                model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_pred = model.predict(self.X_test)
                
                # Compute metrics
                metrics = {
                    'R²': r2_score(self.y_test, y_pred),
                    'MSE': mean_squared_error(self.y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                    'MAE': mean_absolute_error(self.y_test, y_pred),
                    'MAPE': mean_absolute_percentage_error(self.y_test, y_pred)
                }
                
                # Calculate adjusted R²
                n = self.X_test.shape[0]
                p = self.X_test.shape[1]
                metrics['Adjusted R²'] = 1 - (1 - metrics['R²']) * (n - 1) / (n - p - 1)
                
                # Store results
                result_entry = {'Model': name, **metrics}
                results.append(result_entry)
                
                print(f"✓ Completed - R²: {metrics['R²']:.4f}")
            
            except Exception as e:
                print(f"❌ Error with {name}: {str(e)}")
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Adjusted R²', ascending=False).set_index('Model')
        
        # Display results
        print("\n📊 Regression Model Comparison:")
        print(results_df)
        
        return results_df

# Example usage function
def example_automl_usage(csv_path: str, target_column: str, task_type: str = 'classification'):
    """
    Demonstrate AutoML usage with a CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file
    target_column : str
        Name of the target column
    task_type : str, optional
        Type of machine learning task ('classification' or 'regression')
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Initialize AutoML Processor
    automl = AutoMLProcessor(df)
    
    try:
        # Run AutoML
        results = automl.run_automl(target_column, task_type)
        
        # Optionally, save results to a CSV
        results_path = f"{task_type}_automl_results.csv"
        results.to_csv(results_path)
        print(f"\n💾 Results saved to {results_path}")
    
    except Exception as e:
        print(f"AutoML processing failed: {e}")

# # Optional main block for direct script execution
# if __name__ == "__main__":
#     import sys
    
#     # Check if correct number of arguments is provided
#     if len(sys.argv) < 3:
#         print("Usage: python script.py <csv_path> <target_column> [task_type]")
#         print("Example: python script.py data.csv price regression")
#         sys.exit(1)
    
#     # Parse command line arguments
#     csv_path = sys.argv[1]
#     target_column = sys.argv[2]
#     task_type = sys.argv[3] if len(sys.argv) > 3 else 'classification'
    
#     # Run example usage
#     example_automl_usage(csv_path, target_column, task_type)