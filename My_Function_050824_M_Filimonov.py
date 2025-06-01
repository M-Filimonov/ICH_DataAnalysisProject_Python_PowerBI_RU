import os
import numpy as np
import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sns

from typing import NoReturn
from colorama import Fore, Back, Style
from datetime import time, timedelta


from datetime import time, timedelta
from collections import Counter
  
from tqdm import tqdm  # progress bar 
import re                  

#**********************************************************************************      
#*                                                                                *      
#*                         Functions used in data cleaning                        *      
#*                                                                                *      
#**********************************************************************************      
####################################################################################### df_pre_clean()
def df_pre_clean(df_name: DataFrame) -> NoReturn:
    """
    DataFrame pre-cleaning function:
    Performs the following operations:
    - Removes duplicate rows.
    - Replaces spaces in column names with underscores.
    - Removes columns where all values are NaN.
    - Removes rows where all values are NaN.
    
    Arguments:
    - df_name (DataFrame): The original DataFrame to be cleaned.

    Returns:
    - None: All changes are applied directly to the provided DataFrame.
    """    
    print(Back.YELLOW + f"DataFrame '{df_name.name}' info:")
    print(Style.RESET_ALL)
    print(f"ROWs number: {df_name.shape[0]}")      
    print(f"COLUMNs number: {df_name.shape[1]}")    

    # Remove rows where all values are NaN
    rows_before = df_name.shape[0]
    df_name.dropna(axis=0, how='all', inplace=True)
    rows_after = df_name.shape[0]
    rows_removed = rows_before - rows_after
    if rows_removed > 0:
        print(Fore.BLUE + f"Removed: {rows_removed} rows, where all values are NaN")
    else:
        print(Fore.BLUE + f"No ROWs, where all values are NaN found")

    # Calculate and remome duplicate rows
    num_duplicates = df_name.duplicated().sum()
    if num_duplicates > 0:
        print(Fore.MAGENTA + f"Number of duplicate rows: {num_duplicates}")
        df_name.drop_duplicates(inplace=True)
        print(Fore.MAGENTA + "All duplicates ROWs have been removed.")
    else:
        print(Fore.MAGENTA + "No duplicated ROWs found.")
        
    # Remove columns where all values are NaN
    initial_columns = set(df_name.columns)
    df_name.dropna(axis=1, how='all', inplace=True)
    removed_columns = initial_columns - set(df_name.columns)
    if removed_columns:
        print(Fore.YELLOW + "COLUMNs with all NaN values removed:")
        for column in removed_columns:
            print(f"- {column}")
    else:
        print(Fore.YELLOW +"No COLUMNs with all NaN values were found.\n")
    print(Style.RESET_ALL)
    
    # Replace spaces in column names with underscores
    print(f"Old columns names: {list(df_name.columns)}\n")
    df_name.columns = df_name.columns.str.replace(' ', '_')
    print(f"New columns names: {list(df_name.columns)}\n")

######################################################################################## my_columns_describe()
def my_columns_describe(df_name: DataFrame) -> None:
    """
    Function to analyze DataFrame columns based on their data types,
    display statistical summaries, and provide information about unique values and missing data.
    
    Args:
        df_name (DataFrame): Input DataFrame to be analyzed.
        
    Returns:
        None
    """
    print("\nDataFrame info:")
    print(df_name.info())
    
    print("\nDataFrame describe:")
    print(df_name.isnull().sum())     
    
    #--------------------------------------------------------OBJECT columns
    print(Back.YELLOW + "\nList of OBJECT-type columns" + Style.RESET_ALL)
    object_columns = df_name.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        for col in object_columns:
            
            # print column name
            print(Back.WHITE + f"Column: '{col}'" + Style.RESET_ALL)
                        
            # check for data type homogeneity
            check_type = df_name[col].dropna().apply(type).value_counts()
            if len(check_type) > 1:
                print(Fore.RED + "Attention! The data in the column is not homogeneous:"  + Style.RESET_ALL)
                print(check_type)                
                non_string_rows = df_name[~df_name[col].apply(lambda x: isinstance(x, str) or pd.isna(x))]
                if len(non_string_rows) > 1:
                    # Generate list of dictionaries with non-str values
                    non_str_values = [{index: value} for index, value in non_string_rows[col].items()]
                    # Output the list of dictionaries
                    if len(non_str_values) <= 100:
                        print(Fore.YELLOW + "Non-string values with their row indices:" + Style.RESET_ALL)
                        print(non_str_values)  
                    else:
                        print("Total Non-string= ",len(non_str_values))  
                        
            #  checking unique data and its distribution  
            list_unique = df_name[col].unique()
            print(Fore.GREEN + f"Unique values: {len(list_unique)}" + Style.RESET_ALL)
            if len(list_unique) > 200:
                 # nan values count and percentage
                nan_count = df_name[col].isna().sum()
                count_rec = len(df_name[col])
                if count_rec > 0:
                    print(Fore.MAGENTA + f"Missing values (NaN): {nan_count} | {nan_count/count_rec*100:.0f}%")                
                else:
                    print(Fore.RED+"Column is empty, percentage of missing values cannot be calculated." + Style.RESET_ALL)
                print(Style.RESET_ALL)    
                continue
            counts = df_name[df_name[col].isin(list_unique)][col].value_counts()
            print("Values distribution:")
            print(counts)            
            
            # nan values count and percentage
            nan_count = df_name[col].isna().sum()
            count_rec = len(df_name[col])
            if count_rec > 0:
                print(Fore.MAGENTA + f"Missing values (NaN): {nan_count} | {nan_count/count_rec*100:.0f}%")                
            else:
                print(Fore.RED+"Column is empty, percentage of missing values cannot be calculated." + Style.RESET_ALL)
            print(Style.RESET_ALL)    
    else:
        print(Fore.RED + "No object-type columns")        
        print(Style.RESET_ALL)    

 #--------------------------------------------------------DataTime columns
    print(Back.YELLOW + "\nList of DataTime-type columns" + Style.RESET_ALL)
    dt_columns = df_name.select_dtypes(include=['datetime']).columns.tolist()
    if dt_columns:
        for col in dt_columns:
            
            # print column name
            print(Back.WHITE + f"Column: '{col}'" + Style.RESET_ALL)
                        
            # Check for data type homogeneity
            check_type = df_name[col].dropna().apply(type).value_counts()
            if len(check_type) > 1:
                print(Fore.RED + "Attention! The data in the column is not homogeneous:" + Style.RESET_ALL)
                print(check_type)
    
                # Filter rows that are not of datetime type (and are not NaN)
                non_dt_rows = df_name[~df_name[col].apply(lambda x: isinstance(x, pd.Timestamp) or pd.isna(x))]
    
                if len(non_dt_rows) > 0:
                    # Generate list of dictionaries with non-datetime values
                    non_dt_values = [{index: value} for index, value in non_dt_rows[col].items()]
        
                    # Output the list of dictionaries
                    if len(non_dt_values) <= 100:
                        print(Fore.YELLOW + "Non-datetime values with their row indices:" + Style.RESET_ALL)
                        print(non_dt_values)
                    else:
                        display(non_dt_rows)
            
            #  checking unique data and its distribution  
            list_unique = df_name[col].unique()
            print(Fore.GREEN + f"Unique values: {len(list_unique)}" + Style.RESET_ALL)
            if len(list_unique) > 200:
                 # nan values count and percentage
                nan_count = df_name[col].isna().sum()
                count_rec = len(df_name[col])
                if count_rec > 0:
                    print(Fore.MAGENTA + f"Missing values (NaN): {nan_count} | {nan_count/count_rec*100:.0f}%")                
                else:
                    print(Fore.RED+"Column is empty, percentage of missing values cannot be calculated." + Style.RESET_ALL)
                print(Style.RESET_ALL)    
                continue
            counts = df_name[df_name[col].isin(list_unique)][col].value_counts()
            print("Values distribution:")
            print(counts)            
            
            # nan values count and percentage
            nan_count = df_name[col].isna().sum()
            count_rec = len(df_name[col])
            if count_rec > 0:
                print(Fore.MAGENTA + f"Missing values (NaN): {nan_count} | {nan_count/count_rec*100:.0f}%")                
            else:
                print(Fore.RED+"Column is empty, percentage of missing values cannot be calculated." + Style.RESET_ALL)
            print(Style.RESET_ALL)    
    else:
        print(Fore.RED + "No DataTime -type columns")        
        print(Style.RESET_ALL)    
    
    #--------------------------------------------------------FLOAT columns
    print(Back.BLUE+"List of FLOAT-type columns:"+ Style.RESET_ALL)
    
    float_columns = df_name.select_dtypes(include=['float']).columns.tolist()
    if float_columns:
        for col in float_columns:            
            print(Back.WHITE + f"Column: '{col}':" + Style.RESET_ALL)    
                       
            print('Column Statistic:')
            print(df_name[col].describe())

            #  checking unique values
            print(Fore.GREEN+'Column unique values:')
            unique_values = df_name[col].unique()
            if len(unique_values) <= 200:
                formatted_values = [f"{value:.2f}" if isinstance(value, float|int) else value for value in unique_values]
                print(formatted_values)
                print(Style.RESET_ALL)
            else:
                print(Fore.GREEN + f"Unique values count: {len(list_unique)}" + Style.RESET_ALL)
            # nan values count and percentage
            nan_count = df_name[col].isna().sum()
            count_rec = len(df_name[col])
            if count_rec > 0:
                print(Fore.MAGENTA + f"Missing values (NaN): {nan_count} | {nan_count/count_rec*100:.0f}%" + Style.RESET_ALL)                
            else:
                print(Fore.RED+"Column is empty, percentage of missing values cannot be calculated.")
            print(Style.RESET_ALL)
    else:
        print(Fore.RED+"No float-type columns")    
        print(Style.RESET_ALL)
    #--------------------------------------------------------INTEGER columns
    print(Back.CYAN + "List of INTEGER-type columns:" + Style.RESET_ALL)
    int_columns = df_name.select_dtypes(include=['int', 'int8', 'int16', 'int32', 'int64']).columns.tolist()
    if int_columns:
        for col in int_columns:
            print(Back.WHITE + f"Column: '{col}':" +Style.RESET_ALL)
                        
            print('Column Statistic:')
            print(df_name[col].describe())

            #  checking unique values
            print(Fore.GREEN+'Column unique values:')
            unique_values = df_name[col].unique()
            if len(unique_values) <= 200:
                formatted_values = [f"{value:.2f}" if isinstance(value, float|int) else value for value in unique_values]
                print(formatted_values)
                print(Style.RESET_ALL)
            else:
                print(Fore.GREEN + f"Unique values count: {len(list_unique)}" + Style.RESET_ALL)
            
            # nan values count and percentage
            nan_count = df_name[col].isna().sum()
            count_rec = len(df_name[col])
            if count_rec > 0:
                print(Fore.MAGENTA + f"Missing values (NaN): {nan_count} | {nan_count/count_rec*100:.0f}%")
                print(Style.RESET_ALL)
            else:
                print(Fore.RED + "Column is empty, percentage of missing values cannot be calculated.")            
                print(Style.RESET_ALL)
    else:
        print(Fore.RED + "No integer-type columns") 
        print(Style.RESET_ALL)    
        
    #--------------------------------------------------------BOOLEAN columns
    print(Back.GREEN + "List of BOOL-type columns:" + Style.RESET_ALL)
    
    bool_columns: list[str] = df_name.select_dtypes(include=['bool']).columns.tolist()
    if bool_columns:
        for col in bool_columns:
            
            print(Back.WHITE + f"Column: '{col}'" + Style.RESET_ALL)
                        
            # check for data type homogeneity
            check_type = df_name[col].apply(type).value_counts()
            if len(check_type) > 1:
                print(Fore.RED + "Attention! The data in the column is not homogeneous:" + Style.RESET_ALL)
                print(check_type)
                non_bool_rows = df_name[~df_name[col].apply(lambda x: isinstance(x, bool) or pd.isna(x))]
                if len(non_bool_rows) > 1:
                    # Generate list of dictionaries with non-bool values
                    non_bool_values = [{index: value} for index, value in non_bool_rows[col].items()]
                    # Output the list of dictionaries
                    if len(non_bool_values) <= 100:
                        print(Fore.YELLOW + "Non-bool values with their row indices:" + Style.RESET_ALL)
                        print(non_bool_values)  
                    else:
                        print("Total Non-bool values= ",len(non_str_values))  

            #  checking unique values
            print(Fore.GREEN+'Column unique values:')
            unique_values = df_name[col].unique()
            if len(unique_values) <= 200:
                formatted_values = [f"{value:.2f}" if isinstance(value, bool) else value for value in unique_values]
                print(formatted_values)
                print(Style.RESET_ALL)
            else:
                print(Fore.GREEN + f"Unique values count: {len(list_unique)}" + Style.RESET_ALL)
            
            # nan values count and percentage
            nan_count = df_name[col].isna().sum()
            count_rec = len(df_name[col])
            if count_rec > 0:
                print(Fore.MAGENTA + f"Missing values (NaN): {nan_count} | {nan_count/count_rec*100:.0f}%")
                print(Style.RESET_ALL)
            else:
                print(Fore.RED + "Column is empty, percentage of missing values cannot be calculated.")            
                print(Style.RESET_ALL)
    else:
        print(Fore.RED+"No boolean-type columns")        
        print(Style.RESET_ALL)    


####################################################################################### my_df_info()
def my_df_info(df_name: DataFrame, name=None) -> None:
    """
    Displays detailed information about a DataFrame, including its size, 
    data types, and the count of missing values.

    Args:
        df_name (DataFrame): The DataFrame to analyze.

    Returns:
        None: This function does not return anything, but prints the DataFrame information.
    """
    
    pd.reset_option("display.max_rows") 
    
    # Display size of DataFrame
    print(f"'{name}' DataFrame size (row, column): {df_name.shape}")
    
    # Display data types
    print(Fore.BLUE + "DataTypes info:" + Style.RESET_ALL)
    print(df_name.dtypes)
    
    # Display number of missing values
    print(Fore.BLUE + "Number of missing values:" + Style.RESET_ALL)
    print(df_name.isnull().sum())


    
#**********************************************************************************      
#*                                                                                *      
#*                Functions used in data discribing and analysis                  *      
#*                                                                                *      
#**********************************************************************************      

###############################################################  Analyzes the numeric fields of a dataframe

def num_fields_analyze(df, numeric_fields=None, log_fields=None):
    '''
    Analyzes the numeric fields of a dataframe:
    - Outputs the mean, median, mode, and range for the specified numeric columns.
    - Logarithms the specified fields if the log_fields list is passed.
    - Constructs a histogram and boxplot for each numeric field (before and after logarithmization, if applicable).
    
    Parameters:
    df (pd.DataFrame): The source dataframe.
    numeric_fields (list, optional): List of numeric fields to analyze. If None, processes all numeric fields.
    log_fields (list, optional): List of fields to logarithmize. If None, no logarithmization is performed.
    '''

    # Define the list of numeric columns, if not specified
    if numeric_fields is None:
        #numeric_fields = df.select_dtypes(include=[np.number]).columns.tolist()
        #numeric_fields = df.select_dtypes(include=['int','Int8','float']).columns.tolist()
        numeric_fields = df.select_dtypes(include=['number']).columns.tolist()
        

    #Check which of the specified fields are subject to logarithmization
    if log_fields is None:
        log_fields = []
     
    print(Back.YELLOW + "Statistical characteristics of number fields:\n" + Style.RESET_ALL)
    for col in numeric_fields:
        max_val = df[col].max()
        min_val = df[col].min()
        mean_val = df[col].mean()
        std_val = df[col].std()
        median_val = df[col].median()
        mode_val = df[col].mode()[0]  # Берем первое значение моды
        range_val = df[col].max() - df[col].min()
        
        print(Fore.BLUE + f"{col}:" + Style.RESET_ALL)
        print(f" Max: {max_val:.2f}")
        print(f" Min: {min_val:.2f}")
        print(f" Mean: {mean_val:.2f}")
        print(f" std: {std_val:.2f}")
        print(f" Median: {median_val:.2f}")
        print(f" Mode: {mode_val:.2f}")
        print(f" Range: {range_val:.2f}")
        print("-" * 40)

    # Logarithmize the desired columns
    for col in log_fields:
        if col in numeric_fields:  # Убеждаемся, что поле присутствует в числовых
            df[col + "_log"] = np.log1p(df[col])

    # Data visualization
    for col in numeric_fields:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

       # Histogram of value distribution
        sns.histplot(df[col], bins=30, kde=True, ax=axes[0], color="blue")
        axes[0].set_title(f'Histogram of {col}')
        
        # Boxplot
        sns.boxplot(x=df[col], ax=axes[1], color="orange")
        axes[1].set_title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.show()

        # Если поле логарифмировано, строим его графики тоже
        if col in log_fields:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Гистограмма логарифмированных данных
            sns.histplot(df[col + "_log"], bins=30, kde=True, ax=axes[0], color="green")
            axes[0].set_title(f'Log Histogram of {col}')
            
            # Boxplot логарифмированных данных
            sns.boxplot(x=df[col + "_log"], ax=axes[1], color="yellow")
            axes[1].set_title(f'Log Boxplot of {col}')
            
            plt.tight_layout()
            plt.show()


####################################################################### analyzing categorical fields
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_categorical_distribution(df, columns=None):
    """
    Function for analyzing categorical variables in a DataFrame:
    - Prints the distribution of values in each categorical column with percentages.
    - Builds countplot for each categorical variable, adding numerical labels and percentages.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list, optional): List of columns to process. If None, all categorical and object-type columns are processed.
    """

    # Select categorical columns
    all_cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # If columns are None, process all categorical columns
    cat_cols = columns if columns is not None else all_cat_cols

    # Check if there are any valid columns to process
    if not cat_cols:
        print("⚠ No categorical variables found in the DataFrame!")
        return
    
    print(Back.YELLOW + "Distribution of categorical variables:\n"+ Style.RESET_ALL)
   
    # Define the grid size
    rows = max(1, (len(cat_cols) + 1) // 2)  # Ensure at least one row
    fig, axes = plt.subplots(rows, 2, figsize=(18, 6 * rows))

    # Ensure axes is a 2D array when rows == 1
    if rows == 1:
        axes = [axes]  

    # Total number of records
    total_records = len(df)

    for i, col in enumerate(cat_cols):
        if col not in df.columns:  # Skip invalid columns
            print(f"⚠ Column '{col}' not found in DataFrame, skipping.")
            continue

        value_counts = df[col].value_counts()
        percent_counts = (value_counts / total_records) * 100  # Percentage of total records
        
        # Print distribution table (limit to 30 entries)
        print(Fore.BLUE + f"{col} (Unique values: {len(value_counts)}):"+ Style.RESET_ALL)
        for idx, (val, count, percent) in enumerate(zip(value_counts.index, value_counts.values, percent_counts)):
            if idx >= 30:
                print("  ... (too many categories, showing first 30)")
                break
            print(f"  {val}: {count} records ({percent:.1f}%)")
        print("-" * 40)

        # Select correct subplot
        ax = axes[i // 2][i % 2]  

        # Build countplot
        sns.countplot(ax=ax, x=col, data=df, color='lightblue')

        # Rotate X-axis labels
        ax.tick_params(axis='x', rotation=90)

        # Add numerical labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=10, padding=3, label_type='edge')

        # Remove unnecessary borders, leaving only X-axis
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Remove Y-axis labels and ticks
        ax.set_yticks([])  
        ax.set_ylabel("")  
        ax.set_xlabel(col, fontsize=14)

    # Adjust spacing between plots
    fig.subplots_adjust(hspace=0.9)  

    # Add a title to the entire figure
    fig.suptitle("Distribution of Categorical Variables (Visualisation)", fontsize=16)

    plt.show()


################################## Calculation for each deal:  number of payments transaction (T), average check (AOV) and paid amount (Payd)

def fill_Paid(row: pd.Series) -> pd.Series:
    """
    Calculates 'Paid' (total amount paid), 'T' (number of transactions), and
    'AOV' (average order value) based on payment details.

    Args:
        row (pd.Series): A row of a DataFrame containing payment details.

    Returns:
        pd.Series: A series with calculated 'Paid', 'T', and 'AOV' values.
    """
    
    # Fill missing values ​​with zeros
    months = row["Months_of_study"] if pd.notna(row["Months_of_study"]) else 0
    initial_paid = row["Initial_Amount_Paid"] if pd.notna(row["Initial_Amount_Paid"]) else 0
    offer_total = row["Offer_Total_Amount"] if pd.notna(row["Offer_Total_Amount"]) else 0
    course_duration = row["Course_duration"] if pd.notna(row["Course_duration"]) and row["Course_duration"] > 1 else 1
    
    Paid, T, AOV = 0, 0, 0  

    if months == 0:
        Paid, T, AOV = 0, 0, 0       
    else:    
        if initial_paid == offer_total:
            Paid = round((offer_total / course_duration) * months, 2)
            T = int(months)  
        else:
            Paid = round(initial_paid + ((offer_total - initial_paid) / (course_duration - 1)) * months, 2)
            T = int(months)
        
        AOV = round(Paid / T if T > 0 else 0, 2)

    return pd.Series({"Paid": Paid, "T": T, "AOV": AOV})

