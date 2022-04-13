import pandas as pd
from collections import OrderedDict
import os
from os import listdir
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import base64
import webbrowser
import re
import seaborn as sns
import sys
import warnings
import calendar
import missingno as msno
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def create_report_head(data_df):
    """
    This function creates a short snippet about data consisting data shape(number of rows and columns), memory consumed
    by dataset and sample head of the data.

    Args:
        data_df (pandas DataFrame): A pandas DataFrame which is the data to be profiled
        raw_html_file (list): A list which will contain the HTML format of the data.
    Returns:
        A list with HTML converted info
    """
    report_head_content = []
    file_size = data_df.memory_usage(index=True).sum()
    report_head_content.append('<center><h1 style="font-size:2.5vw;">Exploratory Data Analysis Report</h1></center>')
    text_cursor = '<h1 style="font-size:1.5vw;">Data Shape: Rows = {}, Columns = {}</h1>'.format(
        data_df.shape[0], data_df.shape[1])
    report_head_content.append(text_cursor)
    text_cursor = '<h1 style="font-size:1.5vw;">Data size in memory = {} MB</h1>'.format(
        round(file_size/1000000, 1))
    report_head_content.append(text_cursor)
    # Creating the sample data in HTML
    report_head_content.extend(html_converter(table=data_df.head(10),  heading='Preview'))
    return report_head_content


def html_converter(table=None, image=None, heading=''):
    """
    This function converts the table or image to html object and format it for report.

    Args:
        table (pandas DataFrame): A pandas DataFrame which is to be used fo rreporting
        image (string): It is the name of the image file.
    Returns:
        A list with HTML converted info
    """
    temp_html_content = []
    try:
        if not table.empty:
            html_obj = pd.DataFrame.to_html(table)
    except:
        pass
    if image:
        image_data = base64.b64encode(open(image, 'rb').read()).decode('utf-8').replace('\n', '')
        html_obj = '<img src="data:image/png;base64,{}"s>'.format(image_data)
    temp_html_content.append('<br /><h1 style="font-size:1.5vw;">{}<br /></h1>'.format(heading))
    temp_html_content.append(html_obj)
    temp_html_content.append('<br /><br /><br /><br />')
    return temp_html_content


def html_writer(raw_html_file):
    """
    This function writes the html content in list to a web browser as html file.

    Args:
        raw_html_file (list): It is the list which have all the html converted objects
    Returns:
        Does not return anything
    """
    f = open('EDA_Report.html', 'w')
    f.write(''.join(raw_html_file))
    f.close()
    webbrowser.open_new_tab('EDA_Report.html')


def determine_data_column_types(data_df, date_time_var=None):
    """
    Checks the data type of all the column and also converts the column to date time
    if it is in the format like dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy

    Args:
        data_df (pandas DataFrame): A pandas DataFrame which is the data to be profiled
        date_time_var (list): It is a list having date_time columns
    Returns:
        num_clms: List of all numeric columns
        cat_clms: List of all categorical columns
        bool_clms: List of all boolean columns
        date_time_clms: List of all date time columns
        clm_desc: dictionary with column names and data types
    """
    date_time_clms = []
    clm_desc = {}
    # checking if date_time_var is a list
    if isinstance(date_time_var, list):
        if len(date_time_var) > 0:
            for clm in date_time_var:
                if clm in data_df.columns:
                    date_time_clms.append(clm.strip())
                else:
                    print("The date time column '{}' you entered is not in data".format(date_time_var))
    for clm in data_df.columns:
        if data_df.dtypes[clm] == 'O':
            if len(data_df[clm].unique()) == len(data_df[clm]):
                clm_desc[clm] = 'Text'
            elif _date_time_check(data_df, clm) | (clm in date_time_clms):
                data_df[clm] = pd.to_datetime(data_df[clm], dayfirst=True)
                clm_desc[clm] = 'DateTime'
            else:
                clm_desc[clm] = 'Object'
        elif (data_df.dtypes[clm] == 'int64') | (data_df.dtypes[clm] == 'int32') | \
                (data_df.dtypes[clm] == 'float64'):
            if len(data_df[clm].unique()) < int((5/100) * len(data_df[clm])):
                clm_desc[clm] = 'Object'
                data_df[clm] = data_df[clm].astype('category')
            else:
                clm_desc[clm] = 'Integer'
        elif data_df.dtypes[clm] == 'float64':
            clm_desc[clm] = 'Float'
        elif (data_df.dtypes[clm] == 'datetime64[ns]') | (data_df.dtypes[clm] == '<M8[ns]'):
            clm_desc[clm] = 'DateTime'
        elif data_df.dtypes[clm] == 'bool':
            clm_desc[clm] = 'Object'
        else:
            clm_desc[clm] = str(data_df.dtypes[clm])
    return clm_desc


def _date_time_check(data_df, clm):
    """
    Checks if a column in a data_df is a date type column by checking the format of the value
    like dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        clm (string): The column name
    Returns:
        check (bool): True, if it is a date type column otherwise False
    """
    regex_list = ['(\d+/\d+/\d+)', '(\d+-\d+-\d+)', '(\d+\.\d+\.\d+)']
    print(regex_list)# htd
    check = any(re.match(regex, data_df[clm][~pd.isnull(data_df[clm])].iloc[2])
                for regex in regex_list)
    return check


def calc_summary(data_df,  clm_desc=None, html_report=False):
    """
    This function calculates summary of the data frame

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        clm_desc (dictionary): Contains all column name as 'keys' and their data type as 'values'
        html_report (bool) : True, if report needs to be generated , False otherwise
        raw_html_file (list) : HTML content which eventually gets written
    Returns:
        summary_df, raw_html_file: A tuple, having first element as summary table(pandas data frame)
                                    and html content list if available, otherwise None
    """
    summary_html_content = []
    if not clm_desc:
        clm_desc = determine_data_column_types(data_df)
    num_clms = [clm for clm, dtype in clm_desc.items() if dtype in ['Integer', 'Float']]
    summary_df = data_df[num_clms].describe()
    if html_report:
        summary_df_rounded = summary_df.apply(lambda col: round(col, 2))
        summary_html_content = html_converter(table=summary_df_rounded, heading='Data Summary')
    return summary_df, summary_html_content


def calc_data_type(data_df, clm_desc=None, html_report=False):
    """
    This function calculates data type of all the columns

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        clm_desc (dictionary): Contains all column name as 'keys' and their data type as 'values'
        html_report (bool) : True, if report needs to be generated , False otherwise
        raw_html_file (list) : HTML content which eventually gets written
    Returns:
        data_type_df, raw_html_file: A tuple, having first element as data type table(pandas data frame)
                                    and html content list if available, otherwise None
    """
    data_type_html_content = []
    if not clm_desc:
        clm_desc = determine_data_column_types(data_df)
    column_list = []
    data_type_list = []
    for k,v in clm_desc.items():
        column_list.append(k)
        data_type_list.append(v)
    data_type_df = pd.DataFrame({'Column': column_list, 'Data Type': data_type_list}, index=None)
    if html_report:
        data_type_html_content.extend(html_converter(table=data_type_df, heading='Data Type'))
    return data_type_df, data_type_html_content


def calc_distinct(data_df, plot=False, html_report=False):
    """
    This function calculates distinct percentage of each column of the data_df

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        plot (bool): True, if plot needs to be created. False, otherwise
        html_report (bool) : True, if report needs to be generated , False otherwise
    Returns:
        distinct_value_data_df,distinct_plot,raw_html_file (tuple):
                            distinct_value_data_df - Distinct percentage table
                            distinct_plot - Matplotlib object containing bar chart of distinct percentage
                            raw_html_file - html content list if available
    """
    distinct_html_content = []
    distinct_value_data_df = pd.DataFrame(data_df.apply(lambda col: len(col.unique())))
    distinct_value_data_df.reset_index(inplace=True)
    distinct_value_data_df.columns = ['Column Names', 'Distinct Value']
    distinct_value_data_df['Distinct Percentage'] = \
        distinct_value_data_df['Distinct Value'].apply(lambda x: round((x/len(data_df)*100), 2))
    distinct_value_data_df.reset_index(drop=True, inplace=True)
    if plot:
        plt.gcf().clear()
        distinct_plot = bar_plotter(distinct_value_data_df, x_axis='Column Names',
                                    y_axis='Distinct Percentage')
    else:
        distinct_plot = None
    if html_report:
        distinct_html_content.extend(html_converter(table=distinct_value_data_df,  heading='Distinct Value Table'))
        if plot:
            plt.savefig('plot_distinct.png', bbox_inches='tight')
            distinct_html_content.extend(html_converter(image='plot_distinct.png', heading='Distinct plot'))
    return distinct_value_data_df, distinct_plot, distinct_html_content


def calc_skewness(data_df, plot=False, html_report=False):
    """
    This function calculates skewness of each column of the data_df

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        plot (bool): True, if plot needs to be created. False, otherwise
        html_report (bool) : True, if report needs to be generated , False otherwise
        raw_html_file (list) : HTML content which eventually gets written
    Returns:
        skew_data_df, skew_plot, raw_html_file (tuple):
                            skew_data_df - Skewness table
                            skew_plot - Matplotlib object containing bar chart of skewness table
                            raw_html_file - html content list if available
    """
    skewness_html_content = []
    skew_data_df = pd.DataFrame(data_df.skew(), columns=['skewness'])
    skew_data_df.reset_index(inplace=True)
    skew_data_df.columns = ['Column Names', 'Skewness Value']
    if plot:
        plt.gcf().clear()
        skew_plot = bar_plotter(skew_data_df, x_axis='Column Names', y_axis='Skewness Value')
    else:
        skew_plot = None
    if html_report:
        skew_data_df_rounded = skew_data_df.copy()
        skew_data_df_rounded['Skewness Value'] = skew_data_df_rounded['Skewness Value'].apply(lambda x: round(x, 2))
        skewness_html_content.extend(html_converter(table=skew_data_df_rounded, heading='Skewness Value Table'))
        if plot:
            plt.savefig('plot_skew.png', bbox_inches='tight')
            skewness_html_content.extend(html_converter(image='plot_skew.png', heading='Skewness Plot'))
    return skew_data_df, skew_plot, skewness_html_content


def calc_kurtosis(data_df, plot=False, html_report=False):
    """
    This function calculates kurtosis of each column of the data_df

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        plot (bool): True, if plot needs to be created. False, otherwise
        html_report (bool) : True, if report needs to be generated , False otherwise
    Returns:
        kurtosis_df, kurtosis_plot, raw_html_file (tuple):
                            skurtosis_df - Kurtosis table
                            kurtosis_plot - Matplotlib object containing bar chart of kurtosis table
                            raw_html_file - html content list if available
    """
    kurtosis_html_content = []
    kurtosis_df = pd.DataFrame(data_df.kurtosis(), columns=['kurtosis'])
    kurtosis_df.reset_index(inplace=True)
    kurtosis_df.columns = ['Column Names', 'Kurtosis Value']
    if plot:
        plt.gcf().clear()
        kurtosis_plot = bar_plotter(kurtosis_df, x_axis='Column Names',
                                    y_axis='Kurtosis Value')
    else:
        kurtosis_plot = None
    if html_report:
        kurtosis_df_rounded = kurtosis_df.copy()
        kurtosis_df_rounded['Kurtosis Value'] = kurtosis_df_rounded['Kurtosis Value'].apply(lambda x: round(x, 2))
        kurtosis_html_content.extend(html_converter(table=kurtosis_df_rounded,  heading='Kurtosis Value Table'))
        if plot:
            plt.savefig('plot_kurt.png', bbox_inches='tight')
            kurtosis_html_content.extend(html_converter(image='plot_kurt.png', heading='Kurtosis plot'))
    return kurtosis_df, kurtosis_plot, kurtosis_html_content


def calc_missing(data_df, plot=False, html_report=False):
    """
    This function calculates missing values of each column of the data_df

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        plot (bool): True, if plot needs to be created. False, otherwise
        html_report (bool) : True, if report needs to be generated , False otherwise
    Returns:
        missing_df, missing_plot, raw_html_file (tuple):
                            missing_df - Missing percentage table
                            missing_plot - Matplotlib object containing bar chart of missing percentage table
                            raw_html_file - html content list if available
    """
    missing_html_content = []
    missing_plot = None
    null_counts = pd.isnull(data_df).sum()
    missing_df = pd.DataFrame({'Column Names': list(null_counts.index),
                               'Missing Value': list(null_counts)})
    missing_df['Missing Percentage'] = round(
        (missing_df['Missing Value']/len(data_df))*100, 2)
    if plot:
        plt.gcf().clear()
        missing_plot = bar_plotter(missing_df, x_axis='Column Names',
                                   y_axis='Missing Percentage')
    else:
        missing_plot = None
    if html_report:
        missing_html_content.extend(html_converter(table=missing_df,  heading='Missing Value Table'))
        if plot:
            plt.savefig('plot_miss.png', bbox_inches='tight')
            missing_html_content.extend(html_converter(image='plot_miss.png',  heading='Missing plot'))
    return missing_df, missing_plot, missing_html_content


def calc_cat_count(data_df, plot=False, html_report=False,  clm_desc=None):
    """
    This function calculates count of each categorical column and creates a data_df of that

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        plot (bool): True, if plot needs to be created. False, otherwise
        html_report (bool) : True, if report needs to be generated , False otherwise
        clm_desc (dict) : Contains all the columns as keys and its data type as values
    Returns:
        cat_count_df, cat_count_plot, raw_html_file (tuple):
                            cat_count_df - Categorical count table
                            cat_count_plot - Matplotlib object containing count plot
                            raw_html_file - html content list if available
    """
    categorical_count_html_content = []
    if not clm_desc:
        clm_desc = determine_data_column_types(data_df)
    cat_clms = [clm for clm, dtype in clm_desc.items() if dtype in ['Object', 'Text']]
    cat_count_df = pd.DataFrame(columns=['Column Name', 'Category Level', 'Total Count'])
    for clm in cat_clms:
        temp_df = pd.DataFrame(data_df[clm].groupby(data_df[clm]).count())
        temp_df['Column Name'] = clm
        temp_df.rename(columns={clm: 'Total Count'}, inplace=True)
        temp_df['Category Level'] = temp_df.index
        cat_count_df = cat_count_df.append(temp_df,ignore_index=True)
    if plot:
        plt.gcf().clear()
        row_num = int(np.ceil(len(cat_clms)/3))
        col_num = 3
        fig, axs = plt.subplots(row_num, col_num, figsize=(18, row_num*4), facecolor='w', edgecolor='k')
        axs = axs.ravel()
        for i in range(0,len(cat_clms)):
            cat_count_plot = sns.countplot(data_df[cat_clms[i]], data=data_df, ax=axs[i])
        fig.suptitle('Categorical Count Plot', fontsize=22)
        if html_report:
            categorical_count_html_content.extend(html_converter(table=cat_count_df, heading='Categorical Count Table'))
            if plot:
                plt.savefig('plot_cat_count.png', bbox_inches='tight')
                categorical_count_html_content.extend(html_converter(image='plot_cat_count.png', heading=''))
    else:
        cat_count_plot = None
    return cat_count_df, cat_count_plot, categorical_count_html_content


def show_numeric_dist_plot(data_df, html_report=False,  clm_desc=None):
    """
    This function calculates count of each categorical column and creates a data_df of that

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        plot (bool): True, if plot needs to be created. False, otherwise
        html_report (bool) : True, if report needs to be generated , False otherwise
        clm_desc (dict) : Contains all the columns as keys and its data type as values
    Returns:
        cat_count_df, cat_count_plot, raw_html_file (tuple):
                            cat_count_df - Categorical count table
                            cat_count_plot - Matplotlib object containing count plot
                            raw_html_file - html content list if available
    """
    numerical_dist_plot_html_content = []
    if not clm_desc:
        clm_desc = determine_data_column_types(data_df)
    num_clms = [clm for clm, dtype in clm_desc.items() if dtype in ['Integer', 'Float']]
    row_num = int(np.ceil(len(num_clms)/3))
    col_num = 3
    plt.gcf().clear()
    fig, axs = plt.subplots(row_num, col_num, figsize=(18, row_num*4), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    for i in range(0,len(num_clms)):
        num_dist_plot = sns.distplot(data_df[num_clms[i]].dropna(), ax=axs[i], axlabel=False,
                                     label=str(num_clms[i]))
        axs[i].legend(["{}".format(num_clms[i])])
    fig.suptitle('Distribution Plot', fontsize=22)
    if html_report:
        plt.savefig('plot_dist.png', bbox_inches='tight')
        numerical_dist_plot_html_content.extend(html_converter(image='plot_dist.png', heading=''))
    return num_dist_plot, numerical_dist_plot_html_content


def show_numeric_box_plot(data_df, plot=False, html_report=False,  clm_desc=None):
    """
    This function creates box plot of the provided data_df(for numerical columns)

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        plot (bool): True, if plot needs to be created. False, otherwise
        html_report (bool) : True, if report needs to be generated , False otherwise
        clm_desc (dict) : Contains all the columns as keys and its data type as values
    Returns:
        num_box_plot, raw_html_file (tuple):
                            num_box_plot - Matplotlib object containing box plot
                            raw_html_file - html content list if available
    """
    numeric_box_plot_html_content = []
    if not clm_desc:
        clm_desc = determine_data_column_types(data_df)
    num_clms = [clm for clm, dtype in clm_desc.items() if dtype in ['Integer', 'Float']]
    row_num = int(np.ceil(len(num_clms)/3))
    col_num = 3
    plt.gcf().clear()
    fig, axs = plt.subplots(row_num, col_num, figsize=(18, row_num*4), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    for i in range(0, len(num_clms)):
        num_box_plot = sns.boxplot(x=data_df[num_clms[i]].dropna(), showmeans=True, ax=axs[i], orient='v')
        axs[i].legend(["{}".format(num_clms[i])])
    fig.suptitle('Box Plot', fontsize=22)
    if html_report:
        plt.savefig('plot_box.png', bbox_inches='tight')
        numeric_box_plot_html_content.extend(html_converter(image='plot_box.png', heading=''))
    return num_box_plot, numeric_box_plot_html_content


def calc_correlation(data_df, plot=False, html_report=False,  clm_desc=None):
    """
    Function plots a graphical correlation matrix(pearson) for each pair of columns in the data_df

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        plot (bool): True, if plot needs to be created. False, otherwise
        html_report (bool) : True, if report needs to be generated , False otherwise
        clm_desc (dict) : Contains all the columns as keys and its data type as values
    Returns:
        corr_df, num_corr_plot, raw_html_file (tuple):
                            corr_df - Correlation table
                            num_corr_plot - Matplotlib object containing correlation plot
                            raw_html_file - html content list if available
    """
    correlation_html_content = []
    if not clm_desc:
        clm_desc = determine_data_column_types(data_df)
    num_clms = [clm for clm, dtype in clm_desc.items() if dtype in ['Integer', 'Float']]
    size = 7
    corr_df = data_df[num_clms].corr()
    if plot:
        plt.gcf().clear()
        fig, ax = plt.subplots(figsize=(size, size))
        num_corr_plot = sns.heatmap(corr_df, mask=np.zeros_like(corr_df, dtype=np.bool),
                                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                                    square=True, ax=ax)
        plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
        plt.yticks(range(len(corr_df.columns)), corr_df.columns)
        fig.suptitle('Correlation Plot', fontsize=22)
    else:
        num_corr_plot = None
    if html_report:
        correlation_html_content.extend(html_converter(table=corr_df, heading='Correlation Matrix'))
        if plot:
            plt.savefig('plot_corr.png', bbox_inches='tight')
            correlation_html_content.extend(html_converter(image='plot_corr.png',  heading=''))
    return corr_df, num_corr_plot, correlation_html_content


def show_row_wise_missing(data_df, html_report=False):
    """
    Function creates a table(pandas data_df) having all the rows having more than 50% of the columns missing

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        html_report (bool) : True, if report needs to be generated , False otherwise
    Returns:
        row_wise_miss_val_df, raw_html_file (tuple):
                            row_wise_miss_val_df - row-wise missing table
                            raw_html_file - html content list if available
    """
    row_wise_missing_html_content = []
    if type(data_df.index[1]) is str:  # In case if the index is some date, or some UPC code, resetting it to numbers
        data_df.reset_index(inplace=True)
    row_wise_miss_value_count = list(pd.isnull(data_df).sum(axis=1))
    row_wise_miss_val_df = \
        data_df[pd.Series(row_wise_miss_value_count).apply(
            lambda val:val >= (math.ceil(0.50 * len(data_df.columns))))]
    if html_report:
        row_wise_missing_html_content.extend(html_converter(table=row_wise_miss_val_df, heading='Correlation Matrix'))
    return row_wise_miss_val_df, row_wise_missing_html_content


def show_missing_visual(data_df, html_report=False):
    """
    Function creates a visual plot of the data frame showing the missing values(sample image is present in readme file)

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        html_report (bool) : True, if report needs to be generated , False otherwise
    Returns:
        missing_visual_plot, raw_html_file (tuple):
                            missing_visual_plot - Matplotlib object containing missing value plot
                            raw_html_file - html content list if available
    """
    missing_visual_html_content = []
    missing_visual_plot = msno.matrix(data_df, figsize=(12, 6), fontsize=12)
    if html_report:
        plt.savefig('plot_miss_vis.png', bbox_inches='tight')
        missing_visual_html_content.extend(html_converter(image='plot_miss_vis.png', heading='Missing Data Pattern'))
    return missing_visual_plot, missing_visual_html_content


def bar_plotter(data_df=None, x_axis=None, y_axis=None):
    """
    Function creates a bar plot for the given data_df

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        x_axis (string) : x- axis column name for bar plot
        y_axis (string) : y- axis column name for bar plot
    Returns:
        plot : Matplotlib object having bar plot
    """
    plt.gcf().clear()
    if data_df is None:
        sys.exit("Please provide data_df to plot a chart")
    elif not x_axis:
        sys.exit("Please provide x_axis")
    elif not y_axis:
        sys.exit("Please provide y_axis")
    x_pos = np.arange(len(data_df)*8, step=8)
    plot = plt.bar(x_pos, data_df[y_axis], color='b', width=3, align='center')
    plt.xticks(x_pos, data_df[x_axis], rotation=90)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    return plot


def calc_date_time_clm_exploration(data_df, html_report=False, clm_desc=None):
    """
    Function creates a visual line plot of the date time column showing discontinuity in the period for whch the data
     is unavailable

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        html_report (bool) : True, if report needs to be generated , False otherwise
        clm_desc (dict) : Contains all the columns as keys and its data type as values
    Returns:
        date_time_clms_res, raw_html_file (tuple):
                            date_time_clms_res - It is a dictionary having structure as below:
            date_time_clms_res = { 'column_name' : {'month_vs_year_count' : {'df': table, 'plot': matplotlib_object},
                                                   {'day_vs_year_count' : {'df': table, 'plot': matplotlib_object},
                                                   {'year_wise_count' : {'df': table, 'plot': matplotlib_object},
                                                   {'month_wise_count' : {'df': table, 'plot': matplotlib_object},
                                                   {'day_wise_count' : {'df': table, 'plot': matplotlib_object}
                                 }
    """
    date_time_clm_exploration_html_content = []
    date_time_clms_res = {}
    if not clm_desc:
        clm_desc = determine_data_column_types(data_df)
    date_time_clms = [clm for clm, dtype in clm_desc.items() if dtype in ['DateTime']]
    for clm in date_time_clms:
        date_time_clms_res = \
            {clm: {'month_vs_year_count': {},
                   'day_vs_year_count': {},
                   'year_wise_count': {},
                   'month_wise_count': {},
                   'day_wise_count': {}}}

        date_time = pd.DataFrame({'count_col': list(data_df[clm])}, index=data_df[clm])
        del date_time.index.name

        # # Time period missing monthly
        pv = pd.pivot_table(date_time, index=date_time.index.month, columns=date_time.index.year,
                            values='count_col', aggfunc='count')
        pv.index = pd.Series(pv.index).apply(lambda elem: int(elem))
        pv.columns = pd.Series(pv.columns).apply(lambda elem: int(elem))
        pv.index = pd.Series(pv.index).apply(lambda row: calendar.month_abbr[int(row)])
        date_time_clms_res[clm]['month_vs_year_count']['df'] = pv
        plt.gcf().clear()
        fig, axs = plt.subplots(1, 2, figsize=(14, 4), facecolor='w', edgecolor='k')
        date_time_clms_res[clm]['month_vs_year_count']['plot'] = pv.plot(ax=axs[0])
        axs[0].legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)

        # # Time period missing daily
        pv = pd.pivot_table(date_time, index=date_time.index.day, columns=date_time.index.year,
                            values='count_col', aggfunc='count')
        pv.index = pd.Series(pv.index).apply(lambda elem: int(elem))
        pv.columns = pd.Series(pv.columns).apply(lambda elem: int(elem))
        date_time_clms_res[clm]['day_vs_year_count']['plot'] = pv.plot(ax=axs[1])
        date_time_clms_res[clm]['day_vs_year_count']['df'] = pv
        axs[1].legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.savefig('plot_dt_time.png', bbox_inches='tight')
        if html_report:
            date_time_clm_exploration_html_content.extend(html_converter(image='plot_dt_time.png',
                                                                         heading='''Exploring column {} on monthly
                                                                         and daily basis to see any missing time period
                                                                         (any discontinuity in curve shows data
                                                                         unavailability for that
                                                                         period)'''.format(clm)))
        plt.gcf().clear()
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), facecolor='w', edgecolor='k')
        axs = axs.ravel()

        # year wise count
        date_time_clms_res[clm]['year_wise_count']['df'] = \
            data_df.groupby(data_df[clm].dt.strftime('%Y'))[clm].count()
        date_time_clms_res[clm]['year_wise_count']['plot'] = \
            data_df.groupby(data_df[clm].dt.strftime('%Y'))[clm].count().plot(kind='line', ax=axs[0])
        # month wise count
        date_time_clms_res[clm]['month_wise_count']['df'] = \
            data_df.groupby(data_df[clm].dt.strftime('%B'))[clm].count()
        date_time_clms_res[clm]['month_wise_count']['plot'] = \
            data_df.groupby(data_df[clm].dt.strftime('%B'))[clm].count().plot(kind='line', ax=axs[1])
        # day wise count
        date_time_clms_res[clm]['day_wise_count']['df'] = \
            data_df.groupby(data_df[clm].dt.strftime('%d'))[clm].count()
        date_time_clms_res[clm]['day_wise_count']['plot'] = \
            data_df.groupby(data_df[clm].dt.strftime('%d'))[clm].count().plot(kind='line', ax=axs[2])
        plt.savefig('plot_dt_time_2.png', bbox_inches='tight')
        if html_report:
            date_time_clm_exploration_html_content.extend(html_converter(image='plot_dt_time_2.png',
                                                                         heading='{}'.format(clm.title())))
    return date_time_clms_res, date_time_clm_exploration_html_content


def del_image_file():
    """
    Function deletes all the .png files
    """
    working_path_dir = os.getcwd()
    if ('home' in working_path_dir.split('/')) & ('cdsw' in working_path_dir.split('/')):
        file_path = '{}{}'.format(working_path_dir, '/')
    else:
        file_path = '{}{}'.format(working_path_dir, '\\')
    direc = os.listdir(file_path)
    for item in direc:
        if item.endswith(".png"):
            os.remove(item)


def profile(data_df, get_summary=None,
            get_data_type=None,
            get_skewness=None,
            get_kurtosis=None,
            get_missing=None,
            get_missing_visual=None,
            get_distinct=None,
            get_categorical_count=None,
            get_numerical_dist_plot=None,
            get_numeric_box_plot=None,
            get_row_wise_missing=None,
            get_correlation=None,
            date_time_clm_exploration=None,
            date_time_var=None, generate_html_report=False):
    """
    Function creates complete data profile

    Args:
        data_df (pandas DataFrame): A pandas DataFrame whose column needs to be checked
        get_summary (bool) : if True then execute
        get_data_type (bool) : if True then execute
        get_skewness (dict) :
                            get_skewness = {'df':True, 'plot':True}, if True then execute
        get_kurtosis (dict) :
                            get_kurtosis = {'df':True, 'plot':True}, if True then execute
        get_missing (dict) :
                            get_missing = {'df':True, 'plot':True}, if True then execute
        get_missing_visual (bool) : if True then execute
        get_distinct (dict) :
                            get_distinct = {'df':True, 'plot':True}, if True then execute
        get_categorical_count (dict) :
                                    get_categorical_count = {'df':True, 'plot':True}, if True then execute
        get_numerical_dist_plot (bool) : if True then execute
        get_numeric_box_plot (bool) : if True then execute
        get_row_wise_missing (bool) : if True then execute
        get_correlation (dict) :
                                get_correlation = {'df':True, 'plot':True}, if True then execute
        date_time_clm_exploration (bool) : if True then execute
        date_time_var (list) : user entered date time column
        generate_html_report (bool) : if True, create HTML report, else otherwise

    Returns:
        aggregated_result : dictionary having each functionality as a 'key' and their respective output as 'value'
    """
    if data_df is None:
        sys.exit("WARNING : Please input argument \'data_df\' as pandas data frame")
    elif not isinstance(data_df, pd.DataFrame):
        sys.exit("WARNING : Input argument \'data_df\' should be of type pandas.Dataframe")
    raw_html_file = []
    aggregated_result = {}
    if generate_html_report:
        report_head_html_content = create_report_head(data_df)
        raw_html_file.extend(report_head_html_content)

    # Data Profiling
    clm_desc = determine_data_column_types(data_df, date_time_var)
    if get_summary:
        summary, summary_html_content = calc_summary(data_df, clm_desc=clm_desc, html_report=generate_html_report)
        raw_html_file.extend(summary_html_content)
        aggregated_result['summary'] = summary
        print("Summary done")

    if get_data_type:
        data_type_df, data_type_html_content = calc_data_type(data_df, clm_desc=clm_desc,
                                                              html_report=generate_html_report)
        raw_html_file.extend(data_type_html_content)
        aggregated_result['data_type'] = data_type_df
        print("Data Type done")

    if get_skewness:
        if get_skewness['df'] or get_skewness['df']:
            skewness, skewness_plot, skewness_html_content = calc_skewness(data_df, plot=get_skewness['plot'],
                                                                           html_report=generate_html_report)
            raw_html_file.extend(skewness_html_content)
            aggregated_result['skewness'] = {}
            aggregated_result['skewness']['df'] = skewness
            if get_skewness['plot']:
                aggregated_result['skewness']['plot'] = skewness_plot
            print("Skewness done")

    if get_kurtosis:
        if get_kurtosis['df'] or get_kurtosis['df']:
            kurtosis, kurtosis_plot, kurtosis_html_content = calc_kurtosis(data_df, plot=get_kurtosis['plot'],
                                                                           html_report=generate_html_report)
            raw_html_file.extend(kurtosis_html_content)
            aggregated_result['kurtosis'] = {}
            aggregated_result['kurtosis']['df'] = kurtosis
            if get_kurtosis['plot']:
                aggregated_result['kurtosis']['plot'] = kurtosis_plot
            print("Kurtosis done")

    if get_missing:
        if get_missing['df'] or get_missing['df']:
            missing_df, missing_plot, missing_html_content = calc_missing(data_df, plot=get_missing['plot'],
                                                                          html_report=generate_html_report)
            raw_html_file.extend(missing_html_content)
            print(missing_plot)
            aggregated_result['missing_df'] = {}
            aggregated_result['missing_df']['df'] = missing_df
            if get_missing['plot']:
                aggregated_result['missing_df']['plot'] = missing_plot
            print("Missing percentage done")

    if get_missing_visual:
        missing_visual_plot, missing_visual_html_content = show_missing_visual(data_df,
                                                                               html_report=generate_html_report)
        raw_html_file.extend(missing_visual_html_content)
        aggregated_result['missing_visual_plot'] = {}
        aggregated_result['missing_visual_plot']['plot'] = missing_visual_plot
        print("Missing Visual done")

    if get_distinct:
        if get_distinct['df'] or get_distinct['df']:
            distinct, distinct_plot, distinct_html_content = calc_distinct(data_df, plot=get_distinct['plot'],
                                                                           html_report=generate_html_report)
            raw_html_file.extend(distinct_html_content)
            aggregated_result['distinct'] = {}
            aggregated_result['distinct']['df'] = distinct
            if get_distinct['plot']:
                aggregated_result['distinct']['plot'] = distinct_plot
            print("Distinct percentage done")

    if get_categorical_count:
        if get_categorical_count['df'] or get_categorical_count['df']:
            cat_count_df, cat_count_plot, categorical_count_html_content = \
                calc_cat_count(data_df, plot=get_categorical_count['plot'], html_report=generate_html_report,
                               clm_desc=clm_desc)
            raw_html_file.extend(categorical_count_html_content)
            aggregated_result['cat_count_df'] = {}
            aggregated_result['cat_count_df']['df'] = cat_count_df
            if get_categorical_count['plot']:
                aggregated_result['cat_count_df']['plot'] = cat_count_plot
            print("Categorical Count done")

    if get_numerical_dist_plot:
        num_dist_plot, numerical_dist_plot_html_content = show_numeric_dist_plot(data_df,
                                                                                 html_report=generate_html_report,
                                                                                 clm_desc=clm_desc)
        raw_html_file.extend(numerical_dist_plot_html_content)
        aggregated_result['num_dist_plot'] = {}
        aggregated_result['num_dist_plot']['plot'] = num_dist_plot
        print("Numerical Dist plot done")

    print("----", get_numeric_box_plot)# htd
    if get_numeric_box_plot:
        print("----", get_numeric_box_plot) #htd
        num_box_plot, numeric_box_plot_html_content = show_numeric_box_plot(data_df, plot=get_numerical_dist_plot,
                                                                            html_report=generate_html_report,
                                                                            clm_desc=clm_desc)
        raw_html_file.extend(numeric_box_plot_html_content)
        aggregated_result['num_box_plot'] = {}
        aggregated_result['num_box_plot']['plot'] = num_box_plot
        print("Numeric box plot done")

    if get_correlation:
        if get_correlation['df'] or get_correlation['df']:
            corr_df, num_corr_plot, correlation_html_content = calc_correlation(data_df, plot=get_correlation['plot'],
                                                                                html_report=generate_html_report,
                                                                                clm_desc=clm_desc)
            raw_html_file.extend(correlation_html_content)
            aggregated_result['corr_df'] = {}
            aggregated_result['corr_df']['df'] = corr_df
            if get_categorical_count['plot']:
                aggregated_result['corr_df']['plot'] = num_corr_plot
            print("Correlation done")

    if get_row_wise_missing:
        row_wise_miss_val_df, row_wise_missing_html_content = show_row_wise_missing(data_df, html_report=False)
        raw_html_file.extend(row_wise_missing_html_content)
        aggregated_result['row_wise_missing'] = {}
        aggregated_result['num_box_plot']['df'] = row_wise_miss_val_df
        print("Row wise missing done")

    if date_time_clm_exploration:
        date_time_clms_res, date_time_clm_exploration_html_content = \
            calc_date_time_clm_exploration(data_df, html_report=generate_html_report, clm_desc=clm_desc)
        raw_html_file.extend(date_time_clm_exploration_html_content)
        aggregated_result['date_time_clm_exploration'] = date_time_clms_res
    print("Date-Time Exp done")

    if generate_html_report:
        html_writer(raw_html_file)
    del_image_file()
    return aggregated_result
