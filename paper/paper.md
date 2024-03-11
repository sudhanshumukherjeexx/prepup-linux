---
title: 'Prepup-Linux: A Python package for pre-processing tabular data on command line interface(CLI)'
tags:
  - Python
  - data pre-processing
  - command line interface
  - Tabular data
  - machine learning
authors:
  - name: Sudhanshu Mukherjee
    orcid: 0009-0007-5020-6153
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Alfa Heryudono
    orcid: 0000-0001-7531-2891
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: University of Massachusetts Dartmouth, USA
   index: 1
 - name: Department of Mathematics, University of Massachusetts Dartmouth, USA
   index: 2
date: 11 march 2024
bibliography: paper.bib
---

# Summary

`Prepup-Linux` is an innovative command-line tool built to streamline data pre-processing in data science workflow. By combining the capabilities of polars  [Vink, 2024] for faster data loading and pandas  [@McKinney, 2010; @McKinney, 2011] for data manipulation and scikit-learn [@Pedregosa, 2011; @Grisel, 2015] for advanced preprocessing, it introduces a concise way to pre-process datasets using just five terminal flags[` -inspect`,` -explore`,` -visualize`,` -impute` and `-standardize`]. Prepup-Linux supports `CSV`, `Excel`, and `Parquet` file formats and it also simplifies data analysis tasks—from examining data types and missing values to complex visualizations and normalization processes—directly from the terminal. This approach minimizes the need for graphical user interfaces (GUI), making data preprocessing accessible and efficient for researchers and developers. Prepup-Linux allows you to save all your modified files locally. By focusing on ease of use and functionality, prepup-Linux offers a robust solution for rapid data pre-processing, allowing users to concentrate on analytical insights rather than the preprocessing workflow, improving productivity and efficiency in data analysis.

# Statement of need

The preprocessing phase, which includes data cleaning, visualization, normalization, and handling missing values, is critical for ensuring data quality and reliability in subsequent analyses. However, this phase often involves a complex workflow that requires toggling between different tools and programming environments, which can be both time-consuming and prone to errors.

Prepup-linux addresses these challenges by offering a streamlined, command-line based solution that integrates multiple preprocessing steps into a single, user-friendly package. Despite the wide array of tools available in the Python ecosystem  [@Van Rossum, 1995] for data manipulation and analysis, there exists a gap for a tool that can perform comprehensive data preprocessing directly from the terminal. This gap not only slows down the data analysis process but also increases the learning curve for individuals new to data science.

By consolidating essential preprocessing functionalities into concise terminal commands, prepup-linux significantly reduces the time and effort required to prepare datasets for analysis. It eliminates the need for extensive coding or switching between various libraries and interfaces, thus democratizing data science by making it more accessible to users with varying levels of programming expertise. Furthermore, prepup-linux's emphasis on command-line operations caters to scenarios where graphical user interfaces (GUIs) are not feasible, such as remote server environments or when working with large datasets that demand considerable computational resources.

The need for prepup-linux arises from the growing demand for a more efficient, integrated, and accessible approach to data preprocessing in the data science community. It fills a critical void by enabling swift and straightforward data preprocessing, thereby allowing researchers and developers to focus more on extracting insights and less on the mechanics of data preparation.

# Methodology 

The development of prepup-linux was guided by a structured approach to address specific needs in the data preprocessing domain, particularly focusing on command-line interface (CLI) efficiency, flexibility, and user-friendliness. The methodology encompassed several key phases: requirement analysis, design and implementation, testing, and user feedback integration.

### Requirement Analysis

The initial phase involved an extensive review of existing data preprocessing tools, identifying gaps in CLI-based data manipulation and visualization capabilities. This review highlighted the need for a tool that could perform a wide range of preprocessing tasks directly from the terminal, catering to environments where graphical user interface (GUI) applications are impractical or unavailable.

### Design and Implementation

Based on the identified requirements, prepup-linux was designed with a focus on simplicity, leveraging Python’s powerful data manipulation libraries (pandas [@McKinney, 2011], sklearn [@Pedregosa, 2011]) and visualization tools (plotext [@Piccolomo, n.d.]) to offer a comprehensive preprocessing toolkit. The Prepup class, central to the package, encapsulates all preprocessing functionalities, enabling tasks such as data inspection, cleaning, feature scaling, and visualization through intuitive terminal flags.

- **Data Inspection and Cleaning:** Functions were implemented to allow users to easily identify and handle missing values, detect outliers, and explore data types and feature distributions.
- **Visualization:** Leveraging plotext for terminal-based plotting, methods were developed to generate histograms and bar charts, facilitating a visual understanding of the data without the need for a GUI.
- **Feature Scaling:** Integration with sklearn’s preprocessing modules enabled the inclusion of standard scaling and robust scaling techniques, critical for preparing data for machine learning models.

### Testing

Testing was conducted to ensure the reliability and accuracy of prepup-linux functionalities. Unit tests covered each method in the Prepup class, while integration tests verified the package’s performance on real datasets following best practices in machine learning [@Müller, 2016]. This phase also assessed the package’s usability in different operating environments, ensuring compatibility and performance consistency.

# Example

The following code demonstrates how `prepup-linux` cab be used directly on the command line. Once prepup-linux is installed, it allows direct access to preprocessing functionalities, catering to the varied formats of data including CSV, Excel, and Parquet, and leveraging the fast, multi-threaded capabilities of Polars [Vink, 2024] for data loading. One can directly access the package using the entry point keyword `prepup` on the terminal. I will consider `titanic.csv` dataset for the example.

```
$ prepup -h  #this command will display all the terminal flags along with their use.

# Observe the dataset and its Features. Provides a brief overview of a data including missing value count.
$ prepup titanic.csv -inspect 

# Observe correlation between features, checks if feature is normally distributed, detects outliers, Skewness and Kurtosis present in the data.
$ prepup titanic.csv -explore 

# Plots a histogram for each feature present in the dataset.
$ prepup titanic.csv -visualize 

# Impute missing values - there are 8 methods available to impute a missing value
$ prepup titanic.csv -impute

# Perform Scaling on features - 2 options available - Standard Scaling and Robust Scaling
$ prepup titanic.csv -standardize


```

# Overview

The essence of `prepup-linux` lies in its innovative approach to data preprocessing through a command-line interface (CLI), harnessing the strength of Python's most prominent data science libraries—polars [Vink, 2024], pandas  [@McKinney, 2010; @McKinney, 2011], sklearn [@Pedregosa, 2011; @Grisel, 2015], scipy  [@Jones, 2001], and plotext [@Piccolomo, n.d.]. It processes data using pandas DataFrames, ensuring compatibility with the widespread data structures in Python's data science ecosystem.

At its foundation, `prepup-linux` is inspired by the efficiency and simplicity of CLI tools, extending these principles to the domain of data preprocessing. It employs intuitive terminal flags that correspond to various preprocessing tasks, such as data inspection, visualization, cleaning, and feature scaling. This design choice not only aligns with the ease of use but also significantly reduces the learning curve for users accustomed to CLI operations.

`prepup-linux` innovates by integrating data visualization capabilities directly in the terminal, leveraging the plotext library. This unique feature allows for immediate feedback on data distribution and preprocessing outcomes, facilitating a more dynamic and interactive data analysis workflow. Moreover, it incorporates robust data manipulation and scaling techniques from pandas and sklearn, providing a versatile toolkit for preparing datasets for machine learning models.

Central to `prepup-linux` is the `Prepup` class, which acts as the nucleus of the package, orchestrating the preprocessing tasks. It encapsulates the functionality for handling missing values, detecting outliers, scaling features, and assessing dataset balance, among others. This encapsulation ensures that the preprocessing workflow is both streamlined and comprehensive, covering the full spectrum of tasks required to transform raw data into a clean, analysis-ready format.

`prepup-linux` emphasizes minimizing manual effort and maximizing efficiency. It achieves this through a methodology that intelligently automates the selection and application of preprocessing techniques based on the characteristics of the input dataset. For instance, it automatically identifies the appropriate scaling method based on the presence of outliers and selects the best strategy for handling missing values to preserve data integrity.

Furthermore, `prepup-linux` is designed with extensibility in mind, allowing for easy integration of additional preprocessing functions and compatibility with future data formats. This forward-looking approach ensures that `prepup-linux` remains relevant and useful as the field of data science evolves.

In summary, `prepup-linux` stands as a comprehensive, CLI-based solution for data preprocessing in Python. It bridges the gap between raw data and prepared datasets through a seamless, efficient, and user-friendly interface, catering to the needs of data scientists and researchers who prefer or require a CLI environment for their data analysis tasks.


# Acknowledgements

We extend sincere thanks to University of Massachusetts at Dartmouth,  our family, friends, colleagues, and all who contributed directly or indirectly to this python package. Your support, insights, and encouragement have been invaluable. 

# References