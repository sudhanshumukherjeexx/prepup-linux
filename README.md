# 💻 Prepup: Interactive Data Preprocessing Toolkit

![Static Badge](https://img.shields.io/badge/Built_with_%E2%99%A5%EF%B8%8F-Sudhanshu_Mukherjee-black?link=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fsudhanshumukherjeexx%2F)

![Python Versions](https://img.shields.io/badge/python-3.7+-blue.svg)
[![image](https://img.shields.io/pypi/v/prepup-linux.svg)](https://pypi.python.org/pypi/prepup-linux)
![Static Badge](https://img.shields.io/badge/Linux-Supported-green)
![Static Badge](https://img.shields.io/badge/macOS-Supported-blue)
![Static Badge](https://img.shields.io/badge/Ubuntu-Supported-red)
![License](https://img.shields.io/badge/license-MIT-green.svg)


# ⚠️ PACKAGE RENAMED: prepup-linux → ride-cli

> **IMPORTANT**: This package has been renamed to `ride-cli`. Please use the new package for all future installations and updates.

## Migration Instructions

To migrate to the new package:

```bash
# Uninstall the old package
pip uninstall prepup-linux

# Install the new package
pip install ride-cli
```

All functionality remains the same. The only change is the package name and command:
- Old command: `prepup`
- New command: `ride` or `ride-cli`

## Why the Change?

Prepup began in summer 2023 as the Preprocessing Utility Package (PrePUP) with just 5 terminal flags—a learning project that evolved into a comprehensive data tool. After creating prepup-linux to address cross-platform compatibility issues, we realized the name incorrectly suggested Linux exclusivity, when our vision has always been platform independence. We also tested our first menu-driven approach in prepup-linux. We're now transitioning to RIDE-CLI (Rapid Insights Data Engine), a name that better reflects our tool's capabilities: rapid data preprocessing, meaningful insights generation, and cross-platform functionality. This rebranding represents our growth from a simple utility to a robust data engine, while maintaining our commitment to continuous improvements and expanded features across all platforms.

---


## 🚀 Quick Overview

Prepup is a powerful, user-friendly data preprocessing tool designed to simplify and streamline your data analysis workflow directly from the terminal. Whether you're a data scientist, analyst, or researcher, Prepup provides an intuitive interface for exploring, cleaning, and preparing your datasets.

## ✨ Features

### Interactive Mode
- 📊 Load datasets from various formats (CSV, Excel, Parquet)
- 🔍 Comprehensive data inspection
- 📈 Advanced data exploration
- 🧹 Missing value handling
- 📊 Feature visualization
- 🤖 Automatic Machine Learning (AutoML) model selection

### Key Functionalities
- Data Loading
- Feature Inspection
- Correlation Analysis
- Distribution Checking
- Outlier Detection
- Missing Value Imputation
- Feature Standardization
- Automatic Model Training

## 🛠 Installation

> **⚠️ Important:** Creating a virtual environment is highly recommended when installing prepup-linux. As a data processing library, it has various dependencies that may conflict with your existing packages.

### Setting Up a Virtual Environment

#### Windows
```bash
# Create virtual environment
python -m venv prepup-env

# Activate virtual environment
prepup-env\Scripts\activate

# Deactivate when done
deactivate
```

#### Linux/macOS
```bash
# Create virtual environment
python3 -m venv prepup-env

# Activate virtual environment
source prepup-env/bin/activate

# Deactivate when done
deactivate
```

### Using pip
```bash
# Inside your activated virtual environment
pip install prepup-linux
```

### From Source
```bash
# Inside your activated virtual environment
git clone https://github.com/sudhanshumukherjeexx/prepup-linux.git
cd prepup-linux
pip install .
```

## 💻 Usage

### Interactive Mode
```bash
prepup
```

### Loading a Specific Dataset
```bash
prepup path/to/your/dataset.csv
```

### Main Menu Options
1. Load Dataset
2. Inspect Data
3. Explore Data
4. Visualize Data
5. Impute Missing Values
6. Standardize Features
7. Export Data
8. AutoML (Train & Evaluate Models)

## 🎮 Interactive Workflow Example

1. **Launch Prepup** ```prepup```

2. **Load Your Dataset:** Choose option 1 and enter your dataset path

3. **Inspect Data:** Use option 2 to explore features, data types, and missing values

4. **Preprocess:** Impute missing values | Standardize features

5. **Analyze:** Visualize data distributions | Perform correlation analysis | Run AutoML for model selection

## 🤖 AutoML Capabilities
- Supports both Classification and Regression tasks
- Evaluates multiple machine learning algorithms
- Provides performance metrics
- Saves results to CSV

## 📦 Dependencies
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- and more (see requirements.txt)

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 License
Distributed under the MIT License. See `LICENSE` for more information.

## 🔄 Migration Notice

This package is deprecated and will no longer receive updates. Please migrate to [ride-cli](https://github.com/sudhanshumukherjeexx/ride-cli) for the latest features and support.

### New Package Links
- GitHub: https://github.com/sudhanshumukherjeexx/ride-cli
- PyPI: https://pypi.org/project/ride-cli/