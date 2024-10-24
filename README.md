# Kaggle Springleaf Marketing Response Project

## Overview

This project aims to tackle the Kaggle Springleaf Marketing Response problem, with the goal of predicting customer responses to marketing campaigns. The project includes data ingestion, feature engineering, and eventually intends to apply machine learning techniques, such as XGBoost, for predictive modeling.

The primary focus is on creating a data pipeline that converts raw data into usable formats, performs feature engineering, and reduces dimensionality to improve model performance, ultimately using an XGBoost model for prediction.

## Data Ingestion

The data ingestion process is detailed in the `data_ingestion.ipynb` notebook. The steps involved are:

1. **Loading the Data**: The original data comes in a zip format containing CSV files. The script handles unzipping and loading these CSV files.
2. **Conversion to Parquet**: Given the large size of the dataset, the data is converted from CSV to Parquet format for more efficient processing. Parquet files are faster to read and take up less storage space, making subsequent analysis much more manageable. This approach also facilitates integration with data warehouses for future scalability.

## Feature Engineering

Feature engineering was conducted in the `feature_engineering.ipynb` notebook, with the goal of preparing the data for machine learning modeling. Key steps include:

1. **Data Cleaning**: Several parsers were created for cleaning the dataset, including:
   - **Datetime Parser (`datetime_parser.py`)**: Converts dates into a consistent format and extracts useful temporal features.
   - **Boolean Parser (`bool_parser.py`)**: Handles inconsistencies in Boolean values.

**WORK IN PROGRESS**: Everything below this point is a work in progress.

2. **Feature Building**:
   - **City Parser (`city_parser.py`)**: Deals with city names, handling variations and correcting errors (work in progress).
     - **Clustering and Labeling**: Clustering and labeling groups of cities since the cities are manually entered and need to be standardized.
     - **Geolocation**: Using a geocoder to grab the latitude and longitude to build a location feature.
   - **Job Title Parsing**: A new parser will be created to extract meaning from job titles, identifying similarities (e.g., recognizing that CEO is similar to COO but different from admin).

## Dimensionality Reduction

To improve model performance and reduce computational requirements, dimensionality reduction techniques are planned. This will include evaluating feature importance and eliminate features with low variance or highly correlated features. This will help focus the model on the most informative features.

## Next Steps

- **XGBoost Modeling**: Complete the modeling using XGBoost to predict customer responses.
- **Model Evaluation**: Evaluate the model's performance and iterate on feature engineering and modeling to improve results.

## Productionalizing the Solution

To take this solution to production, several key design choices make the code well-suited for production environments:

1. **Modular Code Structure**: The feature engineering scripts are modular and self-contained. Each parser, such as the `bool_parser.py` or `city_parser.py`, focuses on a specific task, making it easier to maintain and extend. This modularity allows for individual components to be tested and updated without affecting the entire pipeline.

2. **Efficient Data Storage**: By converting the data from CSV to Parquet format, the code significantly improves data processing efficiency due to the columnar storage. This also facilitates integration with data warehouses, making it easier to scale the solution in production environments.

3. **Notebook to Script Conversion**: Notebooks were used to handle the pipeline and conversion as opposed to Airflow. Many tools (Databricks, Snowflake) now allow the loading and triggering of notebooks as pipelines. Organizing this as a notebook lends itself well to this approach.
