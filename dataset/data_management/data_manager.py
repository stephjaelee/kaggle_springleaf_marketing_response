import logging
import os

import pandas as pd


class DataManager(object):
    """
    Manages the DataFrame by allowing resets to the latest changes without creating multiple copies.
    Facilitates efficient development by enabling in-memory work without frequent writes to disk.
    """

    def __init__(self, data_file_path: str, write_changes_mode=False):
        self.data_file_path = data_file_path
        self.logger = logging.getLogger(__name__)
        self.write_changes = write_changes_mode
        if write_changes_mode:
            logging.info("All changes will be written to file when commited")
        self.df = pd.DataFrame()

    def load_data_file(self, source_path=None):
        """
        Load data from a specified file path (parquet or CSV).
        """
        if source_path is None:
            source_path = self.data_file_path
        if os.path.exists(source_path):
            if source_path.endswith('.parquet'):
                self.df = pd.read_parquet(source_path)
            elif source_path.endswith('.csv'):
                self.df = pd.read_csv(source_path, header=0)
                self.logger.debug(f"{source_path} as been loaded")
            else:
                self.logger.error(f"Unsupported file format: {source_path}")
        else:
            self.logger.error(f"File not found: {source_path}. Please check the file path and try again.")

    def write_data_file(self):
        """
        Write the DataFrame to a parquet file.
        """
        try:
            self.df.to_parquet(self.data_file_path, index=False)
            self.logger.debug(f"Data successfully written to {self.data_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to write data to {self.data_file_path}: {e}")

    def commit_changes(self, write_changes=None):
        """
        Commit changes by writing to the data file if write_changes is enabled.
        If the `write_changes` parameter is provided, it will override the current mode.
        """
        if write_changes is not None:
            self.write_changes = write_changes
        if self.write_changes:
            self.write_data_file()

    def pull_dataframe(self):
        """
        Reload the DataFrame from the saved file if write_changes is enabled.
        """
        if self.write_changes:
            self.load_data_file()

    def get_columns_startswith(self, col_list: list):
        """
        Get all columns in the DataFrame that match the given list of column prefixes.
        """
        matching_cols = []
        for prefix in col_list:
            matching_cols.extend(self.df.filter(regex=fr'^{prefix}.*').columns.tolist())

        return self.df[matching_cols]

    def get_unique_values_summary(self
                                  , data_types: list
                                  , sample_size=10) -> pd.DataFrame:
        """
        Get a sample of unique values, total unique count, and percent null for each column of specified data types.
        """
        self.logger.debug(f"Filtering DataFrame for data types: {data_types}")
        filtered_df = self.df.select_dtypes(include=data_types)
        self.logger.debug(f"Filtered DataFrame columns: {filtered_df.columns.tolist()}")

        summary_df = pd.DataFrame({
            'unique_values_sample': filtered_df.apply(lambda col: str(col.unique()[:sample_size])),
            'total_unique_count': filtered_df.nunique(),
            'percent_null': (filtered_df.isnull().mean() * 100).round(2)
        })

        self.logger.debug("Summary DataFrame created")
        return summary_df
