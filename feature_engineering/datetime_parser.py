import logging

import numpy as np
import pandas as pd


class DatetimeParser(object):
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DatetimeParser object. This class checks for datetime columns
        and parses them to extract features

        Parameters:
        df (pd.DataFrame): The DataFrame containing the columns to be parsed.

        Note:
        A new DataFrame will be created based on the input DataFrame. To preserve changes,
        the resulting DataFrame must be explicitly assigned back to the original variable containing
        the original dataframe after using the methods of this class.
        """
        self.df = df.copy()
        self.datetime_col = []
        self.logger = logging.getLogger(__name__)

    def parse_columns(self
                      , col_names: list
                      , datetime_format='%d%b%y:%H:%M:%S'
                      , cutoff_ratio=0
                      , extract_features=True
                      , drop_original=True
                      ):
        """
        Parse columns to datetime format and optionally extract datetime features.

        Parameters:
        col_names (list): List of column names to be parsed.
        datetime_format (str): The expected format of the datetime strings.
        cutoff_ratio (float): The minimum ratio of successful conversions required to consider the column as datetime.
        extract_features (bool): Whether to extract features from the datetime column.
        drop_original (bool): Whether to drop the original column after parsing.
        """
        self.logger.debug(f" column count at start of parse_columns = {len(self.df.columns)}")

        for col_name in col_names:
            parsed_col = self.try_parse_datetime(col_name
                                                 , datetime_format
                                                 , cutoff_ratio)
            if parsed_col is not None:
                self.df[col_name] = parsed_col
                self.logger.debug(f"Column {col_name} successfully parsed to datetime")
                self.datetime_col.append(col_name)
                if extract_features:
                    self.extract_datetime_features(col_name)
                if drop_original:
                    self.logger.debug(f"Dropping original column: {col_name}, since drop_original= {drop_original}")
                    self.df.drop(columns=[col_name], inplace=True)
                    if col_name in self.df.columns:
                        self.logger.error(f"Column {col_name} not dropped")

        self.logger.debug(f" column count at end of loop = {len(self.df.columns)}")

    def try_parse_datetime(self
                           , col_name: str
                           , datetime_format='%d%b%y:%H:%M:%S'
                           , cutoff_ratio=0) -> pd.Series:
        """
        Try to parse a column as datetime based on a given format.

        Parameters:
        col_name (str): The name of the column to be parsed.
        datetime_format (str): The expected format of the datetime strings.
        cutoff_ratio (float): The minimum ratio of successful conversions required to consider the column as datetime.

        Returns:
        pd.Series: Parsed datetime column if successful, otherwise None.
        """
        col_series = self.df[col_name]
        parsed_col = pd.to_datetime(col_series
                                    , format=datetime_format
                                    , errors='coerce')
        successful_convert_ratio = parsed_col.notna().mean()
        self.logger.debug(f"Column {col_name} successful conversion ratio: {successful_convert_ratio}")
        if successful_convert_ratio > cutoff_ratio:
            return parsed_col
        else:
            return None

    def extract_datetime_features(self, col_name: str):
        """
        Extract date and time features from a datetime column.

        Parameters:
        col_name (str): The name of the datetime column.
        """
        col_dt = self.df[col_name].dt
        self.extract_date_features(col_name)
        self.extract_time_features(col_name, col_dt)

    def extract_date_features(self
                              , col_name: str):
        """
        Extract date-related features such as year, month, day, day of the week, and cyclical encodings.

        Parameters:
        col_name (str): The name of the datetime column.
        """
        col_dt = self.df[col_name].dt
        self.df[f'{col_name}_year'] = col_dt.year
        self.df[f'{col_name}_month'] = col_dt.month
        self.df[f'{col_name}_day'] = col_dt.day

        self.df[f'{col_name}_day_of_week'] = col_dt.dayofweek
        self.df[f'{col_name}_is_weekend'] = self.df[f'{col_name}_day_of_week'].apply(
            lambda x: 1 if x >= 5 else 0)

        self.apply_sin_cos_encoding(f'{col_name}_day_of_week', 7)
        self.apply_sin_cos_encoding(f'{col_name}_month', 12)

    def extract_time_features(self, col_name: str, col_dt):
        """
        Extract time-related features such as hour, minute, second, and cyclical encodings.

        Parameters:
        col_name (str): The name of the datetime column.
        col_dt: The datetime properties of the column.
        """
        hour = col_dt.hour
        minute = col_dt.minute
        second = col_dt.second

        if self.is_all_null_or_zero(hour, minute, second):
            self.logger.debug(f"Column '{col_name}' contains only null or zero values")
            return
        else:
            self.df[f'{col_name}_hour'] = hour
            self.df[f'{col_name}_minute'] = minute
            self.df[f'{col_name}_second'] = second
            self.apply_sin_cos_encoding(f'{col_name}_hour', 24)

    def apply_sin_cos_encoding(self
                               , col_name: str
                               , period: int):
        """
        Apply sine and cosine encoding to represent cyclical features.

        Parameters:
        col_name (str): The name of the column to be encoded.
        period (int): The period of the cyclical feature (e.g., 7 for day of the week).
        """
        self.df = pd.concat([self.df, pd.DataFrame({
            f'{col_name}_sin': np.sin(2 * np.pi * self.df[col_name] / period)
            , f'{col_name}_cos': np.cos(2 * np.pi * self.df[col_name] / period)
        })], axis=1)

    def is_all_null_or_zero(self
                            , hour_series: pd.Series
                            , minute_series: pd.Series
                            , seconds_series: pd.Series) -> bool:
        all_values = np.concatenate([hour_series.values, minute_series.values, seconds_series.values])
        return np.all(np.isin(all_values, [0.0]) | pd.isna(all_values))
