import logging
import pandas as pd


class BoolParser(object):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.bool_col = []
        self.logger = logging.getLogger(__name__)

    def parse_columns(self, col_names: list):
        self.bool_col = [col for col in col_names if self.is_bool(col)]
        self.logger.debug(f"Boolean columns: {self.bool_col}")
        for col in self.bool_col:
            self.convert_bool_to_int(col)

    def is_bool(self, col_name: str) -> bool:
        """
        Check if a column is boolean type.
        """
        return all(isinstance(x, bool) for x in self.df[col_name].dropna())

    def convert_bool_to_int(self, col_name: str):
        self.df[col_name] = self.df[col_name].astype('Int64')
        self.logger.debug(f"Column {col_name} has been converted to Int64.")
        self.one_hot_encode_nan(col_name)
        return

    def one_hot_encode_nan(self, col_name: str):
        self.df[f'{col_name}_is_nan'] = self.df[col_name].isna().astype('Int64')
        self.df.fillna({col_name:0}, inplace=True)
        self.logger.debug(f"Column {col_name}'s nan has been one-hot encoded.")
        return
