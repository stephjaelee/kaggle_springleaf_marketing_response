import logging
import os
from collections import defaultdict

import pandas as pd
from thefuzz import fuzz, process


class CityParser(object):

    def __init__(self, df: pd.DataFrame, city_fact_path: str, city_col, state_col):
        self.df = df.copy()
        self.city_fact_path = city_fact_path
        self.city_fact_table = self.load_city_fact_table()
        self.logger = logging.getLogger(__name__)

        self.zip_replacement_value = '_ZIP_CODE_'
        self.temp_suffix = '_temp'

        self.city_col = city_col
        self.state_col = state_col

        self.original_city_col = 'original_city_col'
        self.zip_code_col = 'zip_code'
        self.lat_col = 'lat'
        self.lon_col = 'lon'

        self.city_group_col = 'city_group'

        self.city_fact_city_col = 'city'
        self.city_fact_state_col = 'state'

        self.city_fact_columns = [self.city_fact_city_col
            , self.city_fact_state_col
            , self.lat_col
            , self.lon_col
            , self.zip_code_col]

        self.city_to_group = {}
        self.group_to_cleaned_city = {}
        self.threshold = 90

    def load_city_fact_table(self) -> pd.DataFrame:
        """
        Load city reference table from a parquet file or create it if it doesn't exist
        """
        if os.path.exists(self.city_fact_path):
            return pd.read_parquet(self.city_fact_path)
        else:
            df_city_fact = pd.DataFrame(columns=self.city_fact_columns)
            df_city_fact.to_parquet(self.city_fact_path, index=False)
            return df_city_fact

    def push_data_to_city_fact_table(self):
        """
        Append new city data to the city reference table
        """
        self.city_fact_table.to_parquet(self.city_fact_path, index=False)

    def parse_column(self):

        self._add_location_columns()

        self._clean_city_col()
        self.logger.debug(f"city column cleaned")

        self._from_zip_add_city_lat_lon()
        self._from_city_add_lat_lon()

    def _add_location_columns(self):
        self.df[self.original_city_col] = self.df[self.city_col]
        new_location_columns = [self.zip_code_col, self.lat_col, self.lon_col]
        for col in new_location_columns:
            if col not in self.df.columns:
                self.df[col] = pd.NA

    def _clean_city_col(self):
        self._move_zip_codes_to_zip_column()
        self._remove_digits_w_str_from_city_col()
        self._string_format_city_col()


    def _move_zip_codes_to_zip_column(self):
        """
        Some city names are actually zip codes
        """
        is_zip_code = ((self.df[self.city_col].str.len() == 5)
                       & (self.df[self.city_col].str.isdigit()))

        self.df[self.zip_code_col] = self.df[self.city_col].where(is_zip_code)
        self.df[self.city_col] = self.df[self.city_col].where(
            ~is_zip_code, other=self.zip_replacement_value)

    def _remove_digits_w_str_from_city_col(self):
        """
        Some city names have digits in them
        """
        self.df[self.city_col] = (self.df[self.city_col]
                                  .str.replace(r'\d+', '', regex=True)
                                  .str.strip())

    def _string_format_city_col(self):
        self.df[self.city_col] = self.df[self.city_col].str.strip().str.upper()

    def _from_zip_add_city_lat_lon(self):

        df_merged = self.df.merge(
            self.city_fact_table
            , left_on=[self.zip_code_col, self.state_col]
            , right_on=[self.zip_code_col, self.city_fact_state_col]
            , how='left'
            , suffixes=('', self.temp_suffix)
        )

        previously_pulled_zip = (
                (df_merged[self.city_col] == self.zip_replacement_value)
                & (df_merged[self.city_fact_city_col].notna())
        )

        original_location_cols = [self.city_col, self.lat_col, self.lon_col]
        fact_location_cols = [
            self.city_fact_city_col
            , f'{self.lat_col}{self.temp_suffix}'
            , f'{self.lon_col}{self.temp_suffix}']

        for original_col, fact_col in zip(original_location_cols, fact_location_cols):
            self.df.loc[previously_pulled_zip, original_col] = df_merged.loc[previously_pulled_zip, fact_col]

    def _from_city_add_lat_lon(self):

        df_merged = self.df.merge(
            self.city_fact_table
            , left_on=[self.city_col, self.state_col]
            , right_on=[self.city_fact_city_col, self.city_fact_state_col]
            , how='left'
            , suffixes=('', self.temp_suffix)
        )

        previously_pulled_city = (
                (df_merged[self.lat_col].isna())
                & (df_merged[self.city_fact_city_col].notna())
        )

        original_location_cols = [self.zip_code_col, self.lat_col, self.lon_col]
        fact_location_cols = [
            f'{self.zip_code_col}{self.temp_suffix}'
            , f'{self.lat_col}{self.temp_suffix}'
            , f'{self.lon_col}{self.temp_suffix}']

        for original_col, fact_col in zip(original_location_cols, fact_location_cols):
            self.df.loc[previously_pulled_city, original_col] = df_merged.loc[previously_pulled_city, fact_col]

    def fuzzy_match_and_group_cities(self):

        self._find_similar_cities()
        self.logger.debug(f"similar cities found")
        self._label_similar_cities()
        self.logger.debug(f"similar cities labeled")
        self._apply_grouping_to_main_df()
        self.logger.debug(f"city column grouped")

    def _find_similar_cities(self):
        df = self.df.dropna(subset=[self.city_col])
        self.unique_city_by_state = df.groupby(self.state_col)[self.city_col].unique().reset_index()
        self.similar_city_mapping = defaultdict(list)

        for _, row in self.unique_city_by_state.iterrows():
            state = row[self.state_col]
            unique_cities = row[self.city_col]

            for i, city in enumerate(unique_cities):
                other_cities = unique_cities[i + 1:]
                self.logger.debug(f"Finding similar cities for {city} in {state}")
                matches = process.extract(city, other_cities, scorer=fuzz.WRatio, limit=None)

                for match, score in matches:
                    if score >= self.threshold:
                        self.similar_city_mapping[city].append(match)
                        self.similar_city_mapping[match].append(city)

    def _label_similar_cities(self):
        group_id = 0

        for city, similar_cities in self.similar_city_mapping.items():
            if city not in self.city_to_group:
                group_id += 1
                current_group = set(similar_cities + [city])

                for similar_city in current_group:
                    self.city_to_group[similar_city] = group_id

                group_df = self.df[self.df[self.city_col].isin(current_group)]
                most_frequent_city = group_df[self.city_col].mode().iloc[0] if not group_df.empty else city
                self.group_to_cleaned_city[group_id] = most_frequent_city

    def _apply_grouping_to_main_df(self):
        self.df[self.city_group_col] = self.df[self.city_col].map(self.city_to_group).fillna(1)
        self.df[self.city_col] = (
            self.df[self.city_group_col].map(self.group_to_cleaned_city).fillna(self.df[self.city_col])
        )
