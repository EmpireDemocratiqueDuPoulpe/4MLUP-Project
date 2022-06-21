# #################################################################################################################### #
#       dataframe.py                                                                                                   #
#           Functions used on pandas dataframe.                                                                        #
# #################################################################################################################### #

import pandas
from colorama import Style, Fore


def first_look(df: pandas.DataFrame):
    print(f"Shape: {Fore.LIGHTGREEN_EX}{df.shape}{Fore.RESET}")
    print(df.head())
    print(df.dtypes)


def missing_values(df: pandas.DataFrame, keep_zeros=True):
    data_count = df.shape[0] * df.shape[1]
    missing = missing_df = df.isna().sum()
    missing = round((missing.sum() / data_count) * 100, 2)

    if not keep_zeros:
        missing_df = missing_df[missing_df > 0]

    missing_df = missing_df.sort_values(ascending=False).apply(lambda m: f"{m} ({round((m * 100) / df.shape[0], 2)}%)")

    print(f"Valeurs manquantes: {Fore.LIGHTGREEN_EX}{missing}%")

    if len(missing_df) > 0:
        print(f"{Style.DIM}{Fore.WHITE}{missing_df}{Style.RESET_ALL}")
