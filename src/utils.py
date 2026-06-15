import matplotlib.pyplot as plt
import numpy as np

COLOR_PRIMARY = "steelblue"
COLOR_SECONDARY = "darkorange"
COLOR_TERTIARY = "cadetblue"
COLOR_ACCENT = "indianred"
COLOR_REFERENCE = "dimgray"
COLOR_MEAN = "black"


def check_convergence(df):
    is_na = df["energy"].isna()

    na_groups = (~is_na).cumsum()[is_na]
    lengths_consecutive_na = na_groups.groupby(na_groups).size()
    print("\n== Series Convergence Check ==")
    print(lengths_consecutive_na.value_counts().sort_index())
