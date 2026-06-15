import pandas as pd


def format_time_series(df):
    df["Datetime"] = pd.to_datetime(
        df["Datetime"], errors="raise", format="%Y-%m-%d %H:%M:%S"
    )
    df.rename(columns={"PJME_MW": "energy"}, inplace=True)
    return df


def reindex_time_series(df):
    start_date = df["Datetime"].min()
    end_date = df["Datetime"].max()
    n_expected_hours = int((end_date - start_date).total_seconds() // 3600) + 1
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="h")

    print("== Date Range ==")
    print(f"Start: {start_date}")
    print(f"End:   {end_date}")
    print(f"Expected hourly observations: {n_expected_hours:,}")

    df.set_index("Datetime", inplace=True)

    n_duplicates = df.index.duplicated().sum()
    print(f"== Duplicate Timestamps: {n_duplicates} ==")
    if n_duplicates:
        print(df[df.index.duplicated()])

    df = df[~df.index.duplicated()]
    df = df.reindex(full_date_range)

    print(f"Rows after reindexing: {len(df):,}")
    return df


def impute_time_series(df):
    df["energy"] = df["energy"].interpolate(method="linear")
    return df
