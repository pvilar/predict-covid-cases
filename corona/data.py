""" Defines functions for importing data"""

from pathlib import Path
from functools import reduce
import logging
import inflection
import numpy as np
import pandas as pd

logger = logging.getLogger()


def read_covid_cases(
    countries: list = None
) -> pd.DataFrame:
    """
    Returns number of COVID-19 confirmed cases, deaths, and recovered cases
    as a pandas DataFrame. Data will be streamed from Github every time this
    function is called to ensure the most recent data is available.

    Args
      countries: A list of countries to subset by

    Returns
      df_cases: A pandas DataFrame

    """

    base_url = (
        "https://raw.githubusercontent.com/CSSEGISandData/"
        "COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
        "time_series_19-covid-{field}.csv"
    )

    dfs = list()

    for field in ["Confirmed", "Recovered", "Deaths"]:
        df_cases_ = pd.melt(
            pd.read_csv(base_url.format(field=field)),
            id_vars=["Province/State", "Country/Region", "Lat", "Long"],
            var_name="Date",
            value_name=field,
        )

        df_cases_.columns = [
            inflection.underscore(col).replace(" ", "_").replace("/", "_")
            for col in df_cases_.columns
        ]

        df_cases_["date"] = pd.to_datetime(df_cases_["date"])
        dfs.append(df_cases_)

    df_cases = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=["date", "country_region", "province_state", "lat", "long"],
            how="left",
        ),
        dfs,
    )


    mask = (df_cases[["confirmed", "recovered", "deaths"]] > 0).any(1)
    df_cases = df_cases[mask]
    df_cases[["country_region", "province_state"]] = df_cases[
        ["country_region", "province_state"]
    ].fillna("UNKNOWN")

    df_cases[["confirmed", "recovered", "deaths"]] = df_cases[
        ["confirmed", "recovered", "deaths"]
    ].fillna(0)

    for column in df_cases.dtypes[df_cases.dtypes == "object"].index.values:
        df_cases[column] = df_cases[column].str.upper()

    if countries:
        countries = [c.upper() for c in countries]

        groupby = ["country_region"]

        for country in countries:
            if country not in df_cases["country_region"].unique():
                countries_ = str(df_cases["country_region"].unique())
                error = "Country " + country + " is not in \n" + countries_
                logger.error(error)
                raise ValueError(error)

        df_cases = df_cases[df_cases["country_region"].isin(countries)]

    else:
        groupby = ["country_region", "province_state"]

    df_cases = df_cases.groupby(["date"] + groupby).sum().reset_index()

    df_cases["virus_active"] = np.where(df_cases["confirmed"] > 0, 1, 0)
    df_cases["virus_active20"] = np.where(df_cases["confirmed"] > 20, 1, 0)

    df_cases["days_active"] = df_cases.groupby(groupby)["virus_active"].apply(
        lambda x: x.cumsum()
    )
    df_cases["days_active20"] = df_cases.groupby(groupby)["virus_active20"].apply(
        lambda x: x.cumsum()
    )

    df_cases["total_active"] = df_cases["confirmed"] - df_cases["recovered"]
    df_cases["new_cases"] = df_cases.groupby(groupby)["confirmed"].diff()

    df_cases = df_cases[
        ["date"]
        + groupby
        + [
            "confirmed",
            "recovered",
            "deaths",
            "virus_active",
            "virus_active20",
            "days_active",
            "days_active20",
            "total_active",
            "new_cases",
        ]
    ]

    return df_cases
