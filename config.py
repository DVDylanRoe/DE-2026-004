from dataclasses import dataclass
import os
from pathlib import Path
import polars as pl


@dataclass(frozen=True)
class ColumnConfig:
    numeric_string_columns: tuple[str, ...] = ("Mins", "Pas A", "Ps C")
    numeric_columns: tuple[str, ...] = (
        "Mins",
        "Hdrs A",
        "Clear",
        "Cr A",
        "Drb",
        "FA",
        "Itc",
        "Pas A",
        "Ps C",
        "Shots",
        "Pens",
        "Tck W",
        "Yel",
        "Red",
        "Fls",
        "CCC",
    )
    percentage_columns: tuple[str, ...] = ("Tck R",)
    per_ninety_source_columns: tuple[str, ...] = (
        "Hdrs A",
        "Clear",
        "Cr A",
        "Drb",
        "FA",
        "Itc",
        "Pas A",
        "Ps C",
        "Non Penalty Shots",
        "Tck A",
        "Yel",
        "Red",
        "Fls",
    )
    per_ninety_columns: tuple[str, ...] = (
        "Hdrs A per 90",
        "Clear per 90",
        "Cr A per 90",
        "Drb per 90",
        "Itc per 90",
        "Pas A per 90",
        "Non Penalty Shots per 90",
        "Tck A per 90",
    )
    zscore_feature_columns: tuple[str, ...] = tuple(
        column + " Z" for column in per_ninety_columns
    )


@dataclass(frozen=True)
class TransformContext:
    config: ColumnConfig
    uid: str


@dataclass(frozen=True)
class Baseline:
    chance_creation_rate: float
    pass_completion_rate: float


@dataclass(frozen=True)
class SnowflakeCredentials:
    account: str = os.getenv("SNOWFLAKE_ACCOUNT")
    user: str = os.getenv("SNOWFLAKE_USER")
    authenticator: str = os.getenv("SNOWFLAKE_AUTHENTICATOR")
    role: str = os.getenv("SNOWFLAKE_ROLE")
    warehouse: str = os.getenv("SNOWFLAKE_WAREHOUSE")
    database: str = os.getenv("SNOWFLAKE_DATABASE")
    schema: str = os.getenv("SNOWFLAKE_SCHEMA")


@dataclass
class LoadTarget:
    table: str
    df: pl.DataFrame
    csv_path: Path

    @property
    def csv_path_str(self):
        return str(self.csv_path)
