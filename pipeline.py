import polars as pl
from dataclasses import dataclass


from cleaning import clean_data
from features import add_dervied_columns
from similarity import find_similar_players
from shortlist import create_shortlist
from config import TransformContext


def transform(df: pl.DataFrame, context: TransformContext):
    transform_df = clean_data(df, context.config)
    transform_df = add_dervied_columns(transform_df, context)
    similar_df = find_similar_players(transform_df)
    shortlist_df = create_shortlist(similar_df, context)
    return transform_df, similar_df, shortlist_df