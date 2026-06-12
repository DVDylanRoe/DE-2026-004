import polars as pl
from dataclasses import dataclass


from cleaning import clean_data
from features import add_features
from similarity import find_similar_players, compute_similarity
from shortlist import create_shortlist
from config import TransformContext


@dataclass
class PipelineResult:
    raw: pl.DataFrame
    transformed: pl.DataFrame
    similar: pl.DataFrame
    shortlist: pl.DataFrame


def transform(df: pl.DataFrame, context: TransformContext):
    transform_df = clean_data(df, context.config)
    transform_df = add_features(transform_df, context)
    transform_df = compute_similarity(transform_df, context)
    similar_df = find_similar_players(transform_df)
    shortlist_df = create_shortlist(similar_df, context)

    return PipelineResult(df, transform_df, similar_df, shortlist_df)
