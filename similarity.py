import polars as pl
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import TransformContext


def extract_player_vector(df: pl.DataFrame, context: TransformContext) -> np.array:
    player = df.filter(pl.col("UID") == context.uid).select(
        context.config.zscore_feature_columns
    )
    player_vector = player.to_numpy().astype(float)
    return player_vector


def extract_feature_matrix(df: pl.DataFrame, context: TransformContext) -> np.array:
    feature_matrix = df.select(context.config.zscore_feature_columns).to_numpy()

    return feature_matrix


def scale_similarity(similarity_scores: np.array) -> np.array:
    similarity_scores_scaled = (similarity_scores * 50) + 50

    return similarity_scores_scaled


def attach_similarity(df: pl.DataFrame, similarity_scores: np.array) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.Series("Similarity", similarity_scores.tolist()),
        ]
    )

    return df


def compute_similarity(df: pl.DataFrame, context: dataclass) -> pl.DataFrame:
    player_vector = extract_player_vector(df, context)

    feature_matrix = extract_feature_matrix(df, context)

    cosine_similarity_scores = cosine_similarity(
        feature_matrix, player_vector
    ).flatten()
    cosine_similarity_scores_scaled = scale_similarity(cosine_similarity_scores)

    df = attach_similarity(df, cosine_similarity_scores_scaled)

    return df


def find_similar_players(df: pl.DataFrame, threshold: int = 90) -> pl.DataFrame:

    similar_df = df.filter(pl.col("Similarity") >= threshold)

    return similar_df
