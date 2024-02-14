import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def anchor_points_extractor(goal_triples_df: pd.DataFrame, model: SentenceTransformer, filtered_out_triples: list[str]) -> pd.DataFrame:
    goal_triples_df['SCORE'] = goal_triples_df.apply(
        lambda row: pd.Series(
            cosine_similarity(
                model.encode(row['GOAL']).reshape(1, -1),  # goal embedding
                model.encode(row['TRIPLE']).reshape(1, -1),  # simple triples embedding
            ) .flatten() if row['TRIPLE'] not in filtered_out_triples else 0
        ),
        axis=1
    )
    goal_triples_df.sort_values(by=['SCORE'], inplace=True, ascending=False)

    highest_score = goal_triples_df['SCORE'].max()
    anchir_points_df = goal_triples_df[goal_triples_df['SCORE'] >= highest_score * 0.85]

    return anchir_points_df
