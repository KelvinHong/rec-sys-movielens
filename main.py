import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(file_name, index = "movieId", save_as = None) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    # movieIDs = df["movieId"].unique()
    # userIDs = df["userId"].unique()
    
    # Aggregate data into 2-dimensional matrix
    assert index in ["userId", "movieId"]
    if index == "movieId":
        user_movie_mat = df.pivot(
            index="movieId",
            columns="userId",
            values="rating"
        )
    else:
        user_movie_mat = df.pivot(
            index="userId",
            columns="movieId",
            values="rating"
        )
    print("User-Movie relation table has shape ", user_movie_mat.shape)
    total_count = user_movie_mat.shape[0] * user_movie_mat.shape[1]
    print("total cells", total_count)
    missing_count = user_movie_mat.isna().sum().sum()
    print(f"Missing values count is {missing_count}, makes up {round(100 * missing_count / total_count, 2)}% of the data.")
    
    if save_as is not None:
        user_movie_mat.to_csv(save_as)

    return user_movie_mat

if __name__ == "__main__":
    os.makedirs("./output/", exist_ok=True)
    
    file_name = "./ml-latest-small/ratings.csv"
    # Pivot rating.csv into a 2D-matrix.
    processed_df = preprocess(file_name, save_as = "./output/user-movie-rating.csv")
    sim_mat = cosine_similarity(processed_df.fillna(0))
    print("Dataframe shape is", processed_df.shape)
    print("Similarity matrix shape is", sim_mat.shape)
    print(sim_mat[:5, :5])
    print(type(sim_mat))
    print("Count of zeros in similarity matrix is ", np.count_nonzero(sim_mat == 0))