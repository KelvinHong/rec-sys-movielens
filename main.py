import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def preprocess(file_name, index = "movieId", save_as = None) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    # movieIDs = df["movieId"].unique()
    # userIDs = df["userId"].unique()
    
    # Aggregate data into 2-dimensional matrix
    assert index in ["userId", "movieId"]
    if index == "movieId":
        mat = df.pivot(
            index="movieId",
            columns="userId",
            values="rating"
        )
    else:
        mat = df.pivot(
            index="userId",
            columns="movieId",
            values="rating"
        )
    if save_as is not None:
        mat.to_csv(save_as)

    return mat

def visualize_cosine(df, csv, png, fmt = "%.3f"):
    sim_mat = cosine_similarity(df.fillna(0))
    np.savetxt(csv, sim_mat, delimiter=",", fmt=fmt)
    print(f"Dataframe shape {df.shape}, siilarity matrix {sim_mat.shape}.")
    zero_count = np.count_nonzero(sim_mat == 0)
    print(f"Count of zeros in similarity matrix is {zero_count}, {round(100 * zero_count / (sim_mat.shape[0] * sim_mat.shape[1]), 2)}%.")
    plt.matshow(sim_mat)
    plt.savefig(png)
    print(f"Similarity matrix saved at {os.path.abspath(csv)} .")
    print(f"Similarity visualization saved at {os.path.abspath(png)} .")
    

if __name__ == "__main__":
    os.makedirs("./output/", exist_ok=True)
    
    file_name = "./ml-latest-small/ratings.csv"
    # Pivot rating.csv into a 2D-matrix.
    item_row_df = preprocess(file_name, index="movieId", save_as = "./output/movie-user-rating.csv")
    user_row_df = preprocess(file_name, index="userId", save_as = "./output/user-movie-rating.csv")
    
    # Cosine similarity based on user and item separately.
    print("Item based cosine similarity:")
    visualize_cosine(
        item_row_df, 
        csv = "./output/itembased_similarity.csv", 
        png = "./output/itembased_corr.png", 
    )

    print("User based cosine similarity:")
    visualize_cosine(
        user_row_df, 
        csv = "./output/userbased_similarity.csv", 
        png = "./output/userbased_corr.png", 
    )

    # Pearson Similarity based on user and item separately.

    # K nearest neighbor method base on user and item separately.