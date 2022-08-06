import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess(file_name: str, index: str = "movieId", save_as: str = None) -> pd.DataFrame:
    """Take a file with columns "userId", "movieId", "rating", 
    then pivot them to create a rating table. 

    Args:
        file_name (str): The original file to take rating from.
        index (str, optional): The index for the dataframe, row names. Defaults to "movieId".
        save_as (str, optional): Save the pivoted table as csv file. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
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

def visualize_cosine(df: pd.DataFrame, csv: str, png: str, fmt: str = "%.3f"):
    """Visualize the result of cosine similarity.

    Args:
        df (pd.DataFrame): The pivoted data frame obtained from the preprocess function.
        csv (str): The path to save cosine similarity matrix as csv file.
        png (str): The path to save visualization of cosine similarity.
        fmt (str, optional): Format of floats saving into the csv file. Defaults to "%.3f".
    """
    sim_mat = cosine_similarity(df.fillna(0))
    np.savetxt(csv, sim_mat, delimiter=",", fmt=fmt)
    print(f"Dataframe shape {df.shape}, siilarity matrix {sim_mat.shape}.")
    zero_count = np.count_nonzero(sim_mat == 0)
    print(f"Count of zeros in similarity matrix is {zero_count}, {round(100 * zero_count / (sim_mat.shape[0] * sim_mat.shape[1]), 2)}%.")
    plt.matshow(sim_mat)
    plt.savefig(png)
    print(f"Similarity matrix saved at {os.path.abspath(csv)} .")
    print(f"Similarity visualization saved at {os.path.abspath(png)} .")
    
def KNN(df, K=5):
    mat = df.fillna(0).to_numpy()
    ids = df.index.to_numpy()
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(mat)
    distances, indices = nbrs.kneighbors(mat)
    scores = 1/(1+distances)
    ids_indices = ids[indices]
    return scores, ids_indices

if __name__ == "__main__":
    os.makedirs("./output/", exist_ok=True)
    
    file_name = "./ml-latest-small/ratings.csv"
    # Pivot rating.csv into a 2D-matrix.
    item_row_df = preprocess(file_name, index="movieId", save_as = "./output/movie-user-rating.csv")
    user_row_df = preprocess(file_name, index="userId", save_as = "./output/user-movie-rating.csv")
    
    # Cosine similarity based on user and item separately.
    # Note that Cosine Similarity is based on rows.
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

    # # Pearson Correlation based on user and item separately.
    # # Note that Pearson Correlation is based on columns.
    print("item based Pearson Correlation:")
    mat_pitem = user_row_df.corr(method="pearson")
    mat_pitem.to_csv("./output/itembased_pearson.csv", float_format = "%.3f")

    print("User based Pearson Correlation:")
    mat_puser = item_row_df.corr(method="pearson")
    mat_puser.to_csv("./output/userbased_pearson.csv", float_format = "%.3f")

    # K nearest neighbor method base on user and item separately.
    # Note that KNN is based on rows.
    K = 5
    print("User based KNN")
    uscores, uids_indices = KNN(user_row_df, K=K)

    print("Item based KNN")
    iscores, iids_indices = KNN(item_row_df, K=K)