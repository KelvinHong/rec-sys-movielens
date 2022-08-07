import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os


def get_movie_map():
    df = pd.read_csv("./ml-latest-small/movies.csv")
    ids = df["movieId"]
    titles = df["title"]
    genres = df["genres"]
    return {id: (title, genre) for id, title, genre in zip(ids, titles, genres)}

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
    
def KNN(df, save_as, K=5):
    mat = df.fillna(0).to_numpy()
    ids = df.index.to_numpy()
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(mat)
    distances, indices = nbrs.kneighbors(mat)
    scores = 1/(1+distances)
    ids_indices = ids[indices]
    # Make sure ids_indices rows are started with the correct index
    for i in range(len(ids_indices)):
        row_begin = ids[i]
        if ids_indices[i][0] != row_begin:
            if row_begin in ids_indices[i]:
                # Pull the correct index to index 0.
                to_delete = np.argmax(ids_indices[i] == row_begin)
                ids_indices[i] = np.array([row_begin] + list(np.delete(ids_indices[i], to_delete)))
            else: 
                # Add the correct index at front, 
                # then push back the other five and remove the last one
                ids_indices[i] = np.array([row_begin] + list(ids_indices[i][:4]))
    
    if save_as:
        np.savez(save_as, scores = scores, ids_indices = ids_indices)
    return scores, ids_indices