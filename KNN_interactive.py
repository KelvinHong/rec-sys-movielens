import pandas as pd
import numpy as np
import os
import PySimpleGUI as sg
from utils import *

def check_similar_movie(df, iscore, iids, movie_map):
    movie_ids = df.index.to_numpy()
    num_movie = len(movie_ids)
    buttons = [
        [
            sg.Button("-", key=f"button-{id}", enable_events = True),
            sg.Text("-", key=f"movie-{id}"),
        ] for id in range(1,6)
    ]
    recommendations_meta = [
        [
            [
                sg.Text("Rank 1: [ID]None -- [Title]None (None)", key=f"rec-{i}")
            ],
            [
                sg.Text("\t[Genre(s)]Any", key=f"gen-{i}")
            ]
        ] for i in range(1,6)
    ]
    recommendations = []
    for rec in recommendations_meta:
        recommendations += rec
    layout = [
        [
            sg.Text("Similar Movie Finder", font=("Any", 17))
        ],
        [
            sg.Text("Choose a movie ID (1-193609): "),
            sg.Input(default_text=100, key="movieId", enable_events=True),    
        ],
        *buttons,
        [
            sg.Text("Top-5 movies:", font=("Any", 17))
        ],
        *recommendations,
    ]

    window = sg.Window("Similar Movie Finder", layout)
    window.finalize()
    window.bind("<Escape>", "-ESCAPE-")

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "-ESCAPE-"):
            break
        elif event == "movieId":
            try:
                id = int(values["movieId"])
            except ValueError: # If user provide non-numeric, skip it.
                continue
            # Get index that is just large than user input
            slight_large_ind = np.argmax(movie_ids >= id)
            if id > max(movie_ids):
                slight_large_ind = num_movie
            if slight_large_ind <= 2:
                observing_inds = list(range(0,5))
            elif slight_large_ind >= num_movie - 3:
                observing_inds = list(range(num_movie-5, num_movie))
            else:
                observing_inds = list(range(slight_large_ind-2, slight_large_ind+3))
            movie_titles = [movie_map[id][0] for id in movie_ids[observing_inds]]
            for i in range(5):
                window[f"button-{i+1}"].Update(movie_ids[observing_inds][i])
                window[f"movie-{i+1}"].Update(movie_titles[i])
        elif "button" in event:
            try:
                movie_id = int(window[event].get_text())
            except ValueError:
                sg.popup("Please first enter some movie ID for searching. ")
                continue
            # movie_title = movie_map[movie_id]
            # Show similar movies by index
            in_row = np.where(movie_ids == movie_id)[0][0]
            sim_movie_ids = iids[in_row]
            for j, ID in enumerate(sim_movie_ids):
                window[f"rec-{j+1}"].Update(f"Rank {j+1}: [ID]{ID} -- [Title]{movie_map[ID.item()][0]}")
                window[f"gen-{j+1}"].Update(f"\t[Genre(s)]{', '.join(movie_map[ID.item()][1].split('|'))}")
            

if __name__ == "__main__":
    os.makedirs("./output/", exist_ok=True)
    file_name = "./ml-latest-small/ratings.csv"
    # Pivot rating.csv into a 2D-matrix.
    print("Loading rating dataframe...")
    item_row_df = preprocess(file_name, index="movieId", save_as = "./output/movie-user-rating.csv")
    save_file = "./output/knn-itembased.npz"
    if os.path.isfile(save_file):
        print("Existing data found, load it instead.")
        file_data = np.load(save_file)
        iscore = file_data["scores"]
        iids = file_data["ids_indices"]
    else:
        print("Calculating nearest neighbors... (Might take up to 1 minute)")
        iscore, iids = KNN(item_row_df, save_as = save_file)
    print("Retrieving movies ID mapping...")
    movie_map = get_movie_map()
    check_similar_movie(item_row_df, iscore, iids, movie_map)