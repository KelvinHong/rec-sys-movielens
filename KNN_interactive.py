import pandas as pd
import numpy as np
import os
from main import KNN, preprocess
import PySimpleGUI as sg

def check_similar_movie(df):
    movie_ids = df.index.to_numpy()
    num_movie = len(movie_ids)
    print(movie_ids[:5], movie_ids[-5:])
    layout = [
        [
            sg.Text("Choose a movie ID (1-193609): "),
            sg.Input(default_text=100, key="movieId", enable_events=True),    
        ]
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
            print(movie_ids[observing_inds])
            


if __name__ == "__main__":
    os.makedirs("./output/", exist_ok=True)
    file_name = "./ml-latest-small/ratings.csv"
    # Pivot rating.csv into a 2D-matrix.
    item_row_df = preprocess(file_name, index="movieId", save_as = "./output/movie-user-rating.csv")
    
    check_similar_movie(item_row_df)