import os
from utils import *

if __name__ == "__main__":
    os.makedirs("./output/", exist_ok=True)

    file_name = "./ml-latest-small/ratings.csv"
    # Pivot rating.csv into a 2D-matrix.
    item_row_df = preprocess(file_name,
                             index="movieId",
                             save_as="./output/movie-user-rating.csv")
    user_row_df = preprocess(file_name,
                             index="userId",
                             save_as="./output/user-movie-rating.csv")

    # Cosine similarity based on user and item separately.
    # Note that Cosine Similarity is based on rows.
    print("Item based cosine similarity:")
    visualize_cosine(
        item_row_df,
        csv="./output/itembased_similarity.csv",
        png="./output/itembased_corr.png",
    )

    print("User based cosine similarity:")
    visualize_cosine(
        user_row_df,
        csv="./output/userbased_similarity.csv",
        png="./output/userbased_corr.png",
    )

    # # Pearson Correlation based on user and item separately.
    # # Note that Pearson Correlation is based on columns.
    print("item based Pearson Correlation:")
    mat_pitem = user_row_df.corr(method="pearson")
    mat_pitem.to_csv("./output/itembased_pearson.csv", float_format="%.3f")

    print("User based Pearson Correlation:")
    mat_puser = item_row_df.corr(method="pearson")
    mat_puser.to_csv("./output/userbased_pearson.csv", float_format="%.3f")

    # K nearest neighbor method base on user and item separately.
    # Note that KNN is based on rows.
    K = 5
    print("User based KNN")
    userbased_save_as = "./output/knn-userbased.npz"
    uscores, uids_indices = KNN(user_row_df, userbased_save_as, K=K)

    print("Item based KNN")
    itembased_save_as = "./output/knn-itembased.npz"
    iscores, iids_indices = KNN(item_row_df, itembased_save_as, K=K)
