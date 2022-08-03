from sklearn.metrics.pairwise import cosine_similarity

def similarity(df, method = "cosine"):
    if method == "cosine":
        ret = cosine_similarity(df)
    
    return ret