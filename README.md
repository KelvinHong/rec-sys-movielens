# Recommendation System for MovieLens data

The data can be downloaded from [here](https://grouplens.org/datasets/movielens/#:~:text=MovieLens%20Latest%20Datasets&text=Small%3A%20100%2C000%20ratings%20and%203%2C600,Last%20updated%209%2F2018.&text=Full%3A%2027%2C000%2C000%20ratings%20and%201%2C100%2C000,relevance%20scores%20across%201%2C100%20tags.), see the "recommended for education and development" section. 
Unzip the data and it should looked like below:
```
.
├── ml-latest-small
|   ├── links.csv
|   ├── movies.csv
|   ├── ratings.csv
|   ├── README.txt
|   └── tags.csv
├── README.md
└── ...
```
## Understand the data

The `movies.csv` file has 3 columns.
It consists of the Movie's id, its title and its genre. 
Note that a movie can belongs to multiple genres, where genres
are seperated by pipes `|` in the file. 
Some of the movie titles has double quotes on both ends, some do not, 
regardless of whether there is whitespace in the title. 
Moreover, it seems very consistently there will be the release year of the movie
at the end of the movie title, in parentheses.  

The `links.csv` file has 3 columns.
It consists of the Movie's id, and its corresponding ids in IMDB and TMDB database. 
To utilize the ID's, view [TMDB API](https://developers.themoviedb.org/3/find/find-by-id) and [Guide for IMDB ID](https://zappiti.uservoice.com/knowledgebase/articles/1979001--identification-use-imdb-id-to-identify-your-movi).

The `tags.csv` file has 4 columns.
It consists of the User ID, Movie ID, Tag, and timestamp.
It represent the time (timestamp) when a single user assign a tag he/she seems relevant to the movie. 

The `ratings.csv` file has 4 columns.
It consists of the User ID, Movie ID, rating, and timestamp.
Rating is a float from 1 to 5, often by 0.5 increment. 

## Analyze the data

Run `analyze.py` after downloaded all necessary CSV files. 
It will gives a summary of the data. 

We found that in `links.csv`, the column `tmdbId` has 8 missing values out of 9742. 
Therefore, it is safe to just remove them without affecting the quality of the model. 

## Computation

Run `main.py` to calculate similarities with various methods based on users and items. 
Currently Cosine similarity and Pearson Correlation are implemented. 

See `./output/` for the output data.

## K-Nearest Neighbors for Movies

[Under development]