# Recommendation System for MovieLens data

The data can be downloaded from [here](https://grouplens.org/datasets/movielens/25m/). 
Unzip the data and it should looked like below:
.
├── ml-25m
│   ├── genome-scores.csv
│   ├── genome-tags.csv
|   ├── links.csv
|   ├── movies.csv
|   ├── ratings.csv
|   ├── README.txt
|   └── tags.csv
├── README.md
└── ...

## Understand the data

The `genome-tags.csv` file has 2 columns. 
It is a simple id mapping to the tag of movies. 
Some example tags are `1920s, 1930s, adventure, amnesia`. 
The tags seems to be different than genres of the movies. 

The `movies.csv` file has 3 columns.
It consists of the Movie's id, its title and its genre. 
Note that a movie can belongs to multiple genres, where genres
are seperated by pipes `|` in the file. 
Some of the movie titles has double quotes on both ends, some do not, 
regardless of whether there is whitespace in the title. 
Moreover, it seems very consistently there will be the release year of the movie
at the end of the movie title, in parentheses.  

The `genome-scores.csv` file has 3 columns. 
It consists of the Movie's id, tag's id, and a score within 0 to 1 describing
the relevance of the movie to the tag.

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

We found that in `links.csv`, the column `tmdbId` has 107 missing values out of 62423. 
Then in `tags.csv`, the column `tag` has 16 missing values out of 1093360.
Therefore, it is safe to just remove them without affecting the quality of the model. 
