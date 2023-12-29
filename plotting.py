from import_modules import *

filepath_titlebasics = 'title.basics.tsv\data.tsv'
filepath_titleratings = 'title.ratings.tsv\data.tsv'
na_vals = ["\\N", "nan"]

df_basics = pd.read_csv(filepath_titlebasics, sep="\t", low_memory=False, na_values=na_vals)
df_ratings = pd.read_csv(filepath_titleratings, sep="\t", low_memory=False, na_values=na_vals)

# showing dist of ratings
plt.figure(figsize=(10,6))
sns.histplot(df_ratings['averageRating'], bins=50, kde=True)
plt.title('Distribution of IMDb Ratings')
plt.xlabel('IMDb rating')
plt.ylabel('Frequency')
plt.show()

# getting genres and displaying unique ones
temp = df_basics.genres.dropna()
vec = CountVectorizer(token_pattern='(?u)\\b[\\w-]+\\b', analyzer='word').fit(temp)
bucket_genres = vec.transform(temp)
unique_genres = vec.get_feature_names_out()
np.array(unique_genres)

# exploring genres and plotting based on percentage
genres = pd.DataFrame(bucket_genres.todense(), columns=unique_genres, index=temp)
genre_perc = 100 * genres.sum() / genres.shape[0]
sorted_genre_perc = genre_perc.sort_values(ascending=False)
plt.figure(figsize=(15,8))
sns.barplot(x=sorted_genre_perc, y=sorted_genre_perc.index, orient="h")
plt.title('Distribution of Genres (%)')
plt.xlabel('Percentage of Films')
plt.ylabel('Genres')
plt.show()

# print(df_basics['startYear'].describe())
# max was 2031, which makes sense so i went with 2023 to match out current year :D

# displaying films by year
sort_by_year = df_basics[df_basics["startYear"].notnull() & (df_basics['startYear'] < 2023)]
count_by_year = sort_by_year.groupby('startYear').agg(numFilms=("tconst", "count"))
max_year = count_by_year["numFilms"].idxmax()

plt.figure(figsize=(10,5))
ax = count_by_year["numFilms"].plot()
plt.title("Number of Films per Year (before 2023)")
plt.xlabel("Year")
plt.ylabel("Number of Films")
plt.show()