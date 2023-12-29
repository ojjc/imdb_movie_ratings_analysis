from import_modules import *

filepath_titlebasics = 'title.basics.tsv\data.tsv'
filepath_titleratings = 'title.ratings.tsv\data.tsv'
na_vals = ["\\N", "nan"]

df_basics = pd.read_csv(filepath_titlebasics, sep="\t", low_memory=False, na_values=na_vals)
df_ratings = pd.read_csv(filepath_titleratings, sep="\t", low_memory=False, na_values=na_vals)
df_basrat = pd.merge(df_basics, df_ratings, on='tconst')

# preprocess data
df_basrat = df_basrat.dropna(subset=['averageRating', 'genres'])
df_basrat['genres'] = df_basrat['genres'].apply(lambda x : x.split(','))

# using feature engineering
mlb = MultiLabelBinarizer()
genre_encoding = pd.DataFrame(mlb.fit_transform(df_basrat['genres']), columns=mlb.classes_, index=df_basrat.index)
df_basrat = pd.concat([df_basrat, genre_encoding], axis=1)

# select features and target
features = genre_encoding.columns
x = df_basrat[features]
y = df_basrat['averageRating']

# data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# model selection and training
model = RandomForestRegressor()
model.fit(x_train, y_train)

# eval model
start_time = time.time()
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Square Error: {mse}')

# analyze features of importance
feat = pd.Series(model.feature_importances_, index=features)
sorted_feat = feat.sort_values(ascending=True)

# calculate time to compute importances
elapsed_time = time.time() - start_time
print(f"Time taken to compute importances: {elapsed_time:.3f} seconds")

# essentially displays which genre has a significant impact on ratings
plt.figure(figsize=(12,8))
sorted_feat.plot(kind='barh', color='skyblue')
plt.title('Impact of Genre on Movie Ratings')
plt.xlabel('Feature Importance')
plt.ylabel('Genre')
plt.show()