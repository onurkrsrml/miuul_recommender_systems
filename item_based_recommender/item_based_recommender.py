###############################################################################
# COLLABORATIVE FILTERING (İŞBİRLİKLİ FİLTRELEME)
###############################################################################

# 1. Item-Based Collaborative Filtering (Ürün Bazlı İşbirlikçi Filtreleme, Memory-Based)
# 2. User-Based Collaborative Filtering (Kullanıcı Bazlı İşbirlikçi Filtreleme, Memory-Based)
# 3. Model-Based Collaborative Filtering (Model Bazlı İşbirlikçi Filtreleme, Latent Factor Model)

##########################################################################
# 1. Item-Based Collaborative Filtering (Ürün Bazlı İşbirlikçi Filtreleme)
##########################################################################

# Veri seti: https://grouplens.org/datasets/movielens/

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

###################################
# Adım 1: Veri Setinin Hazırlanması
###################################
import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId") # df oluştur movie(sol) içine rating değişkenlerini sağına ekle
df.head()


##########################################
# Adım 2: User Movie Df'inin Oluşturulması
##########################################

df.head()
df.shape

df["title"].nunique()

df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["count"] <= 1000].index # notlarda "count" yerine "title" vardı ama hata verdi
common_movies = df[~df["title"].isin(rare_movies)] # rare_movies içinde olmayanları al, `~` olmayanları al, `isin` içinde olanlar
common_movies.shape
common_movies["title"].nunique()
df["title"].nunique()


# memory error, yorumluyorum ve arttırıyorum. bunları indirgemek adına userid lerin rating value_count ına göre alt limit ekleyelim
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

# ilgili userid rating value_count indirgeme işlemi
# df["userId"].nunique()
# df["userId"].value_counts().head()
# user_counts = pd.DataFrame(df["userId"].value_counts())
# rare_users = user_counts[user_counts["count"] >= 100].index
# common_users = df[~df["userId"].isin(rare_users)]
# common_users["userId"].nunique()
# df["userId"].nunique()

# user = "118205.0"
# df[df.index == user]

# del a
# df.iloc[df["userId"]["118205.0"]]
# a = df.iloc[df['userId']] == "9254"
# a.value_counts("rating")

# del comment_counts, df, movie, rare_movies, rare_users, rating, user_counts


# common_df = common_movies.merge(common_users, how="left", on="movieId")
# user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.columns


################################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
################################################

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df)


#########################################
# Adım 4: Çalışma Scriptinin Hazırlanması
#########################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 10000].index # memory error, 1000 den 10000 e çektim
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)