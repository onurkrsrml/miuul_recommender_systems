#######################################################
# Content Based Recommendation (İçerik Temelli Tavsiye)
######################################################

##############################################
# Film Overview'larına Göre Tavsiye Geliştirme
##############################################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

####################################
# 1. TF-IDF Matrisinin Oluşturulması
####################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False) # tek satırda göster
from sklearn.feature_extraction.text import TfidfVectorizer # text verileri vektöre çevirme
from sklearn.metrics.pairwise import cosine_similarity # benzerlik ölçme
# https://www.kaggle.com/rounakbanik/the-movies-dataset


df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapatmak icin
df.head()
df.shape

df["overview"].head()

tfidf = TfidfVectorizer(stop_words="english")

# df[df['overview'].isnull()]
df['overview'] = df['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df['overview'])

tfidf_matrix.shape

df['title'].shape

tfidf.get_feature_names()
# tfidf.get_feature_names_out()

tfidf_matrix.toarray()



###############################################
# 2. Cosine Similarity Matrisinin Oluşturulması
###############################################

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

cosine_sim.shape
cosine_sim[1]



############################################
# 3. Benzerliklere Göre Önerilerin Yapılması
############################################


indices = pd.Series(df.index, index=df['title'])

indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep='last')]

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index # 0. index kendisi olacagi icin 1 den basladik

df['title'].iloc[movie_indices]

# df["title"].iloc[34737] # yabancı karakterli dizi ismi nasıl similarity hesaplandı???

####################################
# 4. Çalışma Scriptinin Hazırlanması
####################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesaplama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)

# Büyük veri seti ile çalışırken hali hazırda oluşturulmuş olan tfidf_matrix ve cosine_sim değerlerinden çıkarılmış öneri
# sonuçları SQL veritabanında tutulabilir ve bu veritabanından çekilerek kullanılabilir. Bu sayede her seferinde bu işlemleri
# tekrar tekrar yapmamıza gerek kalmaz. Bu sayede öneri sonuçları daha hızlı bir şekilde elde edilebilir.
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3 ...

