########################################################################################################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
########################################################################################################################

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

###################
# Apriori Algorithm
###################

# Support (X, Y) = Freq (X, Y) / N
# X ve Y' nin birlikte görülme olasılığı

# Confidence ( X, Y) = Frew (X, Y) / Freq (X)
# X satın alındığında Y' nin satılması olasılığı)

# Lift = Support (X, Y) / (Support(X) * Support(Y)
# X satın alındığında Y' nin satın alınma olasılığı lift kadar artar

########################################
# 1. Veri Ön İşleme (Data Preprocessing)
########################################

# def outlier_thresholds(dataframe, variable):
#     quartile1 = dataframe[variable].quantile(0.01)
#     quartile3 = dataframe[variable].quantile(0.99)
#     interquantile_range = quartile3 - quartile1
#     up_limit = quartile3 + 1.5 * interquantile_range
#     low_limit = quartile1 - 1.5 * interquantile_range
#     return low_limit, up_limit
#
# def replace_with_thresholds(dataframe, variable):
#     low_limit, up_limit = outlier_thresholds(dataframe, variable)
#     dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
#     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
#
# def retail_data_prep(dataframe):
#     dataframe.dropna(inplace=True)
#     dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
#     dataframe = dataframe[dataframe["Quantity"] > 0]
#     dataframe = dataframe[dataframe["Price"] > 0]
#     replace_with_thresholds(dataframe, "Quantity")
#     replace_with_thresholds(dataframe, "Price")
#     return dataframe
#
# df = retail_data_prep(df)
# df.isnull().sum()
# df.describe().T
#
# def create_invoice_product_df(dataframe, id=False):
#     if id:
#         return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
#             applymap(lambda x: 1 if x > 0 else 0)
#     else:
#         return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
#             applymap(lambda x: 1 if x > 0 else 0)
#
# fr_inv_pro_df = create_invoice_product_df(df_fr)
#
# fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)
#
#
# def check_id(dataframe, stock_code):
#     product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
#     print(product_name)
#
# check_id(df_fr, 10120)



#########################################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
#########################################################

# df_fr = df[df['Country'] == "France"]
#
# df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]
#
# def create_invoice_product_df(dataframe, id=False):
#     if id:
#         return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
#             applymap(lambda x: 1 if x > 0 else 0)
#     else:
#         return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
#             applymap(lambda x: 1 if x > 0 else 0)
#
# fr_inv_pro_df = create_invoice_product_df(df_fr)
#
# def check_id(dataframe, stock_code):
#     product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
#     print(product_name)
#
# check_id(df_fr, 10120)



#########################################
# 3. Birliktelik Kurallarının Çıkarılması
#########################################

# frequent_itemsets = apriori(fr_inv_pro_df,
#                             min_support=0.01,
#                             use_colnames=True)

# frequent_itemsets.sort_values("support", ascending=False)

# # metric: support, confidence, lift, leverage, conviction
# rules = association_rules(frequent_itemsets,
#                           metric="support",
#                           min_threshold=0.01)

# rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
# sort_values("confidence", ascending=False)



###################################
# 4. Çalışmanın Scriptini Hazırlama
###################################

# def outlier_thresholds(dataframe, variable):
#     quartile1 = dataframe[variable].quantile(0.01)
#     quartile3 = dataframe[variable].quantile(0.99)
#     interquantile_range = quartile3 - quartile1
#     up_limit = quartile3 + 1.5 * interquantile_range
#     low_limit = quartile1 - 1.5 * interquantile_range
#     return low_limit, up_limit
#
# def replace_with_thresholds(dataframe, variable):
#     low_limit, up_limit = outlier_thresholds(dataframe, variable)
#     dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
#     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
#
# def retail_data_prep(dataframe):
#     dataframe.dropna(inplace=True)
#     dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
#     dataframe = dataframe[dataframe["Quantity"] > 0]
#     dataframe = dataframe[dataframe["Price"] > 0]
#     replace_with_thresholds(dataframe, "Quantity")
#     replace_with_thresholds(dataframe, "Price")
#     return dataframe
#
#
# def create_invoice_product_df(dataframe, id=False):
#     if id:
#         return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
#             applymap(lambda x: 1 if x > 0 else 0)
#     else:
#         return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
#             applymap(lambda x: 1 if x > 0 else 0)
#
#
# def check_id(dataframe, stock_code):
#     product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
#     print(product_name)
#
#
# def create_rules(dataframe, id=True, country="France"):
#     dataframe = dataframe[dataframe['Country'] == country]
#     dataframe = create_invoice_product_df(dataframe, id)
#     frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
#     rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
#     return rules
#
# df = df_.copy()
#
# df = retail_data_prep(df)
# rules = create_rules(df)
#
# rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
# sort_values("confidence", ascending=False)



##############################################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
##############################################################

# !!!!!!****** DİKKAT ******!!!!!!

# product_id = 22492
# check_id(df, product_id)
#
# # burada amaç ürün id si girilen ürünün birliktelik kurallarından yola çıkarak öneri yapmak
# # ürün id si girilen ürünün antecedents leri arasında varsa consequents leri öneri listesine ekle
# # öneri listesini döndür
#
# # döngü içinde product_id bulunduğunda ilgili index i tutacak, product ürünleri kontrol ederken değerini tutacak, enumerate ile
# # sorted_rules["antecedents"] içindeki ürünlerin indexleri ile/dahil bütün değişkenlerini dönecek,
# # diğer döngü j product içindeki ürünleri dönecek, eğer j product_id ye eşitse kontrol yapacak,
# # eğer eşitse recommendation_list e ekleme yapacak, ekleme yaparken de sorted_rules ın iloc sayesinde ilk döngüde tutulan
# # ve product_id ile eşleşen ürünün antecedents index ideki consequents değerini ilk elemanını alacak ve boş listeye ekleme yapacak
#
# # girilen ürün id si ni antecedents içinde ara(çoklu kombinasyonlar içinde olsa bile)
# # bu esnada o ürünün index ini de hafızada tut
# # id yi bulduğun antecedents e karşılık gelen consequents değerini recommendation_list e eklememiz lazım ama nasıl
# # ilgili index i kullanarak sorted_rules ın
# # ilgili index i kullanarak sorted_rules ın ilgili index indeki consequents değerini recommendation_list e ekleyebiliriz
# # döngü haline getirip ilk 3 ürünü öneri listesine ekleyebiliriz
#
# def arl_recommender(rules_df, product_id, rec_count=1):
#     sorted_rules = rules_df.sort_values("lift", ascending=False)
#     recommendation_list = []
#     for i, product in enumerate(sorted_rules["antecedents"]):
#         for j in list(product):
#             if j == product_id:
#                 recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
#
#     return recommendation_list[0:rec_count]
#
#
# arl_recommender(rules, 22492, 1)
# arl_recommender(rules, 22492, 2)
# arl_recommender(rules, 22492, 3)
# check_id(df, 22326)





########################################################################################################################
# CONTENT BASED RECOMMENDATION (İÇERİK TEMELLİ TAVSİYE)
########################################################################################################################

##############################################
# Film Overview'larına Göre Tavsiye Geliştirme
##############################################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

####################################
# 4. Çalışma Scriptinin Hazırlanması
####################################

# def content_based_recommender(title, cosine_sim, dataframe):
#     # index'leri olusturma
#     indices = pd.Series(dataframe.index, index=dataframe['title'])
#     indices = indices[~indices.index.duplicated(keep='last')]
#     # title'ın index'ini yakalama
#     movie_index = indices[title]
#     # title'a gore benzerlik skorlarını hesaplama
#     similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
#     # kendisi haric ilk 10 filmi getirme
#     movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
#     return dataframe['title'].iloc[movie_indices]
#
# content_based_recommender("Sherlock Holmes", cosine_sim, df)
#
# content_based_recommender("The Matrix", cosine_sim, df)
#
# content_based_recommender("The Godfather", cosine_sim, df)
#
# content_based_recommender('The Dark Knight Rises', cosine_sim, df)
#
#
# def calculate_cosine_sim(dataframe):
#     tfidf = TfidfVectorizer(stop_words='english')
#     dataframe['overview'] = dataframe['overview'].fillna('')
#     tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
#     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#     return cosine_sim
#
#
# cosine_sim = calculate_cosine_sim(df)
# content_based_recommender('The Dark Knight Rises', cosine_sim, df)
#
# Büyük veri seti ile çalışırken hali hazırda oluşturulmuş olan tfidf_matrix ve cosine_sim değerlerinden çıkarılmış öneri
# sonuçları SQL veritabanında tutulabilir ve bu veritabanından çekilerek kullanılabilir. Bu sayede her seferinde bu işlemleri
# tekrar tekrar yapmamıza gerek kalmaz. Bu sayede öneri sonuçları daha hızlı bir şekilde elde edilebilir.
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3 ...





########################################################################################################################
# COLLABORATIVE FILTERING (İŞBİRLİKLİ FİLTRELEME)
########################################################################################################################

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

# def create_user_movie_df():
#     import pandas as pd
#     movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
#     rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
#     df = movie.merge(rating, how="left", on="movieId")
#     comment_counts = pd.DataFrame(df["title"].value_counts())
#     rare_movies = comment_counts[comment_counts["title"] <= 10000].index # count yerine title, 10000 yerine 1000 vardı
#     common_movies = df[~df["title"].isin(rare_movies)]
#     user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
#     return user_movie_df
#
# user_movie_df = create_user_movie_df()
#
#
# def item_based_recommender(movie_name, user_movie_df):
#     movie_name = user_movie_df[movie_name]
#     return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)
#
# item_based_recommender("Matrix, The (1999)", user_movie_df)
#
# movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
#
# item_based_recommender(movie_name, user_movie_df)





##########################################################################
# User-Based Collaborative Filtering
##########################################################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

# def create_user_movie_df():
#     import pandas as pd
#     movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
#     rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
#     df = movie.merge(rating, how="left", on="movieId")
#     comment_counts = pd.DataFrame(df["title"].value_counts())
#     rare_movies = comment_counts[comment_counts["title"] <= 1000].index
#     common_movies = df[~df["title"].isin(rare_movies)]
#     user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
#     return user_movie_df
#
# user_movie_df = create_user_movie_df()
#
# # perc = len(movies_watched) * 60 / 100
# # users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
#
#
#
# def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
#     import pandas as pd
#     random_user_df = user_movie_df[user_movie_df.index == random_user]
#     movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
#     movies_watched_df = user_movie_df[movies_watched]
#     user_movie_count = movies_watched_df.T.notnull().sum()
#     user_movie_count = user_movie_count.reset_index()
#     user_movie_count.columns = ["userId", "movie_count"]
#     perc = len(movies_watched) * ratio / 100
#     users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
#
#     final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
#                           random_user_df[movies_watched]])
#
#     corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
#     corr_df = pd.DataFrame(corr_df, columns=["corr"])
#     corr_df.index.names = ['user_id_1', 'user_id_2']
#     corr_df = corr_df.reset_index()
#
#     top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
#         ["user_id_2", "corr"]].reset_index(drop=True)
#
#     top_users = top_users.sort_values(by='corr', ascending=False)
#     top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
#     rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
#     top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
#     top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
#
#     recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
#     recommendation_df = recommendation_df.reset_index()
#
#     movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
#     movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
#     return movies_to_be_recommend.merge(movie[["movieId", "title"]])
#
#
#
# random_user = int(pd.Series(user_movie_df.index).sample(1).values)
# user_based_recommender(random_user, user_movie_df)
# user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)





########################################################################################################################
# Model-Based Collaborative Filtering: Matrix Factorization
########################################################################################################################

# Bir fonksiyonun bir noktadaki türevi o fonksiyonun o noktadaki maksimum artış yönünü verir. Artış yönü/türevi/gradient in
# negatif yönüne negatif olarak gittiğimizde ilgili fonksiyonu minimum laştıracak şekilde parametre değerlerini güncelliyor oluruz.
# Gradient Descent, yakınsayana/yaklaşana kadar tekrar et, bir fonk daki parametrenin değerlerini iteratif olarak değiştir.
# Nasıl? Belirli bir hız ile değiştir. İlgili parametrenin türevini alırsan türevi aldığın noktadaki değerine göre belirli
# bir hızla bu parametrenin değerini önceki değeri ile değiştirerek güncelle. Bu işlemi yaparken de bir learning rate
# dediğimiz bir hız belirliyoruz. Bu hız ne kadar büyük olursa o kadar hızlı öğreniriz ama aşırı büyük olursa da
# aşırı hızlı öğreniriz ve asla yakınsayamayız. Bu hız ne kadar küçük olursa o kadar yavaş öğreniriz ama yakınsama
# garantimiz olur. Bu hızı belirlerken de bir çok deneme yaparak en uygun hızı bulmamız gerekmektedir.

# # !pip install surprise
# import pandas as pd
# from surprise import Reader, SVD, Dataset, accuracy
# from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
# pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

###################################
# Adım 1: Veri Setinin Hazırlanması
###################################

# movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
# rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
# df = movie.merge(rating, how="left", on="movieId")
# df.head()
#
# movie_ids = [130219, 356, 4422, 541]
# movies = ["The Dark Knight (2011)",
#           "Cries and Whispers (Viskningar och rop) (1972)",
#           "Forrest Gump (1994)",
#           "Blade Runner (1982)"]
#
# sample_df = df[df.movieId.isin(movie_ids)]
# sample_df.head()
#
# sample_df.shape
#
# user_movie_df = sample_df.pivot_table(index=["userId"],
#                                       columns=["title"],
#                                       values="rating")
#
# user_movie_df.shape
#
# reader = Reader(rating_scale=(1, 5))
#
# data = Dataset.load_from_df(sample_df[['userId',
#                                        'movieId',
#                                        'rating']], reader)



##############################
# Adım 2: Modelleme
##############################

# trainset, testset = train_test_split(data, test_size=.25)
# svd_model = SVD()
# svd_model.fit(trainset)
# predictions = svd_model.test(testset)
#
# accuracy.rmse(predictions)
#
#
# svd_model.predict(uid=1.0, iid=541, verbose=True)
#
# svd_model.predict(uid=1.0, iid=356, verbose=True)
#
#
# sample_df[sample_df["userId"] == 1]
# rating_value = sample_df[(sample_df["userId"] == 1) & (sample_df["movieId"] == 356)]["rating"].values
# print(f"{rating_value[0] if len(rating_value) > 0 else 'No rating found'}")


##############################
# Adım 3: Model Tuning
##############################

# param_grid = {'n_epochs': [5, 10, 20],
#               'lr_all': [0.002, 0.005, 0.007]}
#
#
# gs = GridSearchCV(SVD,
#                   param_grid,
#                   measures=['rmse', 'mae'],
#                   cv=3,
#                   n_jobs=-1,
#                   joblib_verbose=True)
#
# gs.fit(data)
#
# gs.best_score['rmse']
# gs.best_params['rmse']



###############################
# Adım 4: Final Model ve Tahmin
###############################

# dir(svd_model)
# svd_model.n_epochs
#
# svd_model = SVD(**gs.best_params['rmse'])
#
# data = data.build_full_trainset()
# svd_model.fit(data)
#
# svd_model.predict(uid=1.0, iid=541, verbose=True)




