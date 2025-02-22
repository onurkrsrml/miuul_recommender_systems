###########################################################
# Model-Based Collaborative Filtering: Matrix Factorization
###########################################################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# from item_based_recommender.item_based_recommender import user_movie_df

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

###################################
# Adım 1: Veri Setinin Hazırlanması
###################################


# Bir fonksiyonun bir noktadaki türevi o fonksiyonun o noktadaki maksimum artış yönünü verir. Artış yönü/türevi/gradient in
# negatif yönüne negatif olarak gittiğimizde ilgili fonksiyonu minimum laştıracak şekilde parametre değerlerini güncelliyor oluruz.
# Gradient Descent, yakınsayana/yaklaşana kadar tekrar et, bir fonk daki parametrenin değerlerini iteratif olarak değiştir.
# Nasıl? Belirli bir hız ile değiştir. İlgili parametrenin türevini alırsan türevi aldığın noktadaki değerine göre belirli
# bir hızla bu parametrenin değerini önceki değeri ile değiştirerek güncelle. Bu işlemi yaparken de bir learning rate
# dediğimiz bir hız belirliyoruz. Bu hız ne kadar büyük olursa o kadar hızlı öğreniriz ama aşırı büyük olursa da
# aşırı hızlı öğreniriz ve asla yakınsayamayız. Bu hız ne kadar küçük olursa o kadar yavaş öğreniriz ama yakınsama
# garantimiz olur. Bu hızı belirlerken de bir çok deneme yaparak en uygun hızı bulmamız gerekmektedir.

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

# movie_ids listesindeki movieId lere ait verileri filtrele.
sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

# surprise kutuphanesine uygun formata cevirmek icin pivot_table olusturuyoruz.
user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

# surprise kutuphanesinden Reader sinifini kullanarak rating scale belirliyoruz.
reader = Reader(rating_scale=(1, 5))

# burada surprise kutuphanesinden Dataset sinifini kullanarak surprise icin uygun formata ceviriyoruz.
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

##############################
# Adım 2: Modelleme
##############################

# scikit-learn degil surprise kutuphanesinden train_test_split
# surprise kutuphanesinden train_test_split -> veriyi train ve test olarak ayir. verinin %25 ini test et %75 ini train et.
trainset, testset = train_test_split(data, test_size=.25) # test_size=.25 -> %25 test %75 train
# trainset -> modeli egitmek icin kullanilacak veri seti
# testset -> modelin performansini olcmek icin kullanilacak veri seti

# SVD -> Singular Value Decomposition -> Tekil Değer Ayrışımı -> matris ayrışımı yaparak matrisin içerisindeki latent faktörleri
# bulmamızı sağlar. Bu latent faktörler sayesinde de kullanıcıların ve ürünlerin özelliklerini bulabiliriz.
svd_model = SVD()

# modeli trainset ile fit et. modeli egit.
svd_model.fit(trainset)

# modeli testset ile test et. modelin performansini olc. tahminleri al. gercek degerlerle karsilastir.
predictions = svd_model.test(testset)

# accuracy -> tahminlerin dogrulugunu olc. gercek degerlerle tahmin edilen degerlerin karsilastirilmasi.
# hata kareler ortalamasini al. karekokunu al. rmse yi hesapla. rmse yi ekrana yazdir.
accuracy.rmse(predictions) # root mean square error, hata kareler ortalamasının karekökü

# girilen userId ve movieId ye gore tahmin yap. verbose=True -> tahmin yaparken detaylari ekrana yazdir.
svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)

# girilen userId ye gore gercek rating degerini al.
sample_df[sample_df["userId"] == 1]

# girilen userId ve movieId ye gore gercek rating degerini al.
rating_value = sample_df[(sample_df["userId"] == 1) & (sample_df["movieId"] == 356)]["rating"].values
print(f"{rating_value[0] if len(rating_value) > 0 else 'No rating found'}")




##############################
# Adım 3: Model Tuning
##############################

# param_grid -> model tuning icin parametrelerin belirlenmesi.
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

# GridSearchCV -> Grid Search Cross Validation
# capraz dogrulama yaparak en iyi parametreleri bulmamizi saglar.
# cv -> 3 parcaya bol, 2 parcasi ile model kur 1 parcasi ile test et. daha sonra diger 2(disarida kalan dahil icten biri haric)
# parca ile model kur digeri ile test et. en sonda diger haric 1 parca ile model kurulan modeli test et.(kombinasyon yap)
# daha sonra bu test islemlerinin ortalamasini al. rmse ve mae yi baz alarak hesapla.
# n_jobs=-1 -> islemcileri full performans olarak kullan / paralel olarak calistir.
# joblib_verbose=True -> islemi yaparken rapor / detaylari ekrana yazdir.
gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

# mae -> mean absolute error -> hata mutlak değerlerinin ortalaması. gercek degerlerle tahmin edilen degerlerin farklarinin karelerinin ortalamasini alir.
# rmse -> root mean square error -> hata kareler ortalamasının karekökü. gercek degerlerle tahmin edilen degerlerin farklarinin ortalamanin karekokunu alir.

# modeli fit et. model tuning yap.
gs.fit(data)

# best_score -> en iyi skor cagir
gs.best_score['rmse']

# best_params -> en iyi parametreleri cagir
gs.best_params['rmse']


###############################
# Adım 4: Final Model ve Tahmin
###############################

# SVD

# dir -> object in icindeki method ve attribute leri listeler.
# dir(svd_model)

# n_epochs -> modelin kac epoch ile egitildigini gosterir.
# svd_model.n_epochs

# best_params ile en iyi parametreleri alarak modeli tekrar olustur.
# ** -> dictionary unpacking. best_params dictionary sinin icindeki key ve value leri parametre olarak al.
# ** -> keyworded arguments
svd_model = SVD(**gs.best_params['rmse'])

# full trainset -> trainsetin tamami
# modeli full trainset ile fit et. cunku en iyi parametrelerle modeli olusturduk. simdi buyuk veri ile daha iyi sonuclar elde edebiliriz.
data = data.build_full_trainset()
svd_model.fit(data) # modeli kur

# ilgili userId ve movieId ye gore tahmin yap.
svd_model.predict(uid=1.0, iid=541, verbose=True)





