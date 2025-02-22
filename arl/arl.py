#########################################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
#########################################################

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

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II


df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
# pip install openpyxl
# df_ = pd.read_excel("datasets/online_retail_II.xlsx",
#                     sheet_name="Year 2010-2011", engine="openpyxl")


df.describe().T
df.isnull().sum()
df.shape

df["Invoice"].head()


# aykiri degerleri silme
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)

# eşik değeri hesaplama / tahminsel değerler(Vahit Hoca yöntemi)
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# baskılama işlemi: alt limitin altında kalanı alt sınıra, üst sınırın üstünde kalanları ise üst sınıra baskıla
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# tüm fonk ilgili değişkenlere göre uyarlayıp toparlama (dataframe i yeniden çağır bozuk haliyle)
# üstte oluşturmuş olduğumuz aykırı değer silme fonk getirdik, ek olarakta thresholds ile eşik değerleri oluşturan ve
# replace eden diğer iki fonk u birbirine bağlayan replace_with_thresholds fonk yazdık içine de işlem yaptığımız
# Qantity ve Price değişkenlerini yazıp ilgili değişkenlerde işlemlerimizi gerçekleştirdik.
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


df = retail_data_prep(df)
df.isnull().sum()
df.describe().T

# InvoiceDate kafa karistiriyor, gecici sildim
df.drop("InvoiceDate", axis = 1).describe().T


#########################################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
#########################################################

df.head()

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1


# işlemlerin hızlı gerçekleşmesi ve herkes tarafından uygulanabilmesi için veri setini belirli bir ülkeye indirgeyerek ilerleyecez.
# Ülke olarak Fransa seçiyoruz. Fransa müşterilerinin birliktelik kurallarını türetmiş olacaz.

# Örneğin bu online şirket Avrupa' da örneğin Almanya pazarına giriş yapacak. Almanya pazarından gelen trafik/müşterilere ürün önermek istiyor.
# Ama orada önceden hiç müşterisi yoktu. Peki o zaman nasıl olacak? Almanya ile benzer alışkanlıklar sergilemesini beklediğim bir ülke
# belirlersem ve bende onun verileri olursa mesela Fransa gibi. Buradan birliktelik kuralları çıkarırım. Daha sonra Almanya pazarına girdiğimde
# Fransa' dan öğrendiğim birliktelik kurallarını Almanya' da uygularım.



# ülke seç
df_fr = df[df['Country'] == "France"]

# amaç Invoice product matrisi oluşturmak ama veri setinin yapısı çok uygun değil
# Invoice lara göre groupby a alalım, gözlemlemek adına Description alalım(normalde StockCode göre gidecez)
# Fatura ürün ve hangi üründen kaçar tane alınmış hesaplamak için, Quantity sum alalım
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

# pivot table ile de yapılır, unstack ile ürün isimlerini değişken isimlerine çeviriyoruz, iloc ile seçim base 5x5
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5] # unstack: pivot table misali

# eksik değer yerine 0 yaz, dolu olduğu gibi kalır
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5] # fillna: doldur

# ama problemi çözemedik sebebi boş değer 0 dolu yerlere 1 yazdırmak lazım kendi değeri yerine

# applymap ile ilgili problemi de çözdük.
# Invoice lara göre groupby aldık
# Description/StockCode gözlemledik (isteğe-ihtiyaca göre)
# Quantity nin sum aldık
# unstack() ile pivot table yaptık
# applymap sayesinde değerleri gezip (lambda kullan at fonk) ile boş değer yerine 0 dolu değer yerine 1 yazdık
#
# apply satır sütun bilgisi verilir bir fonk bu satır ya da sütun da döndürür.
# applymap ise bütün gözlemleri gezer
#
# .map güncel hali / .applymap kullanımı sonlandırılmış
df_fr.groupby(['Invoice', 'StockCode']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# yazılan fonk da dataframe ve id=False tanımlanmış ve koşul id ye göre ilerliyor yani id True ise işlem yapacağız
# işlemlerimiz ise az önceki fonk aynısını id=True iken StockCode id=False iken Description a göre işlem yapar
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

# atama yaparak kaydedip fonk çağır argüman girme, df_fr yani dataframe de fransa kısmında işlem yapıyoruz, belirtilmezse id False döner
fr_inv_pro_df = create_invoice_product_df(df_fr)

# id=True iken
fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)

fr_inv_pro_df.head().iloc[0:5, 0:5]

# !!!!!!****** DİKKAT ******!!!!!!
# id True False diye sürekli o fonk tanımlamamak adına bir yöntem
# check_id fonk dataframe ve stock_code argüman, produck_name değişkeni oluştur: dataframe StockCode değişkeninden al ve eşitle,
# Description değişkeninin string değerini al sadece, tolist liste oluştur, product_name yazdır, check_id çağrılınca id girince ver
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

# girilen StockCode ürünün Description verir
check_id(df_fr, 10120)



#########################################
# 3. Birliktelik Kurallarının Çıkarılması
#########################################

frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

# olası ürün çiftlerini gösterir, support değerine göre sıralar
# olası ürün çiftleri ve bunlara karşılık support değerleri
# yani bu ürünlerin tek başına görülme olasılığı veya olası kombinasyon yapıda ürünleri topluca bir item gibi düşünüp birlikte görülme olasılığı
frequent_itemsets.sort_values("support", ascending=False)

# association_rules fonk ile birliktelik kurallarını çıkarır
# metric: support, confidence, lift, leverage, conviction
# min_threshold: kural oluşturulurken minimum değer
# antecedents: X(ilk ürün)
# consequents: Y(ikinci ürün)
# antecedent support: X'in tek başına gözlenme olasılığı
# consequent support: Y'nin tek başına gözlenme olasılığı
# support: X ve Y'nin birlikte alınma olasılığı[set ler / support(X, Y)]
# confidence: X alındığında Y'nin alınma olasılığı
# lift: X alındığında Y'nin alınma olasılığı lift kadar artar(daha az sıklıkta olmasına rağmen birliktelikleri yakalar)
# leverage: X ve Y birlikte alınma olasılığı(kaldıraç etkisi. support u yüksek değerlere öncelik verme eğilimindedir)
# conviction: X alındığında Y'nin alınma olasılığı(şartlı olasılık. Y ürünü olmadan X ürünü beklenen frekansı)
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

# pratikte kullanılan birliktelik kuralları oranları/değerlendirmeleri
# support: 0.05 üzerinde, confidence: 0.1 üzerinde, lift: 5 üzerinde olanları getir
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]

check_id(df_fr, 21086)

# confidence değerine göre büyükten küçüğe sıralama
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

###################################
# 4. Çalışmanın Scriptini Hazırlama
###################################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

##############################################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
##############################################################

# !!!!!!****** DİKKAT ******!!!!!!

# Örnek:
# Kullanıcı örnek ürün id: 22492

product_id = 22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

# burada amaç ürün id si girilen ürünün birliktelik kurallarından yola çıkarak öneri yapmak
# ürün id si girilen ürünün antecedents leri arasında varsa consequents leri öneri listesine ekle
# öneri listesini döndür

# döngü içinde product_id bulunduğunda ilgili index i tutacak, product ürünleri kontrol ederken değerini tutacak, enumerate ile
# sorted_rules["antecedents"] içindeki ürünlerin indexleri ile/dahil bütün değişkenlerini dönecek,
# diğer döngü j product içindeki ürünleri dönecek, eğer j product_id ye eşitse kontrol yapacak,
# eğer eşitse recommendation_list e ekleme yapacak, ekleme yaparken de sorted_rules ın iloc sayesinde ilk döngüde tutulan
# ve product_id ile eşleşen ürünün antecedents index ideki consequents değerini ilk elemanını alacak ve boş listeye ekleme yapacak

# girilen ürün id si ni antecedents içinde ara(çoklu kombinasyonlar içinde olsa bile)
# bu esnada o ürünün index ini de hafızada tut
# id yi bulduğun antecedents e karşılık gelen consequents değerini recommendation_list e eklememiz lazım ama nasıl
# ilgili index i kullanarak sorted_rules ın
# ilgili index i kullanarak sorted_rules ın ilgili index indeki consequents değerini recommendation_list e ekleyebiliriz
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

check_id(df, 22326)

# tamamını fonk haline getirdik
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

# ihtiyaca göre öneri sayısını belirleme
arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)


