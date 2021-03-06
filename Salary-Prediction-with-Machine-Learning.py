import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" %x)
pd.set_option("display.width", 500)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################")
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                       "TARGET_COUNT": dataframe.groupby(categorical_col)[target].count()}), end="\n\n")
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal de??i??kenlerin isimlerini verir.
    Not: Kategorik de??i??kenlerin i??erisine numerik g??r??n??ml?? kategorik de??i??kenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                De??i??ken isimleri al??nmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan de??i??kenler i??in s??n??f e??ik de??eri
        car_th: int, optinal
                kategorik fakat kardinal de??i??kenler i??in s??n??f e??ik de??eri

    Returns
    ------
        cat_cols: list
                Kategorik de??i??ken listesi
        num_cols: list
                Numerik de??i??ken listesi
        cat_but_car: list
                Kategorik g??r??n??ml?? kardinal de??i??ken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam de??i??ken say??s??
        num_but_cat cat_cols'un i??erisinde.
        Return olan 3 liste toplam?? toplam de??i??ken say??s??na e??ittir: cat_cols + num_cols + cat_but_car = de??i??ken say??s??

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # Nmerik g??r??n??l?? kategorikleri ????kartt??k
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
def check_df(dataframe, head=10):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    # Grafikler birbirini ezmesin diye
    plt.show(block=True)
######################################################
# Exploratory Data Analysis (EDA) KE????FC?? VER?? ANAL??Z??
######################################################

df = pd.read_csv("Dataset/hitters.csv")

# Veri setinin incelenmesi

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)

num_summary(df, num_cols)

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

for col in df.columns:
    plot_numerical_col(df, col)



######################################################
# Data Preprocessing (Veri ??n ????leme)
######################################################

df["Salary"].describe([.25, .5, .75, .90, .95, .99]).T

###################################
# Ayk??r?? de??er analizi
####################################

cols = [col for col in num_cols if "Salary" not in col]

for i in cols:
    print(i, " : ", check_outlier(df, i))

for col in cols:
    replace_with_thresholds(df, col)

###################################
# Eksik de??er analizi
####################################

df.isnull().sum()
df.shape
missing_values_table(df)

# Silme

#df.dropna(inplace=True)
#df.shape

# Median ile  doldurma
df["Salary"] = df["Salary"].fillna(df["Salary"].median())


########################################################
# ??zellik M??hendisli??i
########################################################

#####################################################
# Yeni De??i??kenler Olu??turma
######################################################

# Korelasyon, iki veya daha fazla de??i??ken aras??nda bir ili??ki olup olmad??????n??, e??er
# ili??ki varsa bu ili??kinin miktar??n?? ve y??n??n?? say??sal olarak belirlememizi sa??layan
# istatistiksel bir tekniktir. Bu tekni??i kullanarak yeni de??i??ken olu??tururken daha
# fikir edinebiliriz.

df.corr()
df.corrwith(df["Salary"]).sort_values(ascending=False)

# Salary ile en y??ksek korelasyona sahip 4 de??i??ken
# CHits --> Oyuncunun kariyeri boyunca yapt?????? isabetli vuru?? say??s??
# CAtBat --> Oyuncunun kariyeri boyunca topa vurma say??s??
# CRBI --> Oyuncunun kariyeri boyunca ko??u yapt??rd??rd?????? oyuncu say??s??
# CRuns --> Oyuncunun kariyeri boyunca tak??m??na kazand??rd?????? say??

# Oyuncunun kariyeri boyunca yapt?????? isabetli vuru?? say??s?? / Oyuncunun kariyeri boyunca topa vurma say??s??
df.head()
df["Hit_rate"] = df["CHits"] / df["CAtBat"]
# Tak??ma kazand??rd?????? say?? + Oyun icinde tak??m arkada????yla yard??mla??ma(Asist gibi d??????n??lebilir)

df["Total_Run"] = df["PutOuts"] + df["CRuns"]

# 1986-1987 y??l??ndaki istatislikleri
df["1986-1987_Total_Run"] = df["Assists"] + df["Runs"]

# Total_Run + Oyuncun kariyeri boyunca kar???? oyuncuya yapt??rd?????? hata say??s??
df["Total_Run_2"] = df["PutOuts"] + df["CRuns"] + df["CWalks"]

# Total_Run - 1986-1987 sezonundaki oyuncunun hata say??s??
df["Total_Run_3"] = df["PutOuts"] + df["CRuns"] - df["Errors"]

# 1986 - 1987 sezonlar?? aras?? istatislikleri
df["1986-1987"] = df["AtBat"] + df["Hits"] + df["HmRun"] + df["Runs"] + df["RBI"] + df["Assists"] - df["Errors"]
#

df.groupby("League")["Salary"].mean()
df.groupby("NewLeague")["Salary"].mean()

df.loc[(df["League"] == "A") & (df["NewLeague"] == "A"), ["T_league"]] = "AA"
df.loc[(df["League"] == "A") & (df["NewLeague"] == "N"), ["T_league"]] = "AN"
df.loc[(df["League"] == "N") & (df["NewLeague"] == "A"), ["T_league"]] = "NA"
df.loc[(df["League"] == "N") & (df["NewLeague"] == "N"), ["T_league"]] = "NN"

df["Years"].describe().T
df["Deneyim"] = pd.cut(x=df["Years"], bins=[0, 6, 11, 24], labels=["??aylak", "Orta_Tecr??be", "Tecr??beli"])
df.groupby("Deneyim")["Salary"].mean()

df.groupby("Division")["Salary"].mean()

df.head()

#############################################
# Encoding (Label Encoding, One-Hot Encoding)
#############################################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)
df.head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

#############################################
# Feature Scaling (??zellik ??l??eklendirme)
#############################################
df.columns

scale_col = [col for col in df.columns if col not in ["League","Salary","Division", "NewLeague", "Deneyim_Orta_Tecr??be", "Deneyim_Tecr??beli"
                                                      ,"T_league_AN", "T_league_NA","T_league_NN"]]

for col in scale_col:
    df[col] = RobustScaler().fit_transform(df[[col]])
df.head()

######################################################
# Maa?? Tahmin Modeli ve Tahmini
######################################################

df.head()
y = df["Salary"]
X = df.drop(["Salary"], axis=1)
X.info()
reg_model = LinearRegression().fit(X, y)

reg_model.intercept_
reg_model.coef_

####################################
# Ba??ar?? de??erlendirmesi
####################################
y_pred = reg_model.predict(X)

#MSE(Ortalama hatay?? verir)
mean_squared_error(y, y_pred)
# MSE = 82201.54810503636
y.mean() # 515.6009534161489
y.std() # 409.81745894162634
y.describe()
# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# RMSE = 286.70812354210744

#MAE
mean_absolute_error(y, y_pred)
# 209.63193961604355

# R-KARE
reg_model.score(X, y)
# 0.5090356890335104

###################################################################
# 2
###################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)
y_pred = reg_model.predict(X_train)
# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_

#MSE(Ortalama hatay?? verir)
mean_squared_error(y_train, y_pred)
# MSE = 77058.48498518117

#RMSE
np.sqrt(mean_squared_error(y_train, y_pred))
# 277.5941011354189

#MAE
mean_absolute_error(y_train, y_pred)
# 199.89072704086044

# R-KARE
reg_model.score(X_train, y_train)
# 0.47714112510663076

#############################################################
# 5 kat CV RMSE
#############################################################

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 320.85157505544686









