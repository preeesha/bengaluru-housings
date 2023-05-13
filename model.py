# Importing essential libraries
import numpy as np
import pandas as pd

from joblib import dump

from math import floor
from matplotlib import pyplot as plt
from matplotlib import rcParams as rcP

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# Loading the dataset
df = pd.read_csv("data/data.csv")

df.groupby("area_type")["area_type"].agg("count")
df.groupby("availability")["availability"].agg("count")
df.groupby("size")["size"].agg("count")
df = df.drop("society", axis="columns")

# Data Cleaning
df.isnull().sum()

# Applying median to the balcony and bath column
balcony_median = float(floor(df.balcony.median()))
bath_median = float(floor(df.bath.median()))
df.balcony = df.balcony.fillna(balcony_median)
df.bath = df.bath.fillna(bath_median)

df.isnull().sum()

# Dropping the rows with null values because the dataset is huge as compared to null values.
df = df.dropna()
df.isnull().sum()

# Converting the size column to bhk
df["bhk"] = df["size"].apply(lambda x: int(x.split(" ")[0]))
df = df.drop("size", axis="columns")
df.groupby("bhk")["bhk"].agg("count")

# Exploring the total_sqft column
df.total_sqft.unique()


# Since the total_sqft contains range values such as 1133-1384, lets filter out these values
def isFloat(x):
    try:
        float(x)
    except Exception:
        return False
    return True


# Displaying all the rows that are not integers
df[~df["total_sqft"].apply(isFloat)]


# Converting the range values to integer values and removing other types of error
def convert_sqft_to_num(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except Exception:
        return None


df["new_total_sqft"] = df.total_sqft.apply(convert_sqft_to_num)
df = df.drop("total_sqft", axis="columns")

# Removing the rows in new_total_sqft column that hase None values
df.isna().sum()

# Removing the rows in new_total_sqft column that hase None values
df = df.dropna()
df.isna().sum()


# Adding a new column of price_per_sqft
df1 = df.copy()

# In our dataset the price column is in Lakhs
df1["price_per_sqft"] = (df1["price"] * 100000) / df1["new_total_sqft"]
df1.head()

# Checking unique values of "location" column
locations = list(df["location"].unique())

# Removing the extra spaces at the end
df1.location = df1.location.apply(lambda x: x.strip())

# Calulating all the unqiue values in "location" column
location_stats = (
    df1.groupby("location")["location"].agg("count").sort_values(ascending=False)
)
location_stats

# Labelling the locations with less than or equal to 10 occurences to "other"
locations_less_than_10 = location_stats[location_stats <= 10]

df1.location = df1.location.apply(
    lambda x: "other" if x in locations_less_than_10 else x
)
len(df1.location.unique())

# Checking the unique values in "availability column"
df1.groupby("availability")["availability"].agg("count").sort_values(ascending=False)

# Labelling the dates into Not Ready
dates = (
    df1.groupby("availability")["availability"]
    .agg("count")
    .sort_values(ascending=False)
)

dates_not_ready = dates[dates < 10000]
df1.availability = df1.availability.apply(
    lambda x: "Not Ready" if x in dates_not_ready else x
)
# Checking the unique values in "area_type" column
df1.groupby("area_type")["area_type"].agg("count").sort_values(ascending=False)

# Removing the rows that have 1 Room for less than 300sqft
df2 = df1[~(df1.new_total_sqft / df1.bhk < 300)]


# Since there is a wide range for "price_per_sqft" column with min = Rs.267/sqft till max = Rs. 127470/sqft, we remove the extreme ends using the SD
def remove_pps_outliers(df):
    df_out = pd.DataFrame()

    for _, sub_df in df.groupby("location"):
        m = np.mean(sub_df.price_per_sqft)
        sd = np.std(sub_df.price_per_sqft)
        reduce_df = sub_df[
            (sub_df.price_per_sqft > (m - sd)) & (sub_df.price_per_sqft < (m + sd))
        ]
        df_out = pd.concat([df_out, reduce_df], ignore_index=True)

    return df_out


df3 = remove_pps_outliers(df2)


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    rcP["figure.figsize"] = (15, 10)
    plt.scatter(bhk2.new_total_sqft, bhk2.price, color="blue", label="2 BHK", s=50)
    plt.scatter(
        bhk3.new_total_sqft, bhk3.price, color="green", marker="+", label="3 BHK", s=50
    )
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (in Lakhs)")
    plt.title(location)
    plt.legend()


plot_scatter_chart(df3, "Hebbal")


# Here we observe that 3 BHK cost that same as 2 BHK in "Hebbal" location hence removing such outliers is necessary
def remove_bhk_outliers(df):
    exclude_indices = np.array([])

    for location, location_df in df.groupby("location"):
        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0],
            }

        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats["count"] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < (stats["mean"])].index.values,
                )

    return df.drop(exclude_indices, axis="index")


df4 = remove_bhk_outliers(df3)

plot_scatter_chart(df4, "Hebbal")

plt.hist(df4.price_per_sqft, rwidth=0.5)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")

plt.hist(df4.bath, rwidth=0.5)
plt.xlabel("Number of Bathrooms")
plt.ylabel("Count")

# Removing the rows that have "bath" greater than "bhk"+2
df5 = df4[df4.bath < (df4.bhk + 2)]


# Model Building

# Removing the unnecessary columns (columns that were added only for removing the outliers)
df6 = df5.copy()
df6 = df6.drop("price_per_sqft", axis="columns")

df6.head()

# Converting the categorical_value into numerical_values using get_dummies method
dummy_cols = pd.get_dummies(df6.location).drop("other", axis="columns")
df6 = pd.concat([df6, dummy_cols], axis="columns")

# Converting the categorical_value into numerical_values using get_dummies method
dummy_cols = pd.get_dummies(df6.availability).drop("Not Ready", axis="columns")
df6 = pd.concat([df6, dummy_cols], axis="columns")

# Converting the categorical_value into numerical_values using get_dummies method
dummy_cols = pd.get_dummies(df6.area_type).drop("Super built-up  Area", axis="columns")
df6 = pd.concat([df6, dummy_cols], axis="columns")

df6.drop(["area_type", "availability", "location"], axis="columns", inplace=True)
df6.head()

# Size of the dataset
df6.shape

# Splitting the dataset into features and label
X = df6.drop("price", axis="columns")
y = df6["price"]

# Creating a function for GridSearchCV


def find_best_model(X, y):
    models = {
        "linear_regression": {
            "model": LinearRegression(),
            "parameters": {"positive": [True, False]},
        },
        "lasso": {
            "model": Lasso(),
            "parameters": {"alpha": [1, 2], "selection": ["random", "cyclic"]},
        },
        "decision_tree": {
            "model": DecisionTreeRegressor(),
            "parameters": {
                "criterion": ["mse", "friedman_mse"],
                "splitter": ["best", "random"],
            },
        },
    }

    scores = []
    cv_X_y = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)

    for model_name, model_params in models.items():
        gs = GridSearchCV(
            model_params["model"],
            model_params["parameters"],
            cv=cv_X_y,
            return_train_score=False,
        )
        gs.fit(X, y)
        scores.append(
            {
                "model": model_name,
                "best_parameters": gs.best_params_,
                "accuracy": gs.best_score_,
            }
        )

    return pd.DataFrame(scores, columns=["model", "best_parameters", "accuracy"])


find_best_model(X, y)


def getUniqueValues():
    return {
        "location": list(df5["location"].unique()),
        "area_type": list(df5["area_type"].unique()),
        "availability": list(df5["availability"].unique()),
    }


def predict(model, location, bhk, bath, balcony, sqft, area_type, availability):
    loc_index, area_index, avail_index = -1, -1, -1

    if location != "other":
        loc_index = int(np.where(X.columns == location)[0][0])

    if area_type != "Super built-up  Area":
        area_index = np.where(X.columns == area_type)[0][0]

    if availability != "Not Ready":
        avail_index = np.where(X.columns == availability)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = bath
    x[1] = balcony
    x[2] = bhk
    x[3] = sqft

    if loc_index >= 0:
        x[loc_index] = 1
    if area_index >= 0:
        x[area_index] = 1
    if avail_index >= 0:
        x[avail_index] = 1

    return round(model.predict([x])[0] * 1e5)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=20
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

    dump(model, "model.joblib")
