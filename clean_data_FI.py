import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d


# Load the data
original_data_path = (
    r"./Data/immoweb_data_cleaned.csv"
)
df = pd.read_csv(original_data_path)
salary_province = {
    "East Flanders": 22073,
    "Antwerp": 21270,
    "Brussels Capital": 16068,
    "Liège": 18.906,
    "Flemish Brabant": 23527,
    "Hainaut": 17713,
    "Walloon Brabant": 23442,
    "Luxembourg": 20004,
    "Limburg": 20633,
    "Namur": 19867,
    "Other": 21325,
}

df["Salary_prov_med"] = df["Province"].map(salary_province)

# print(df.head())
prix_m2_app = {
    "East Flanders": 2864,
    "Antwerp": 2789,
    "Brussels Capital": 3401,
    "Liège": 2214,
    "Flemish Brabant": 3197,
    "Hainaut": 1854,
    "Walloon Brabant": 3156,
    "Luxembourg": 2395,
    "Limburg": 2488,
    "Namur": 2543,
    "Other": 3759,
}
prix_m2_house = {
    "East Flanders": 2229,
    "Antwerp": 2365,
    "Brussels Capital": 3245,
    "Liège": 1684,
    "Flemish Brabant": 2484,
    "Hainaut": 1382,
    "Walloon Brabant": 2302,
    "Luxembourg": 1574,
    "Limburg": 1898,
    "Namur": 1625,
    "Other": 2017,
}

df["Prix_m2_prov"] = df.apply(
    lambda x: (
        prix_m2_house[x["Province"]]
        if x["Type_of_Property"] == 1
        else prix_m2_app[x["Province"]]
    ),
    axis=1,
)

densita_prov = {
    "East Flanders": 2229,
    "Antwerp": 2365,
    "Brussels Capital": 1241175,
    "Liège": 1684,
    "Flemish Brabant": 1187483,
    "Hainaut": 1356895,
    "Walloon Brabant": 412934,
    "Luxembourg": 1574,
    "Limburg": 1898,
    "Namur": 1625,
    "Other": 2017,
}

df.drop(columns=["Garden", "Terrace", "Fully_Equipped_Kitchen", "Lift"], inplace=True)
label_encoder = LabelEncoder()
columns_to_encode = [
    "Type_of_Property",
    "Subtype_of_Property",
    "State_of_the_Building",
    "Municipality",
    "Province",
]
for col in columns_to_encode:
    df[f"{col}"] = label_encoder.fit_transform(df[col])


df.to_csv("Data_from_Edo")
