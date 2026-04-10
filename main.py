import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset

avtomobili1 = pd.read_csv(
    "data/Nio_vozila_1.csv",
    sep=";",
    encoding="latin-1",
    quotechar='"',
    on_bad_lines="skip"
)

avtomobili2 = pd.read_csv(
    "data/Nio_vozila_2.csv",
    sep=";",
    encoding="latin-1",
    quotechar='"',
    on_bad_lines="skip"
)

avtomobili3 = pd.read_csv(
    "data/Nio_vozila_3.csv",
    sep=";",
    encoding="latin-1",
    quotechar='"',
    on_bad_lines="skip"
)

avtomobili = pd.concat([avtomobili1, avtomobili2, avtomobili3], ignore_index=True)



print(avtomobili.head(10))
print(avtomobili.describe())
print(len(avtomobili))