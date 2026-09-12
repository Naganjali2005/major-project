import pandas as pd

file_path = "data/pcos_updated.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Sheet1"
)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for i, column in enumerate(df.columns):
    print(f"{i}: {column}")

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print("Number of duplicate rows:", df.duplicated().sum())

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df["PCOS (Y/N)"].value_counts())

print("\n" + "=" * 60)
print("TARGET PERCENTAGE")
print("=" * 60)

print(df["PCOS (Y/N)"].value_counts(normalize=True) * 100)

print("\nImportant column values:")

columns_to_check = [
    "II    beta-HCG(mIU/mL)",
    "AMH(ng/mL)",
    "Fast food (Y/N)",
    "Cycle(R/I)",
    "Pregnant(Y/N)"
]

for column in columns_to_check:
    print("\n", column)
    print(df[column].value_counts(dropna=False))

print("\nRows where Cycle(R/I) is not 2 or 4:")

print(df[~df["Cycle(R/I)"].isin([2, 4])][
    ["Patient File No.", "PCOS (Y/N)", "Age (yrs)", "Cycle(R/I)"]
])

print("\nComplete row for Patient File No. 513:")

print(
    df[df["Patient File No."] == 513].T
)


print("\nUnique values of encoded columns:")

columns_to_check = [
    "Blood Group",
    "Cycle(R/I)",
    "Pregnant(Y/N)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)"
]

for column in columns_to_check:
    print(f"\n{column}")
    print(df[column].value_counts(dropna=False))
    
