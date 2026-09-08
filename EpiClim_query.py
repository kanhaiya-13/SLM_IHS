import pandas as pd

df = pd.read_csv("Final_data.csv")

# 0. Basic shape and column sanity check
print("=== SHAPE & COLUMNS ===")
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.head(3))

# 1. Date range and week coverage
print("\n=== DATE RANGE ===")
print("Year range:", df['year'].min(), "-", df['year'].max())
print("Unique years:", sorted(df['year'].unique()))
print("Unique weeks per year (sample):", df.groupby('year')['week'].nunique())

# 2. Disease breakdown — row count and share per disease
print("\n=== DISEASE BREAKDOWN ===")
disease_col = [c for c in df.columns if 'disease' in c.lower()][0]
print(df[disease_col].value_counts())
print("\nAs % of total:")
print((df[disease_col].value_counts(normalize=True) * 100).round(2))

# 3. District coverage
print("\n=== DISTRICT COVERAGE ===")
district_col = [c for c in df.columns if 'district' in c.lower()][0]
print("Unique districts:", df[district_col].nunique())

# rows per district
rows_per_district = df.groupby(district_col).size()
print("Rows per district — min/median/max:", 
      rows_per_district.min(), rows_per_district.median(), rows_per_district.max())
print("\nBottom 10 districts by row count:")
print(rows_per_district.sort_values().head(10))
print("\nTop 10 districts by row count:")
print(rows_per_district.sort_values(ascending=False).head(10))

# 4. Per-district-per-disease density (the real test for windowing feasibility)
print("\n=== DISTRICT x DISEASE COMBINATIONS ===")
combo = df.groupby([district_col, disease_col]).size()
print("Number of (district, disease) combos:", len(combo))
print("Rows per combo — min/median/max:", combo.min(), combo.median(), combo.max())
print("\nHow many combos have >= 12 rows (min needed for a single 12-week input window)?")
print((combo >= 12).sum(), "out of", len(combo))
print("\nHow many combos have >= 52 rows (roughly a year of weekly data)?")
print((combo >= 52).sum(), "out of", len(combo))

# 5. Is this an event log or a full grid? Check for zero-case rows
print("\n=== ZERO-CASE ROW CHECK (event log vs. full grid) ===")
cases_col = [c for c in df.columns if 'case' in c.lower()][0]
print("Rows with 0 cases:", (df[cases_col] == 0).sum())
print("Rows with cases > 0:", (df[cases_col] > 0).sum())
print("Min/max cases:", df[cases_col].min(), df[cases_col].max())

# 6. Check for actual weekly continuity within a single high-volume district+disease
print("\n=== CONTINUITY CHECK (top district+disease pair) ===")
top_combo = combo.sort_values(ascending=False).index[0]
top_district, top_disease = top_combo
sub = df[(df[district_col] == top_district) & (df[disease_col] == top_disease)]
sub_sorted = sub.sort_values(['year', 'week'])
print(f"Top combo: {top_district} / {top_disease}, {len(sub_sorted)} rows")
print(sub_sorted[['year', 'week', cases_col]].to_string())

# 7. Missing values check
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# 8. State-level rollup (in case district-level is too sparse and you need to fall back to state-week granularity)
print("\n=== STATE-LEVEL DENSITY (fallback check) ===")
state_col = [c for c in df.columns if 'state' in c.lower()][0]
state_combo = df.groupby([state_col, disease_col]).size()
print("Number of (state, disease) combos:", len(state_combo))
print("Rows per combo — min/median/max:", state_combo.min(), state_combo.median(), state_combo.max())
print((state_combo >= 52).sum(), "out of", len(state_combo), "have >= 52 rows")