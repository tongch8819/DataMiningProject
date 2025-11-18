import pandas as pd
# from ucimlrepo import fetch_ucirepo # Removed API import
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# PART 1: Data Loading (Local CSV)
# ==========================================
print("--- Step 1: Loading Data from Local CSV ---")

# Load the dataset from a local file
# Note: This specific UCI dataset often uses semi-colons (;) as delimiters 
# and 'cp1252' encoding. If your file is a standard comma-separated CSV, 
# change sep=';' to sep=',' and remove the encoding parameter.
try:
    df = pd.read_csv('apartments_for_rent_classified_10K.csv', sep=';', encoding='cp1252')
    print("Loaded successfully with semi-colon delimiter.")
except Exception as e:
    print(f"Initial load failed ({e}), trying standard CSV format...")
    df = pd.read_csv('apartments_for_rent_classified_10K.csv')

print(f"Dataset loaded. Total records: {len(df)}")
print(f"Columns found: {list(df.columns)}\n")


# ==========================================
# PART 2: Data Preparation & Discretization
# ==========================================
print("--- Step 2: Preparing & Discretizing Data ---")

# We select only the columns that are useful for finding patterns.
# 'id', 'title', and 'body' are too unique and create noise.
cols_to_use = ['price', 'square_feet', 'bedrooms', 'bathrooms', 'state', 'amenities']

# Check if columns exist before proceeding
missing_cols = [c for c in cols_to_use if c not in df.columns]
if missing_cols:
    raise ValueError(f"The following required columns are missing in the CSV: {missing_cols}")

# Create a working copy
df_assoc = df[cols_to_use].copy()

# Handle Missing Values
# For 'amenities', if it's missing, we treat it as a specific category 'NoAmenities'
df_assoc['amenities'] = df_assoc['amenities'].fillna('NoAmenities')

# Drop rows where critical numerical data might be missing
# We coerce errors to NaN just in case there are string artifacts like "1,000" instead of 1000
df_assoc['price'] = pd.to_numeric(df_assoc['price'], errors='coerce')
df_assoc['square_feet'] = pd.to_numeric(df_assoc['square_feet'], errors='coerce')
df_assoc.dropna(subset=['price', 'square_feet'], inplace=True)

# --- DISCRETIZATION (Converting numbers to categories) ---

# 1. Discretize PRICE into 3 bins: Low, Medium, High
# We use qcut (Quantile Cut) to ensure roughly equal number of apartments in each bin.
try:
    df_assoc['Price_Bin'] = pd.qcut(df_assoc['price'], q=3, labels=['Rent_Low', 'Rent_Medium', 'Rent_High'])
except ValueError:
    # Fallback if data is too small or skew prevents 3 unique bins
    df_assoc['Price_Bin'] = pd.cut(df_assoc['price'], bins=3, labels=['Rent_Low', 'Rent_Medium', 'Rent_High'])

# 2. Discretize SQUARE_FEET into 3 bins: Small, Medium, Large
try:
    df_assoc['Size_Bin'] = pd.qcut(df_assoc['square_feet'], q=3, labels=['Size_Small', 'Size_Medium', 'Size_Large'])
except ValueError:
    df_assoc['Size_Bin'] = pd.cut(df_assoc['square_feet'], bins=3, labels=['Size_Small', 'Size_Medium', 'Size_Large'])

# 3. Convert BEDROOMS and BATHROOMS to string labels
# We add prefixes so we know what the number refers to in the final rules (e.g., "2_Beds" vs "2_Baths")
df_assoc['Bed_Label'] = df_assoc['bedrooms'].astype(str) + '_Beds'
df_assoc['Bath_Label'] = df_assoc['bathrooms'].astype(str) + '_Baths'

print("Data discretization complete.")
print(df_assoc[['price', 'Price_Bin', 'square_feet', 'Size_Bin']].head())
print("\n")


# ==========================================
# PART 3: Building Transactions
# ==========================================
print("--- Step 3: Converting to Transactions ---")

transactions = []

for index, row in df_assoc.iterrows():
    # Start the "basket" with our structural attributes
    basket = [
        str(row['Price_Bin']),
        str(row['Size_Bin']),
        str(row['Bed_Label']),
        str(row['Bath_Label']),
        str(row['state'])
    ]
    
    # Process Amenities
    # The dataset has amenities as a string: "Pool,Gym,Internet"
    # We need to split this string into individual items.
    if row['amenities'] != 'NoAmenities':
        # Split by comma and strip whitespace
        amenity_list = [item.strip() for item in str(row['amenities']).split(',')]
        # Add valid amenities to the basket
        basket.extend([a for a in amenity_list if a])
        
    transactions.append(basket)

print(f"Created {len(transactions)} transactions.")
print(f"Sample Transaction 0: {transactions[0]}")
print("\n")


# ==========================================
# PART 4: Association Analysis (Apriori)
# ==========================================
print("--- Step 4: Running Apriori Algorithm ---")

# 1. One-Hot Encoding
# This converts the list of lists into a True/False matrix
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Total distinct items found: {len(te.columns_)}")

# 2. Find Frequent Itemsets
# min_support=0.05: The itemset must appear in at least 5% of the data.
# If you get no results, try lowering this to 0.01.
frequent_itemsets = apriori(df_onehot, min_support=0.05, use_colnames=True)

print(f"Found {len(frequent_itemsets)} frequent itemsets.")

# 3. Generate Rules
# metric="lift": We are interested in rules where the items are correlated (Lift > 1)
# min_threshold=1.2: We only want rules that are at least 20% stronger than random chance.
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# Sort rules by Confidence (reliability) and Lift (strength)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])


# ==========================================
# PART 5: Interpreting Results
# ==========================================
print("\n--- Step 5: Top 10 Association Rules ---")

# Helper function to display rules cleanly
cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
print(rules[cols].head(10))

# Filter for "Price" specific rules (Interpreting factors for Rent)
print("\n--- Specific Insight: What leads to 'Rent_High'? ---")
high_rent_rules = rules[rules['consequents'].apply(lambda x: 'Rent_High' in x)]

if not high_rent_rules.empty:
    print(high_rent_rules[cols].head(5))
else:
    print("No strong rules found pointing directly to Rent_High with current thresholds.")

print("\n--- Specific Insight: What leads to 'Rent_Low'? ---")
low_rent_rules = rules[rules['consequents'].apply(lambda x: 'Rent_Low' in x)]

if not low_rent_rules.empty:
    print(low_rent_rules[cols].head(5))
else:
    print("No strong rules found pointing directly to Rent_Low with current thresholds.")