# Load the required libraries
library(tidyverse) # For data manipulation (equivalent to Pandas)
library(arules)    # For the Apriori algorithm and transaction handling
library(dplyr)     # Explicitly use dplyr functions

# ==========================================
# PART 1: Data Loading (Local CSV)
# ==========================================
cat("--- Step 1: Loading Data from Local CSV ---\n")

# Load the dataset from a local file.
# R's readr::read_csv2 is often good for semi-colon delimited files.
# It uses the encoding 'latin1', which is equivalent to 'cp1252'.
file_path <- 'apartments_for_rent_classified_10K.csv'

# Attempt 1: Semi-colon delimited (like the Python script)
df <- tryCatch({
    read_csv2(file_path, show_col_types = FALSE)
}, error = function(e) {
    cat(paste("Initial load failed (", e$message, "), trying standard CSV format...\n"))
    # Attempt 2: Standard comma-separated CSV
    read_csv(file_path, show_col_types = FALSE)
})

cat(paste("Dataset loaded. Total records:", nrow(df), "\n"))
cat("Columns found:", paste(colnames(df), collapse = ", "), "\n\n")

# Convert the data frame to a 'tibble' (tidyverse's data frame)
df_assoc <- as_tibble(df)

# ==========================================
# PART 2: Data Preparation & Discretization
# ==========================================
cat("--- Step 2: Preparing & Discretizing Data ---\n")

# Define columns to use and check for existence (R's equivalent of Python's check)
cols_to_use <- c('price', 'square_feet', 'bedrooms', 'bathrooms', 'state', 'amenities')
missing_cols <- setdiff(cols_to_use, colnames(df_assoc))
if (length(missing_cols) > 0) {
    stop(paste("The following required columns are missing in the CSV:", paste(missing_cols, collapse = ", ")))
}

# Select and process data using the 'pipe' operator (%>%)
df_assoc <- df_assoc %>%
    # Select columns
    select(all_of(cols_to_use)) %>%

    # Handle Missing Values (fillna)
    mutate(amenities = replace_na(amenities, 'NoAmenities')) %>%

    # Coerce to numeric (equivalent to pd.to_numeric(errors='coerce'))
    # This also converts columns from character/string to numeric where possible
    mutate(
        price = as.numeric(price),
        square_feet = as.numeric(square_feet)
    ) %>%

    # Drop rows where critical numerical data is missing (dropna)
    drop_na(price, square_feet)

# --- DISCRETIZATION (Converting numbers to categories) ---

# 1. Discretize PRICE into 3 bins: Low, Medium, High (equivalent to pd.qcut)
# The 'cut' function with 'breaks = quantile(..., na.rm=TRUE)' is R's most direct equivalent to pd.qcut
df_assoc <- df_assoc %>%
    mutate(
        # We use quantile for equal frequency bins (qcut)
        Price_Bin = cut(
            price,
            breaks = quantile(price, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE, type = 3),
            labels = c('Rent_Low', 'Rent_Medium', 'Rent_High'),
            include.lowest = TRUE, # Ensures min value is included
            right = FALSE,         # Ensures intervals are [x, y)
            dig.lab = 5            # Controls label formatting precision
        ),
        # 2. Discretize SQUARE_FEET into 3 bins (qcut equivalent)
        Size_Bin = cut(
            square_feet,
            breaks = quantile(square_feet, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE, type = 3),
            labels = c('Size_Small', 'Size_Medium', 'Size_Large'),
            include.lowest = TRUE,
            right = FALSE,
            dig.lab = 5
        ),
        # 3. Convert BEDROOMS and BATHROOMS to string labels
        Bed_Label = paste0(as.character(bedrooms), "_Beds"),
        Bath_Label = paste0(as.character(bathrooms), "_Baths")
    ) %>%
    # Remove original numeric columns if no longer needed
    select(-price, -square_feet, -bedrooms, -bathrooms)


cat("Data discretization complete.\n")
print(head(df_assoc %>% select(Price_Bin, Size_Bin, Bed_Label, Bath_Label), 5))
cat("\n")

# ==========================================
# PART 3: Building Transactions
# ==========================================
cat("--- Step 3: Converting to Transactions ---\n")

# R's arules package expects data in a specific format for transaction processing.
# We create a single column of lists/vectors where each row is a transaction.

# 1. Combine the structural bins/labels into one vector per row
df_items <- df_assoc %>%
    select(Price_Bin, Size_Bin, Bed_Label, Bath_Label, state) %>%
    # Convert tibble rows to a list of vectors (the core transaction items)
    rowwise() %>% # operate row-by-row
    mutate(basket_struct = list(c(as.character(Price_Bin), as.character(Size_Bin),
                                  as.character(Bed_Label), as.character(Bath_Label),
                                  as.character(state)))) %>%
    ungroup()

# 2. Process Amenities and add them to the transactions
# This is R's way to split and append the comma-separated amenities string.
transactions_list <- mapply(function(struct, amenities_str) {
    # Check if amenities is not 'NoAmenities'
    if (amenities_str != 'NoAmenities') {
        # Split by comma, remove leading/trailing whitespace, and remove empty strings
        amenities_items <- strsplit(amenities_str, ",")[[1]] %>%
            str_trim() %>%
            keep(~ . != "") # Remove empty strings

        return(c(struct, amenities_items))
    } else {
        return(struct)
    }
}, df_items$basket_struct, df_assoc$amenities, SIMPLIFY = FALSE)

# Convert the final list of transactions to the 'transactions' object required by arules
# This step is the equivalent of TransactionEncoder() in Python
transactions_r <- as(transactions_list, "transactions")

cat(paste("Created", length(transactions_r), "transactions.\n"))
# Print a sample transaction
cat("Sample Transaction 1:", paste(as.character(transactions_r@itemInfo$labels[transactions_r@data[,1] == T]), collapse = ", "), "\n\n")

# ==========================================
# PART 4: Association Analysis (Apriori)
# ==========================================
cat("--- Step 4: Running Apriori Algorithm ---\n")

# Find Frequent Itemsets and Generate Rules (equivalent to apriori and association_rules)
# We set the parameters directly in the 'apriori' function.

# min_support=0.05
# min_threshold=1.2 (for lift)
rules <- apriori(
    transactions_r,
    parameter = list(
        support = 0.05,
        confidence = 0.0, # Start with 0 confidence, we'll filter by lift later
        maxlen = 10,
        target = "rules"
    ),
    appearance = NULL # No specific item constraints
)

cat(paste("Found", length(rules), "frequent rules (before lift filtering).\n"))

# Filter rules based on the 'min_threshold' for lift (Lift > 1.2)
# This is the R equivalent of the 'min_threshold' filter in Python's association_rules call.
rules_filtered <- subset(rules, subset = lift > 1.2)

# Sort rules by Confidence and Lift
rules_sorted <- sort(rules_filtered, by = c("confidence", "lift"), decreasing = TRUE)

# ==========================================
# PART 5: Interpreting Results
# ==========================================
cat("\n--- Step 5: Top 10 Association Rules ---\n")

# Use 'inspect' to display the rules in a readable format
inspect(head(rules_sorted, 10))

# Filter for "Price" specific rules
cat("\n--- Specific Insight: What leads to 'Rent_High'? ---\n")
high_rent_rules <- subset(rules_sorted, subset = rhs %in% "Rent_High")
if (length(high_rent_rules) > 0) {
    inspect(head(high_rent_rules, 5))
} else {
    cat("No strong rules found pointing directly to Rent_High with current thresholds.\n")
}

cat("\n--- Specific Insight: What leads to 'Rent_Low'? ---\n")
low_rent_rules <- subset(rules_sorted, subset = rhs %in% "Rent_Low")
if (length(low_rent_rules) > 0) {
    inspect(head(low_rent_rules, 5))
} else {
    cat("No strong rules found pointing directly to Rent_Low with current thresholds.\n")
}