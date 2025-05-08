# Load required package
library(stats)

# Read predictions from CSV files 

#df1 <- read.csv("/Users/macbook/Documents/PhD_Documents/ECAI/McNemar_test/prediction_files/all_models_ensemble_gemini.csv", header = TRUE)
#df2 <- read.csv("/Users/macbook/Documents/PhD_Documents/ECAI/McNemar_test/prediction_files/all_models_ensemble_BERT.csv", header = TRUE)

df1 <- read.csv("/Users/macbook/Documents/PhD_Documents/ECAI/McNemar_test/prediction_files/all_models_ensemble_HAR.csv", header = TRUE)
df2 <- read.csv("/Users/macbook/Documents/PhD_Documents/ECAI/McNemar_test/prediction_files/all_models_ensemble_Vader.csv", header = TRUE)
# Extract actual values and predictions
# Assumes both files have the same Actual values
actual <- df1$Actual
pred1 <- df1$Predicted
pred2 <- df2$Predicted

# Logical vectors: TRUE if prediction is correct, FALSE otherwise
correct1 <- pred1 == actual
correct2 <- pred2 == actual

# Create 2x2 contingency table: [correct1, correct2]
contingency_table <- table(correct1, correct2)

# Optional: Label rows and columns for clarity
dimnames(contingency_table) <- list("Model 1 Correct" = c("No", "Yes"),
                                    "Model 2 Correct" = c("No", "Yes"))

# Print the contingency table
cat("Contingency Table:\n")
print(contingency_table)

# Perform McNemar's test (with continuity correction)
mcnemar_result <- mcnemar.test(contingency_table, correct = TRUE)

# Print test result
cat("\nMcNemar's Test Result:\n")
print(mcnemar_result)
