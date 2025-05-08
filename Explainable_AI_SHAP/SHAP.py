import pandas as pd
import numpy as np
import shap
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import re

# 1. Load data (exclude unwanted columns, but keep RV for computation)
print("Loading data...")
unwanted_cols = ['BVP', 'RV_neg', 'RV_pos', 'RQ', 'Unnamed: 0', 'target', 'prev_rv']
df = pd.read_csv('/Users/macbook/Documents/PhD_Documents/XAI_embeddings/total_data/Brent/embedding_Fasttext_total_Brent.csv',
                 usecols=lambda col: col not in unwanted_cols)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. Filter the data for the specific date range
print("Filtering data for the date range 2022-02-23 to 2024-01-01...")
df_filtered = df[(df['Date'] >= '2020-07-01') & (df['Date'] < '2022-02-23')]

# 3. Compute RV change (today vs yesterday)
print("Computing RV change...")
df_filtered['prev_rv'] = df_filtered['Date'].map(df_filtered.groupby('Date')['RV'].first().shift(1))
df_filtered['rv_change'] = df_filtered['RV'] - df_filtered['prev_rv']
df_filtered.dropna(subset=['rv_change'], inplace=True)

# 4. Shift target 1 day ahead to avoid leakage
print("Aligning target with next day...")
df_filtered['tomorrow_rv_change'] = df_filtered['Date'].map(df_filtered.groupby('Date')['rv_change'].first().shift(-1))
df_filtered['target'] = (df_filtered['tomorrow_rv_change'] > 0).astype(int)
df_filtered.dropna(subset=['target'], inplace=True)

# 5. Align features: Use YESTERDAY'S news for TODAY'S prediction
print("Shifting features...")
bert_cols = [col for col in df_filtered.columns if col.startswith('feature_')]
df_filtered[bert_cols] = df_filtered.groupby('Date')[bert_cols].shift(1)
df_filtered.dropna(subset=bert_cols, inplace=True)

# 6. Split into train and test based on date
split_date = df_filtered['Date'].quantile(0.8)
train_filtered = df_filtered[df_filtered['Date'] < split_date]
test_filtered = df_filtered[df_filtered['Date'] >= split_date]

X_train, y_train = train_filtered[bert_cols], train_filtered['target']
X_test, y_test = test_filtered[bert_cols], test_filtered['target']

# Ensure target has both classes
if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
    raise ValueError("Target variable contains only one class. Adjust data splitting.")

# 7. Train the model
print("Training model...")
model = LogisticRegression(
    penalty='l2',
    C=0.1,
    solver='liblinear',
    max_iter=500,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# 8. Evaluate the model
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# 9. Compute SHAP values
print("Computing SHAP values...")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 10. Tokenize text and compute word-level SHAP impact
print("Analyzing word-level SHAP importance...")
word_impact = defaultdict(lambda: {'sum': 0, 'count': 0})

stop_words = set(stopwords.words('english'))
custom_excluded_words = {"reuter", "brief", "updat", "briefingcom", "etbbi", "et", "api", "juli", "lng", "pct", "co", "august", "termin", "septemb", "briefomv", "ag", "miss", "scripcod", "dividend", "rosneft", "petrobra", "may", "oper", "certif", "year", "result", "gener", "relianc", "calendar", "briefphillip", "say", "index", "bay", "nashik", "lpg", "daili", "intern", "lubric", "addit"}

# Process texts
for i, text in enumerate(tqdm(test_filtered['Text'].iloc[:100], desc="Processing texts")):
    words = word_tokenize(text.lower())
    filtered_words = [
        word for word in words
        if word.isalpha() and 
        word not in stop_words and 
        word not in custom_excluded_words and 
        not re.match(r".*\.com$", word)
    ]
    
    doc_shap_values = shap_values.values[i]

    if len(filtered_words) > 0:
        word_shap_contrib = np.abs(doc_shap_values).sum() / len(filtered_words)
        for word in filtered_words:
            word_impact[word]['sum'] += word_shap_contrib
            word_impact[word]['count'] += 1

# 11. Compute average SHAP importance per word
word_stats = [
    {'word': word, 'avg_impact': vals['sum'] / vals['count'], 'frequency': vals['count']}
    for word, vals in word_impact.items() if vals['count'] > 1
]

word_df = pd.DataFrame(word_stats).sort_values('avg_impact', ascending=False)

# 12. Display top words
print("\nTop 20 influential words based on SHAP:")
print(word_df.head(20))

# 13. Improved Plot
plt.figure(figsize=(12, 7))
top_words = word_df.head(20)

bars = plt.barh(top_words['word'], top_words['avg_impact'], color='#ff4d4d')  # Richer pink

# Add value labels to each bar (closer to right edge)
for bar in bars:
    plt.text(bar.get_width() + 0.0003,  # Adjust placement for better alignment
             bar.get_y() + bar.get_height() / 2,  
             f"{bar.get_width():.4f}",  
             va='center', ha='left', fontsize=12, fontweight='bold', color='black')

plt.title("Top 20 Words by SHAP Impact (1-Day Ahead Prediction)", fontsize=14, fontweight='bold')
plt.xlabel("Average Absolute SHAP Value", fontsize=12)
plt.ylabel("Words", fontsize=12)
plt.gca().invert_yaxis()  # Highest impact word at top

plt.show()