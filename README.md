# Comment Category Prediction Challenge

## Project Overview

This project is a multi-class text classification solution for the **Comment Category Prediction Challenge** on Kaggle. The goal is to classify social media comments into one of four categories (labels 0, 1, 2, 3) based on the comment text and associated metadata such as upvotes, downvotes, identity flags, emoticons, and posting time.

The final solution achieves a **Kaggle public leaderboard score of 0.82019** using a soft-vote ensemble of LightGBM, XGBoost, and Logistic Regression trained on a rich 25,024-feature sparse matrix.

---

## Dataset

| File | Rows | Columns | Description |
|---|---|---|---|
| `train.csv` | 198,000 | 15 | Training data with labels |
| `test.csv` | 102,000 | 14 | Test data without labels |
| `Sample.csv` | 102,000 | 2 | Submission format template |

### Columns

| Column | Type | Description |
|---|---|---|
| `created_date` | datetime | Timestamp of comment |
| `post_id` | int | Thread/post identifier |
| `emoticon_1/2/3` | int | Emoticon usage counts |
| `upvote` | int | Number of upvotes |
| `downvote` | int | Number of downvotes |
| `if_1`, `if_2` | int | Identity flag scores |
| `race`, `religion`, `gender` | object | Identity category (73% missing) |
| `disability` | bool | Disability flag |
| `comment` | object | Raw comment text |
| `label` | int | Target: 0, 1, 2, or 3 |

### Class Distribution (Training Data)

| Label | Count | Proportion |
|---|---|---|
| 0 | 114,151 | 57.7% |
| 2 | 62,460 | 31.5% |
| 1 | 15,919 | 8.0% |
| 3 | 5,470 | **2.8%** |

> **Note:** The dataset is heavily imbalanced. Label 3 has fewer than 3% of samples, which requires special handling via `class_weight="balanced"` and macro F1 as the evaluation metric.

---

## Project Structure

```
notebook.ipynb               ← Main Kaggle notebook (203 cells)
submission.csv               ← Final predictions for Kaggle submission
README.md                    ← This file
```

### Notebook Sections

| Section | Cell Range | Description |
|---|---|---|
| Setup & Imports | 1–2 | Libraries and Kaggle file listing |
| Data Loading | 3–4 | Load train, test, sample CSVs |
| Milestone 1 Questions | 6–33 | EDA questions (commented out) |
| Exploratory Data Analysis | 35–53 | Visualisations and statistical tests |
| Data Preprocessing | 55–56 | Cleaning, encoding, feature engineering |
| Feature-Target Split | 57–58 | X/y separation and feature groups |
| Train-Validation Split | 60–61 | 80/20 stratified split |
| Preprocessing Pipelines | 62–65 | Numeric, categorical, text transformers |
| Baseline Model | 67–77 | Logistic Regression + confusion matrix |
| Linear Models | 79–90 | LR, SVM, SGD + GridSearchCV |
| Milestone 2 Questions | 92–112 | Text feature questions (commented out) |
| Dimensionality Reduction | 114–118 | TruncatedSVD + Chi-Square selection |
| Multiple Models | 120–123 | Naive Bayes, KNN, LinearSVC |
| Milestone 3 Questions | 125–144 | Preprocessing questions (commented out) |
| Ensemble Models | 146–149 | Bagging, Boosting, Stacking, MLP |
| Milestone 4 Questions | 152–169 | Advanced ML questions (commented out) |
| Milestone 5 Questions | 171–191 | End-to-end pipeline questions (commented out) |
| LightGBM / XGBoost | 193–194 | Milestone comparison models |
| Model Comparison | 196 | F1 comparison table |
| Final Prediction | 198–202 | Test preprocessing, retrain, submission |

---

## Methodology

### 1. Exploratory Data Analysis

Key findings from EDA that shaped the solution:

- **73% missing values** in `race`, `religion`, `gender` → converted to binary presence flags
- **Severe class imbalance** (label 3 = 2.8%) → requires `class_weight="balanced"` and macro F1 metric
- **Right-skewed distributions** in `upvote`, `downvote`, `if_1`, `if_2` → log transformation required
- **ANOVA test** on `engagement_score` (F=78.3, p≈1.2e-50) confirms engagement is statistically significant across labels
- Correlation heatmap reveals low multicollinearity between engineered features

### 2. Data Preprocessing

#### Identity Columns
```python
df["race"]     = df["race"].notnull().astype(int)
df["religion"] = df["religion"].notnull().astype(int)
df["gender"]   = df["gender"].notnull().astype(int)
```
Converted to binary (1 = topic detected, 0 = not mentioned) to eliminate the 73% missing value problem.

#### Log Transform
```python
df["upvote"]   = np.log1p(df["upvote"])
df["downvote"] = np.log1p(df["downvote"])
df["if_1"]     = np.log1p(df["if_1"])
df["if_2"]     = np.log1p(df["if_2"])
```
`log1p(x) = log(1+x)` compresses right-skewed distributions. Handles zero values safely.

#### Datetime Features
```python
df["hour"]       = df["created_date"].dt.hour
df["day_of_week"] = df["created_date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
```

#### Text Cleaning
```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # letters only
    text = re.sub(r"\s+", " ", text).strip()  # normalise whitespace
    return text
```

### 3. Feature Engineering

| Feature | Description | Motivation |
|---|---|---|
| `word_count` | Total words in comment | Comment verbosity signal |
| `unique_words` | Distinct word count | Lexical richness; spam has low uniqueness |
| `caps_ratio` | Proportion of uppercase chars | Emotional intensity proxy |
| `punct_count` | Punctuation character count | Exclamation/question mark patterns |
| `vote_ratio` | (upvote+1)/(downvote+1) | Community reception ratio |
| `total_votes` | upvote + downvote | Engagement volume |
| `emoticon_sum` | emoticon_1 + 2 + 3 | Total emotional expression |
| `identity_flag` | Sum of race+religion+gender+disability | Count of identity topics mentioned |

### 4. Feature Matrix Construction

Three feature groups are built and horizontally stacked into a single sparse matrix:

```
Feature Matrix = [TF-IDF (25,000)] | [Numeric scaled (14)] | [Binary passthrough (10)]
               = 25,024 total features per comment
Shape: (158,400 rows × 25,024 columns) — training
       (39,600  rows × 25,024 columns) — validation
```

**TF-IDF Vectorizer settings:**
```python
TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 2),     # unigrams + bigrams
    min_df=3,               # ignore words in < 3 documents
    sublinear_tf=True,      # log(1+count) instead of raw count
    strip_accents="unicode",
    analyzer="word"
)
```

**Why sparse matrices (`hstack` not `pd.concat`):**
A 158,400 × 25,000 dense float64 matrix = **31 GB RAM**. Sparse format stores only non-zero values (~300 MB). `pd.concat` would convert to dense; `scipy.sparse.hstack` maintains sparsity.

**Why `StandardScaler(with_mean=False)`:**
Standard scaling subtracts the mean, converting all sparse zeros to non-zero values, destroying sparsity. `with_mean=False` skips mean subtraction (divides by std only), preserving the sparse format.

### 5. Models Trained

#### Baseline
| Model | Setup | Val F1 Macro |
|---|---|---|
| Logistic Regression | 10k TF-IDF + MinMaxScaler, Pipeline | ~0.77 |

#### Linear Models (Milestone Comparison)
| Model | Val F1 Macro |
|---|---|
| Logistic Regression (C=1.0) | 0.771 |
| Linear SVM (C=1.0) | 0.777 |
| SGD Classifier (log_loss) | 0.768 |
| SGD + GridSearchCV (best params) | 0.775 |

#### Dimensionality Reduction
| Model | Val F1 Macro |
|---|---|
| TruncatedSVD (300 components) + LR | 0.688 |
| Chi-Square SelectKBest (k=8000) + LR | 0.771 |

#### Multiple Model Types
| Model | Val F1 Macro |
|---|---|
| Naive Bayes | 0.496 |
| KNN (k=5, SVD=300) | 0.408 |
| Linear SVM | 0.777 |

#### Ensemble Models
| Model | Val F1 Macro |
|---|---|
| Bagging (50 Decision Trees) | 0.767 |
| Boosting (AdaBoost, 100 trees) | 0.593 |
| Stacking (LR + SVM + NB → LR meta) | 0.798 |
| MLP (128→64, ReLU, Adam) | 0.787 |

#### Rich Feature Matrix Models (Main Improvement)
| Model | Val Accuracy | Val F1 Macro |
|---|---|---|
| LightGBM (800 trees) | 0.91439 | 0.81689 |
| XGBoost (600 trees) | 0.90902 | 0.78521 |
| Logistic Regression (saga) | 0.89639 | 0.74367 |
| **Soft-Vote Ensemble** | **0.91543** | **0.81272** |

### 6. Soft-Vote Ensemble

Instead of hard voting (majority class wins), each model outputs class probabilities. A weighted average is computed and the highest-probability class is selected:

```python
test_proba = wl * lgbm.predict_proba(X) +
             wx * xgb.predict_proba(X)  +
             wlr * lr.predict_proba(X)
predictions = np.argmax(test_proba, axis=1)
```

**Best weights found by grid search on validation set:**
```
LGBM = 0.35,  XGB = 0.30,  LR = 0.05
```

**Why ensemble beats any single model:** Each model captures different patterns. LightGBM (leaf-wise growth) and XGBoost (level-wise, second-order Taylor approximation) make different errors. Logistic Regression contributes strong linear text signal. Their combined probability is more reliable than any single model's prediction.

### 7. Final Retrain on Full Data

After validation confirms the pipeline works, all three models are retrained on the complete 198,000-row training set:

```python
tfidf_sb.fit_transform(X2["comment"])   # re-fits on ALL 198k comments
f_lgbm.fit(X_full, y2)                  # trains on ALL 198k rows
f_xgb.fit(X_full, y2)
f_lr.fit(X_full, y2)
```

**Why:** During validation, 39,600 rows (20%) were held out. After confirming performance, there's no reason to waste those examples. More training data improves generalisation, especially for rare label 3 (from 4,376 to 5,470 examples).

---

## Results

| Metric | Score |
|---|---|
| Kaggle Public Leaderboard | **0.82019** |
| Validation Accuracy | 0.91543 |
| Validation F1 Macro | 0.81272 |

### Per-Class Validation Performance (Ensemble)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 0.98 | 0.95 | 0.96 |
| 1 | 0.79 | 0.77 | 0.78 |
| 2 | 0.85 | 0.92 | 0.89 |
| 3 | 0.76 | 0.53 | 0.62 |

Label 3 has the lowest recall (0.53) because it has only 2.8% of training examples. `class_weight="balanced"` partially compensates but rare class prediction remains the hardest part of this problem.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Macro F1 over accuracy | Accuracy is misleading with 57.7% majority class |
| `class_weight="balanced"` | Label 3 (2.8%) would be ignored otherwise |
| `log1p` on skewed columns | Prevents extreme outliers dominating gradient updates |
| `StandardScaler(with_mean=False)` | Preserves sparsity; dense conversion would use 31GB RAM |
| `hstack` over `pd.concat` | Maintains sparse format; pd.concat would densify TF-IDF |
| Fit transformers on train only | Prevents data leakage into validation/test |
| Retrain on 100% data | Uses all available labelled data for final test prediction |
| 3 diverse models in ensemble | Different error patterns cancel out; beats any single model |
| Stratified train-val split | Preserves class ratios; ensures reliable label-3 evaluation |

---

## How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn regex scikit-learn lightgbm xgboost scipy
```

### On Kaggle

1. Open the notebook in Kaggle
2. Attach the dataset: `comment-category-prediction-challenge`
3. Click **Run All** or **Save & Run All (Commit)**
4. The final cell saves `submission.csv` to `/kaggle/working/`
5. Submit `submission.csv` to the competition

### Key Note on Session Timeout

The notebook takes approximately 7–11 hours to run fully on Kaggle's hardware. If cancelled mid-run, re-run **only the final submission cell** (cell 202) — it is self-contained and loads all data fresh from disk.

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24 | Array operations, log transforms, argmax |
| `pandas` | ≥1.5 | DataFrame manipulation, CSV I/O |
| `scikit-learn` | ≥1.2 | Preprocessing, pipelines, models, metrics |
| `lightgbm` | ≥3.3 | Gradient boosting (main model) |
| `xgboost` | ≥1.7 | Gradient boosting (ensemble member) |
| `scipy` | ≥1.10 | Sparse matrix operations (hstack, csr_matrix) |
| `matplotlib` | ≥3.6 | Base plotting |
| `seaborn` | ≥0.12 | Statistical visualisations |
| `regex` | ≥2022 | Text cleaning (URL removal, character filtering) |

---

## Model Architecture Summary

```
Raw Data (198,000 × 15)
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │           PREPROCESSING                      │
  │  • Binary encode race/religion/gender        │
  │  • log1p transform upvote/downvote/if_1/if_2 │
  │  • Extract hour, day_of_week, is_weekend      │
  │  • Clean text (lowercase, remove URLs/punct)  │
  │  • Engineer 8 new features                    │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │         FEATURE MATRIX                       │
  │  TF-IDF (25,000) + Numeric (14) + Binary (10)│
  │  → Sparse matrix: 158,400 × 25,024           │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────┐
  │  LightGBM    │  │   XGBoost    │  │  Logistic│
  │  800 trees   │  │  600 trees   │  │ Regression│
  │  leaf-wise   │  │  level-wise  │  │  saga    │
  │  balanced    │  │  hist mode   │  │  C=3.0   │
  └──────┬───────┘  └──────┬───────┘  └────┬─────┘
         │                 │                │
         ▼                 ▼                ▼
  predict_proba    predict_proba    predict_proba
  (39600 × 4)      (39600 × 4)      (39600 × 4)
         │                 │                │
         └────────┬────────┘                │
                  │    wl=0.35              │ wlr=0.05
                  │    wx=0.30              │
                  └──────────────┬──────────┘
                                 │
                                 ▼
                    Weighted Soft-Vote Ensemble
                         (39600 × 4)
                                 │
                           np.argmax
                                 │
                         Predictions (39600,)
                                 │
                      Retrain on ALL 198,000 rows
                                 │
                      Predict on 102,000 test rows
                                 │
                          submission.csv
```

---

## Evaluation Metric

The competition uses **Accuracy** as the primary metric. Internally we optimise for **Macro F1** because:

- Accuracy of 57.7% is achievable by predicting all class 0 — meaningless
- Macro F1 averages F1 score equally across all 4 classes
- A model that never predicts class 3 gets F1=0 for that class, dragging macro average down
- This forces the model to learn all four classes, not just the majority

---

## Acknowledgements

- Competition hosted on Kaggle as part of the **ML Practice Project course**
- Dataset provided by the competition organisers
- Libraries: scikit-learn, LightGBM, XGBoost, SciPy, pandas, numpy
