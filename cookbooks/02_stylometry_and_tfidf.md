# Cookbook 02: Stylometry & TF-IDF on Local CPU

## Purpose
Enables sub-minute CPU feature extraction and training using multi-granular lexical and character n-grams combined with dense stylometric markers.

## Feature Extraction Recipe
```python
from src.features.stylometry import StylometricFeatureExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# 1. Stylometric features
extractor = StylometricFeatureExtractor()
X_sty_train = extractor.fit_transform(train_df['text'])
X_sty_test = extractor.transform(test_df['text'])

# 2. Multi-granular TF-IDF
word_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=25000, sublinear_tf=True)
char_vec = TfidfVectorizer(ngram_range=(2, 6), analyzer='char', max_features=35000, sublinear_tf=True)
char_wb_vec = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=25000, sublinear_tf=True)

# Combine sparse representations
X_text = hstack([word_vec.fit_transform(train_df['text']),
                 char_vec.fit_transform(train_df['text']),
                 char_wb_vec.fit_transform(train_df['text'])])
```

## Out-of-Fold Linear Training
```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

# Train fold by fold using the locked folds to avoid any data leakage
```
