"""
fall_language.py
Consolidated pipeline for Symbolic Language-based Fall Detection.
"""
import numpy as np
from pyts.approximation import PiecewiseAggregateApproximation, SymbolicAggregateApproximation
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from costream.model import CostClassifierCV 
import joblib

class MultivariateMotionTokenizer:
    """
    Dynamically handles 1D (Magnitude) or 3D (Tri-axial) signals.
    Extracts raw amplitude and (optionally) first derivatives, compresses them via PAA,
    and discretizes them via SAX to form rich multi-character "motion words".
    """
    def __init__(self, n_bins=4, strategy='normal', word_size=30, use_diff=True, compression='paa'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.word_size = word_size
        self.use_diff = use_diff
        self.compression = compression
        self.paa = PiecewiseAggregateApproximation(window_size=None, output_size=word_size)
        self.sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy=strategy)
        self.is_fitted = False

    def _get_views(self, X):
        """
        Standardizes input to (N, C, T).
        Returns a list of views (Z-normalized channels and gradients).
        """
        # Auto-reshape 1D inputs to 3D: (N, T) -> (N, 1, T)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
        N, C, T = X.shape
        views =[]
        for c in range(C):
            v = X[:, c, :]
            # 1. Z-Normalize Raw Channel
            v_norm = (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + 1e-6)
            views.append(v_norm)
            
            # 2. Extract & Z-Normalize Gradient
            if self.use_diff:
                g = np.diff(v, axis=1, prepend=v[:, :1])
                g_norm = (g - g.mean(axis=1, keepdims=True)) / (g.std(axis=1, keepdims=True) + 1e-6)
                views.append(g_norm)
        return views
    
    def _compress(self, v):
        if self.compression == "paa":
            return self.paa.transform(v)
        elif self.compression == "pip":
            N = v.shape[0]
            out = np.zeros((N, self.word_size))
            for i in range(N):
                idx = self._get_pip_indices(v[i], self.word_size)
                out[i] = v[i][idx]
            return out
    
    def _get_pip_indices(self, y, k):
        n = len(y)
        if k >= n:
            return np.arange(n)
        
        # initialize with first and last
        pip_idx = np.zeros(k, dtype=int)
        pip_idx[0] = 0
        pip_idx[1] = n - 1
        num_pips = 2

        x = np.arange(n)
        for _ in range(k - 2):
            best_dist = -1
            best_idx = -1
            # sort current pips
            current = np.sort(pip_idx[:num_pips])
            for i in range(len(current) - 1):
                start = current[i]
                end = current[i + 1]
                if end - start <= 1:
                    continue
                xs = x[start + 1:end]
                ys = y[start + 1:end]
                # line through endpoints
                x1, y1 = start, y[start]
                x2, y2 = end, y[end]
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                # vectorized distances
                d = np.abs(ys - (slope * xs + intercept))
                idx_local = np.argmax(d)
                dist_local = d[idx_local]
                if dist_local > best_dist:
                    best_dist = dist_local
                    best_idx = xs[idx_local]
            if best_idx == -1:
                break
            pip_idx[num_pips] = best_idx
            num_pips += 1
        return np.sort(pip_idx[:num_pips])

    def fit(self, X):
        views = self._get_views(X)
        # Stack all views to find global SAX breakpoints (ensures consistency)
        combined = np.vstack([self._compress(v) for v in views])
        self.sax.fit(combined)
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("Tokenizer not fitted! Call fit() before transform().")
        N = X.shape[0]
        views = self._get_views(X)
        
        # Apply compression and SAX to each view independently
        views_sax = [self.sax.transform(self._compress(v)) for v in views]
        X_sentences =[]
        for i in range(N):
            # Zip tokens from all views for window i
            window_tokens = zip(*[v[i] for v in views_sax])
            # Join into "Words" (e.g., "aabbcc") and then into a "Sentence"
            merged = ["".join(letters) for letters in window_tokens]

            # --- Impact token ---
            raw_signal = X[i] if X.ndim == 2 else X[i, 0]
            peak = np.max(np.abs(raw_signal))
            mean = np.mean(np.abs(raw_signal))
            ratio = peak / (mean + 1e-6)
            if peak > 2.5:
                impact = "impact_high"
            elif peak > 1.8:
                impact = "impact_med"
            else:
                impact = "impact_low"
            if ratio > 2.2:
                rel_impact = "rel_impact_high"
            elif ratio > 1.6:
                rel_impact = "rel_impact_med"
            else:
                rel_impact = "rel_impact_low"
            merged.extend([impact, rel_impact])
            X_sentences.append(" ".join(merged))
        return X_sentences

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

class FallLM(BaseEstimator, ClassifierMixin):
    """
    Scikit-Learn wrapper for the Language of Falls pipeline.
    """
    def __init__(
        self, 
        n_bins=4, 
        strategy='normal',
        word_size=30,           # e.g. 30 tokens for a 3-second window
        use_diff=True,          # Combine Amplitude + Gradient
        ngram_range=(1, 2),     # Unigrams and Bigrams
        max_features=5000, 
        vectorizer_type='tfidf',# 'tfidf' or 'count'
        base_estimator= None,
        alpha=2.0,              # Cost parameter for False Negatives
        cv=5,
        random_state=42,
        tune_threshold=True,
        use_pip=False
    ):
        # Scikit-Learn strictly requires exactly these assignments in __init__
        self.n_bins = n_bins
        self.strategy = strategy
        self.word_size = word_size
        self.use_diff = use_diff
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer_type = vectorizer_type
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.base_estimator = base_estimator
        if base_estimator is None:
            self.base_estimator = LogisticRegression(
                class_weight='balanced', max_iter=2000,
                random_state=self.random_state
            )
        # Internal state
        self.tokenizer_ = None
        self.vectorizer_ = None
        self.classifier_ = None
        self.threshold_ = 0.5 
        self.tune_threshold = tune_threshold
        self.use_pip = use_pip
    
    def fit(self, X, y):
        self.tokenizer_ = MultivariateMotionTokenizer(
            n_bins=self.n_bins, 
            strategy=self.strategy, 
            word_size=self.word_size,
            use_diff=self.use_diff,
            compression="pip" if self.use_pip else "paa"
        )
        X_sentences = self.tokenizer_.fit_transform(X)
     
        # Vectorization
        vec_class = TfidfVectorizer if self.vectorizer_type == 'tfidf' else CountVectorizer
        self.vectorizer_ = vec_class(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            token_pattern=r"(?u)\b\w+\b"
        )
        X_tfidf = self.vectorizer_.fit_transform(X_sentences)
        # Base Estimator Selection
        if self.base_estimator is None:
            self.base_estimator = LogisticRegression(class_weight='balanced', max_iter=2000,
                                                     random_state=self.random_state)
            self.classifier_type = 'lr'

        if self.tune_threshold:
            # 4. CostClassifierCV
            self.classifier_ = CostClassifierCV(
                base_estimators=[self.base_estimator],
                cv=self.cv,
                alpha=self.alpha,
                method="stacking",
                random_state=self.random_state
            )
        else:
            self.classifier_ = self.base_estimator
        
        self.classifier_.fit(X_tfidf, np.array(y))

        if self.tune_threshold:
            self.threshold_ = self.classifier_.threshold_
        
        return self

    def predict_proba(self, X):
        X_sentences = self.tokenizer_.transform(X)
        X_tfidf = self.vectorizer_.transform(X_sentences)
        return self.classifier_.predict_proba(X_tfidf)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold_).astype(int)

    def show_grammar(self, top_n=10, n_bins=4):
        """
        Decodes the optimal 6-letter tokens back into human-readable physics.
        """
        if not isinstance (self.base_estimator,  LogisticRegression):
            print("Grammar extraction requires Logistic Regression ('lr').")
            return
            
        feature_names = self.vectorizer_.get_feature_names_out()
        fitted_lr = self.classifier_.fitted_estimators_[0]
        
        try:
            coefficients = fitted_lr.coef_[0]
        except AttributeError:
            coefficients = np.mean([est.base_estimator.coef_[0] for est in fitted_lr.calibrated_classifiers_], axis=0)
            
        sorted_indices = np.argsort(coefficients)
        
        # --- The Post-Processing Translator ---
        # Define the exact same mapping we discussed
        chars =[chr(97 + i) for i in range(n_bins)]
        if n_bins == 3:
            val_adjs =['Drop', 'Mid', 'Peak']
            grad_adjs =['Plunge', 'Flat', 'Surge']
        elif n_bins == 4:
            val_adjs =['Drop', 'Low', 'High', 'Peak']
            grad_adjs = ['Plunge', 'Sink', 'Rise', 'Surge']
        else:
            val_adjs =[f'Val{i}' for i in range(n_bins)]
            grad_adjs = [f'Grad{i}' for i in range(n_bins)]
            
        axes = ['AP', 'ML', 'V']
        
        def decode_word(word):
            if len(word) != 6: return word # Fallback if not a 6-letter word
            
            decoded =[]
            view_idx = 0
            for axis in axes:
                # Value
                val_idx = chars.index(word[view_idx]) if word[view_idx] in chars else 0
                decoded.append(f"[{axis}_{val_adjs[val_idx]}]")
                view_idx += 1
                
                # Gradient (if use_diff=True)
                grad_idx = chars.index(word[view_idx]) if word[view_idx] in chars else 0
                decoded.append(f"[{axis}_{grad_adjs[grad_idx]}]")
                view_idx += 1
                
            return "".join(decoded)

        def translate_ngram(ngram):
            # If it's a bigram (e.g., "acdbca bbcdaa"), split it, decode each, and rejoin with " -> "
            words = ngram.split()
            decoded_words = [decode_word(w) for w in words]
            return "  ➔  ".join(decoded_words)

        # --- Printing the Table ---
        print("\n" + "="*80)
        print("THE GRAMMAR OF FALLS (Highest Predictors)")
        print("="*80)
        for idx in sorted_indices[-top_n:][::-1]:
            raw_ngram = feature_names[idx]
            translation = translate_ngram(raw_ngram)
            print(f"Weight: +{coefficients[idx]:.4f} | Raw: {raw_ngram:<13} | {translation}")
            
        print("\n" + "="*80)
        print("THE GRAMMAR OF NORMALCY (Strongest Negatives)")
        print("="*80)
        for idx in sorted_indices[:top_n]:
            raw_ngram = feature_names[idx]
            translation = translate_ngram(raw_ngram)
            print(f"Weight: {coefficients[idx]:.4f} | Raw: {raw_ngram:<13} | {translation}")