"""
fall_language.py
Consolidated pipeline for Symbolic Language-based Fall Detection.
"""
import numpy as np
from pyts.approximation import PiecewiseAggregateApproximation, SymbolicAggregateApproximation
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
import joblib

class MultivariateMotionTokenizer:
    """
    Dynamically handles 1D (Magnitude) or 3D (Tri-axial) signals.
    Extracts raw amplitude and (optionally) first derivatives, compresses them via PAA,
    and discretizes them via SAX to form rich multi-character "motion words".
    """
    def __init__(self, n_bins=4, strategy='normal', word_size=30, use_diff=True, compression='paa', use_impact=True, impact_mode='category'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.word_size = word_size
        self.use_diff = use_diff
        self.use_impact = use_impact
        self.impact_mode = impact_mode
        self.compression = compression
        self.paa = PiecewiseAggregateApproximation(window_size=None, output_size=word_size)
        self.sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy=strategy)
        self.is_fitted = False

    def _get_views(self, X):
        """
        Standardizes input to (N, C, T).
        Returns a list of views (Z-normalized channels and gradients).
        """
        # Reshape 1D inputs to 3D: (N, T) -> (N, 1, T)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
        N, C, T = X.shape
        views =[]
        for c in range(C):
            v = X[:, c, :]
            # Z-Normalize Raw Channel
            v_norm = (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + 1e-6)
            views.append(v_norm)
            
            # Extract & Z-Normalize Gradient
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
        # Stack all views to find global SAX breakpoints
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
            # Join into words
            merged = ["".join(letters) for letters in window_tokens]

            if self.use_impact:
                raw_signal = X[i] if X.ndim == 2 else X[i, 0]
                peak = np.max(np.abs(raw_signal))
                mean = np.mean(np.abs(raw_signal))
                ratio = peak / (mean + 1e-6)
                if self.impact_mode == 'physical':
                    # Discretize in fixed physical units — dataset-independent
                    peak_bin = min(round(peak * 2) / 2, 6.0)   # nearest 0.5g, capped at 6g
                    ratio_bin = min(round(ratio * 2) / 2, 6.0)  # nearest 0.5, capped at 6
                    impact = f"impact_{peak_bin:.1f}g"
                    rel_impact = f"rel_impact_{ratio_bin:.1f}"
                else:
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
        word_size=30,           
        use_diff=True,          
        ngram_range=(1, 2),     
        max_features=5000, 
        vectorizer_type='tfidf', #'tfidf' or 'count'
        base_estimator= None,
        alpha=2.0,              
        cv=5,
        random_state=42,
        tune_threshold=True,
        use_pip=False,
        use_impact=True,
        impact_mode='category'
    ):
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
        self.use_impact = use_impact
        self.impact_mode = impact_mode
    
    def fit(self, X, y):
        self.tokenizer_ = MultivariateMotionTokenizer(
            n_bins=self.n_bins,
            strategy=self.strategy,
            word_size=self.word_size,
            use_diff=self.use_diff,
            use_impact=self.use_impact,
            impact_mode=self.impact_mode,
            compression="pip" if self.use_pip else "paa"
        )
        X_sentences = self.tokenizer_.fit_transform(X)

        # Clamp ngram_range to actual sentence length so disabled tokens don't cause empty vocabulary
        sentence_len = len(X_sentences[0].split())
        max_n = min(self.ngram_range[1], sentence_len)
        min_n = min(self.ngram_range[0], max_n)

        # Vectorization
        vec_class = TfidfVectorizer if self.vectorizer_type == 'tfidf' else CountVectorizer
        self.vectorizer_ = vec_class(
            ngram_range=(min_n, max_n),
            max_features=self.max_features,
            token_pattern=r"(?u)\b\w+\b"
        )
        X_tfidf = self.vectorizer_.fit_transform(X_sentences)
        # Base Estimator Selection
        if self.base_estimator is None:
            self.base_estimator = LogisticRegression(class_weight='balanced', max_iter=2000,
                                                     random_state=self.random_state)
            self.classifier_type = 'lr'

        self.classifier_ = self.base_estimator
        self.classifier_.fit(X_tfidf, np.array(y))

        if self.tune_threshold:
            probs = cross_val_predict(
                clone(self.classifier_), X_tfidf, np.array(y),
                cv=self.cv, method='predict_proba'
            )[:, 1]
            fpr, tpr, roc_thresholds = roc_curve(y, probs)
            youden_j = tpr + (1 - fpr) - 1
            self.threshold_ = float(roc_thresholds[np.argmax(youden_j)])
        
        return self

    def predict_proba(self, X):
        X_sentences = self.tokenizer_.transform(X)
        X_tfidf = self.vectorizer_.transform(X_sentences)
        return self.classifier_.predict_proba(X_tfidf)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold_).astype(int)