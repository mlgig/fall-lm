import numpy as np, pandas as pd, os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from scipy.signal import resample_poly, find_peaks
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from sklearn.base import clone
from costream.segmentation.streaming_segmenter import (
    sliding_window_inference,
    generate_sliding_windows,
)
from costream.evaluation.event_detection import get_high_confidence_regions, iou
from costream.data import make_csv_loader
from costream.segmentation import create_training_data


def segment_and_save(
    train_dfs, test_dfs, feat_cols, win_sec, thresh, suffix="", freq=100, orig_freq=100
):
    """Segment and save train test data with subject-level splits."""

    if freq != orig_freq:
        print(f"Resampling from {orig_freq}Hz to {freq}Hz...")
        train_dfs = [resample_df(df, feat_cols, orig_freq, freq) for df in train_dfs]
        test_dfs = [resample_df(df, feat_cols, orig_freq, freq) for df in test_dfs]

    X_train, y_train = create_training_data(
        train_dfs,
        feature_cols=feat_cols,
        window_size=win_sec,
        step=1.0,
        freq=freq,
        signal_thresh=thresh,
        drop_below_threshold=True,
        spacing="multiphase",
        use_post_event_data=False,
    )

    # Extract test signals and events (streaming evaluation format)
    test_signals, test_events = extract_test_signals_events(test_dfs, feat_cols)

    # Handle 3D case
    if suffix == "3d":
        X_train = X_train.transpose(0, 2, 1)  # (N, 3, T)
        X_train, y_train = filter_active_windows(X_train, y_train, threshold=1.4)
    d = "" if suffix == "" else f"_{suffix}"

    # Save all splits
    np.save(f"data/preprocessed/X_train{d}_{win_sec}.npy", X_train)
    np.save(f"data/preprocessed/y_train{d}_{win_sec}.npy", y_train)
    np.save(
        f"data/preprocessed/test_signals{d}_{win_sec}.npy",
        np.array(test_signals, dtype=object),
    )
    np.save(
        f"data/preprocessed/test_events{d}_{win_sec}.npy",
        np.array(test_events, dtype=object),
    )

    print(f"WIN_SEC={win_sec}:")
    print(
        f"  Train: X={X_train.shape}, y={y_train.shape}, balance={np.bincount(y_train)} (0=ADL, 1=Fall)"
    )
    print(f"  Test:  {len(test_signals)} signals, {len(test_events)} event lists")

    return X_train, y_train, test_signals, test_events


def extract_test_signals_events(test_dfs, feat_cols):
    """Build (test_signals, test_events) for streaming evaluation from raw test dfs."""
    if len(feat_cols) == 1:
        test_signals = [df[feat_cols[0]].values for df in test_dfs]
    else:
        test_signals = [df[feat_cols].values for df in test_dfs]

    test_events = []
    for df in test_dfs:
        falls = np.where(df["label"] > 0)[0]
        test_events.append(falls.tolist())

    return test_signals, test_events


def _event_list(ep):
    return (
        []
        if (isinstance(ep, int) and ep == -1)
        else ([ep] if isinstance(ep, int) else [e for e in ep if e != -1])
    )


def _gt_ranges(events, window_size, tolerance, freq):
    gs = []
    tol = int(tolerance * freq)
    for ep in events:
        left = (ep - freq) - int((window_size + tolerance) * freq)
        right = (ep + int((window_size - 1) * freq)) + tol
        gs.append(range(int(left), int(right)))
    return gs


def _extract_window(sig, anchor, win_samples, pre_event_samples=100):
    s = max(0, anchor - pre_event_samples)  # same as multiphase (+1s pre-event)
    e = s + win_samples
    if e > len(sig):
        e = len(sig)
        s = max(0, e - win_samples)
    w = sig[s:e]
    return w if len(w) == win_samples else None


def compute_impact_features(window, freq=100, impact_phase_secs=1.0, pre_event_secs=1.0):
    """Compute candidate impact/rel_impact magnitudes for a single window.

    Returns the localized impact-phase peak alongside several "rest of window"
    statistics, plus a few candidate ratio definitions for comparison.
    """
    start = int(pre_event_secs * freq)
    end = start + int(impact_phase_secs * freq)
    impact_phase = np.abs(window[start:end])
    rest = np.abs(np.concatenate([window[:start], window[end:]]))

    peak = np.max(impact_phase)
    rest_mean = np.mean(rest)
    rest_max = np.max(rest)
    rest_std = np.std(rest)

    return {
        "peak": peak,
        "rest_mean": rest_mean,
        "rest_max": rest_max,
        "rest_std": rest_std,
        "ratio_mean": peak / (rest_mean + 1e-6),
        "ratio_max": peak / (rest_max + 1e-6),
        "zscore": (peak - rest_mean) / (rest_std + 1e-6),
    }


def compute_jerk_features(window):
    """Compute candidate jerk (rate-of-change) magnitudes for a single window.

    Mirrors compute_impact_features, but operates on the first difference of
    the raw signal over the whole window (no phase localization).
    """
    jerk = np.abs(np.diff(window))

    peak = np.max(jerk)
    mean = np.mean(jerk)
    std = np.std(jerk)

    return {
        "jerk_peak": peak,
        "jerk_mean": mean,
        "jerk_std": std,
        "jerk_ratio": peak / (mean + 1e-6),
        "jerk_zscore": (peak - mean) / (std + 1e-6),
    }


def compute_peak_count_features(window, freq=100, height_mult=1.5, min_distance_secs=0.3):
    """Count distinct peaks in a window, relative to its own baseline.

    Uses a height threshold relative to the window's own median magnitude
    (no dataset-specific calibration) and a minimum spacing in seconds, so
    it can be computed identically at training and inference time.
    """
    abs_w = np.abs(window)
    height = height_mult * np.median(abs_w)
    distance = max(1, int(min_distance_secs * freq))
    peaks, _ = find_peaks(abs_w, height=height, distance=distance)
    return {"n_peaks": len(peaks)}


def compute_kurtosis_features(window):
    """Excess kurtosis of the window: high = mostly flat with a sharp outlier,
    low = uniformly active/oscillatory. Scale-invariant, no thresholds."""
    return {"kurtosis": kurtosis(window, fisher=True)}


def compute_skewness_features(window):
    """Skewness of the window: captures asymmetry of the amplitude distribution.
    Scale/shift-invariant, no thresholds."""
    return {"skewness": skew(window)}


def symbolic_confusion_trace(
    model,
    test_signals,
    test_events,
    *,
    window_size=3.0,
    step=1.0,
    freq=100,
    signal_thresh=1.4,
    tolerance=20,
    debounce_secs=60,
    include_tn=False,
    tn_max_per_rec=20,
    return_windows=False,
):
    thresh = getattr(model, "threshold_", 0.5)
    win = int(window_size * freq)
    pre = int(1.0 * freq)
    rows = []

    for rec_id, (sig, ep) in enumerate(zip(test_signals, test_events)):
        conf, _ = sliding_window_inference(
            sig, model, window_size, step, freq, signal_thresh
        )
        alarms = get_high_confidence_regions(
            conf, threshold=thresh, min_interval_secs=debounce_secs, freq=freq
        )
        alarms = [] if alarms is None else alarms.tolist()

        events = _event_list(ep)
        gtr = _gt_ranges(events, window_size, tolerance, freq)
        matched = set()

        # TP / FP from alarms
        for a in alarms:
            dr = range(a, a + int((window_size + tolerance) * freq))
            hit = [i for i, r in enumerate(gtr) if iou(dr, r) > 0]
            w = _extract_window(sig, a, win, pre)
            token = (
                None
                if w is None
                else (
                    model.tokenizer_.transform(w[None, ...])[0]
                    if hasattr(model, "tokenizer_")
                    else None
                )
            )
            if hit:
                for i in hit:
                    matched.add(i)
                row = dict(
                    rec_id=rec_id,
                    kind="TP",
                    anchor=a,
                    token_sentence=token,
                    conf=float(conf[a]),
                )
            else:
                row = dict(
                    rec_id=rec_id,
                    kind="FP",
                    anchor=a,
                    token_sentence=token,
                    conf=float(conf[a]),
                )
            if return_windows:
                row["window"] = w
            rows.append(row)

        # FN from unmatched GT events
        for i, ev in enumerate(events):
            if i not in matched:
                w = _extract_window(sig, ev, win, pre)
                token = (
                    None
                    if w is None
                    else (
                        model.tokenizer_.transform(w[None, ...])[0]
                        if hasattr(model, "tokenizer_")
                        else None
                    )
                )
                row = dict(
                    rec_id=rec_id,
                    kind="FN",
                    anchor=ev,
                    token_sentence=token,
                    conf=float(conf[min(ev, len(conf) - 1)]),
                )
                if return_windows:
                    row["window"] = w
                rows.append(row)

        # Optional TN sampling (window-level)
        if include_tn:
            windows, indices, valid_mask, _ = generate_sliding_windows(
                sig, window_size, step, freq, signal_thresh, pad=False
            )
            tn_count = 0
            for (s, e), v in zip(indices, valid_mask):
                if (not v) or tn_count >= tn_max_per_rec:
                    continue
                wr = range(s, e)
                overlaps_gt = any(iou(wr, r) > 0 for r in gtr)
                if overlaps_gt or np.max(conf[s:e]) >= thresh:
                    continue
                w = sig[s:e]
                token = (
                    model.tokenizer_.transform(w[None, ...])[0]
                    if hasattr(model, "tokenizer_")
                    else None
                )
                row = dict(
                    rec_id=rec_id,
                    kind="TN",
                    anchor=s,
                    token_sentence=token,
                    conf=float(np.max(conf[s:e])),
                )
                if return_windows:
                    row["window"] = w
                rows.append(row)
                tn_count += 1

    return pd.DataFrame(rows)


def summarize_error_motifs(trace_df, kind="FP", top_n=10):
    df = trace_df[trace_df.kind == kind]
    tokens = (
        df["token_sentence"]
        .dropna()
        .str.split()
        .explode()
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    tokens.columns = ["token", "count"]
    return tokens


def explode_ngram_motifs(trace_df, ngram_range, kind="FP", top_n=10):
    """Count n-gram motifs (matching a FallLM vectorizer's ngram_range) for a given trace kind."""
    df = trace_df[trace_df.kind == kind].dropna(subset=["token_sentence"])
    if df.empty:
        return pd.DataFrame(columns=["token", "count"])
    vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b")
    counts = vec.fit_transform(df["token_sentence"])
    freqs = np.asarray(counts.sum(axis=0)).ravel()
    motif_df = pd.DataFrame({"token": vec.get_feature_names_out(), "count": freqs})
    return motif_df.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)


def ngram_enrichment(trace_df, ngram_range, kind_a="FP", kind_b="TP", top_n=15, eps=1e-3):
    """Rank n-grams by how much more often they occur in `kind_a` windows than `kind_b` windows.

    Frequencies are normalized by the number of windows of each kind, so the
    comparison is fair even when the two groups have very different sizes.
    """
    df_a = trace_df[trace_df.kind == kind_a].dropna(subset=["token_sentence"])
    df_b = trace_df[trace_df.kind == kind_b].dropna(subset=["token_sentence"])
    if df_a.empty or df_b.empty:
        return pd.DataFrame(columns=["token", f"{kind_a}_freq", f"{kind_b}_freq", "enrichment"])

    vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b")
    vec.fit(pd.concat([df_a["token_sentence"], df_b["token_sentence"]]))

    freq_a = np.asarray(vec.transform(df_a["token_sentence"]).sum(axis=0)).ravel() / len(df_a)
    freq_b = np.asarray(vec.transform(df_b["token_sentence"]).sum(axis=0)).ravel() / len(df_b)

    enrichment_df = pd.DataFrame({
        "token": vec.get_feature_names_out(),
        f"{kind_a}_freq": freq_a,
        f"{kind_b}_freq": freq_b,
        "enrichment": (freq_a + eps) / (freq_b + eps),
    })
    enrichment_df = enrichment_df[enrichment_df[f"{kind_a}_freq"] > 0]
    return enrichment_df.sort_values("enrichment", ascending=False).head(top_n).reset_index(drop=True)


def sentence_overlap(trace_df, kind_a="TP", kind_b="FP"):
    """Fraction of `kind_a`/`kind_b` full token sentences that also occur in the other group."""
    sents_a = set(trace_df.loc[trace_df.kind == kind_a, "token_sentence"].dropna())
    sents_b = set(trace_df.loc[trace_df.kind == kind_b, "token_sentence"].dropna())
    shared = sents_a & sents_b
    return {
        f"unique_{kind_a}": len(sents_a),
        f"unique_{kind_b}": len(sents_b),
        "shared": len(shared),
        f"frac_{kind_a}_shared": len(shared) / len(sents_a) if sents_a else 0.0,
        f"frac_{kind_b}_shared": len(shared) / len(sents_b) if sents_b else 0.0,
    }


def resample_df(df, feat_cols, orig_freq, target_freq):

    if orig_freq == target_freq:
        return df

    up = target_freq
    down = orig_freq

    signal = df[feat_cols].values
    resampled = resample_poly(signal, up, down, axis=0)

    # Map event indices instead of resampling labels
    fall_indices = np.where(df["label"] > 0)[0]
    new_falls = map_event_indices(fall_indices, orig_freq, target_freq)

    new_df = pd.DataFrame(resampled, columns=feat_cols)
    new_df["label"] = 0

    new_falls = [i for i in new_falls if i < len(new_df)]
    new_df.loc[new_falls, "label"] = 1

    return new_df


def map_event_indices(events, orig_freq, target_freq):
    ratio = target_freq / orig_freq
    return [int(round(e * ratio)) for e in events]


class WindowZNormalizer(BaseEstimator, TransformerMixin):
    """
    Fast per-window z-normalization.

    Supports:
        (n_windows, window_length)
        (n_windows, n_channels, window_length)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)

        # Univariate case
        if X.ndim == 2:
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True)

        # Multivariate case
        elif X.ndim == 3:
            mean = X.mean(axis=2, keepdims=True)
            std = X.std(axis=2, keepdims=True)

        else:
            raise ValueError("Unsupported input shape for WindowZNormalizer")

        std = np.where(std < 1e-8, 1.0, std)

        return (X - mean) / std


def extract_top_tokens(model, top_n=20):
    feature_names = model.vectorizer_.get_feature_names_out()
    coef = model.classifier_.coef_[0]
    df = pd.DataFrame(
        {"token": feature_names, "weight": coef, "abs_weight": np.abs(coef)}
    )
    df = df.sort_values("abs_weight", ascending=False)
    top_fall = df[df.weight > 0].head(top_n).copy().sort_values("abs_weight", ascending=False)
    top_normal = df[df.weight < 0].head(top_n).copy().sort_values("abs_weight", ascending=False)

    return top_fall, top_normal


def extract_token_windows(model, X, token, max_samples=50):
    sentences = model.tokenizer_.transform(X)

    windows = []
    for i, s in enumerate(sentences):
        if token in s.lower():
            windows.append(X[i])
        if len(windows) >= max_samples:
            break
    return np.array(windows)


def plot_token_motif(model, X, token):
    wins = extract_token_windows(model, X, token)
    if len(wins) == 0:
        print("No windows found for token:", token)
        return
    mean_signal = wins.mean(axis=0)

    plt.figure(figsize=(5, 3))
    # plot all windows in light gray
    plt.plot(wins.T, color="lightgray", alpha=0.5)
    plt.plot(mean_signal)
    plt.title(f"Motif prototype: {token}")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.show()


def filter_impact_tokens(df):
    return df[
        df.token.str.contains(
            "impact_high|impact_med|impact_low|rel_impact_high|rel_impact_med|rel_impact_low",
            regex=True,
        )
    ]

def select_visualization_motifs(shared_motifs, n=3):
    df = shared_motifs.copy()
    # prefer motifs containing impact tokens
    # df = df[df.token.str.contains("impact_high|impact_med", regex=True)]
    # remove extremely long ngrams if present
    df["len"] = df.token.str.split().str.len()
    df = df[(df["len"] >= 4)]
    # rank by weight
    selected = []
    for d in ["FARSEEING", "FallAllD"]:
        df_f = df[df["dataset"] == d].sort_values("abs_weight", ascending=False).head(n)
        selected.append(df_f)
    return pd.concat(selected, ignore_index=True)


def plot_motif_grid(
    model_farseeing,
    model_fallalld,
    motif_df,
    X_farseeing,
    X_fallalld,
    freq=100,
    max_samples=40,
):
    if motif_df.empty:
        raise ValueError("motif_df is empty.")

    if "type" in motif_df.columns:
        fall_rows = motif_df[motif_df["type"].astype(str).str.lower() == "fall"]
        adl_rows = motif_df[motif_df["type"].astype(str).str.lower() == "adl"]
        fall_motif = (
            fall_rows.iloc[0]["token"] if len(fall_rows) else motif_df.iloc[0]["token"]
        )
        adl_motif = (
            adl_rows.iloc[0]["token"]
            if len(adl_rows)
            else motif_df.iloc[min(1, len(motif_df) - 1)]["token"]
        )
    else:
        motifs = motif_df["token"].unique().tolist()
        fall_motif = motifs[0]
        adl_motif = motifs[1] if len(motifs) > 1 else motifs[0]

    col_specs = [
        ("Fall motion pattern", fall_motif),
        ("ADL motion pattern", adl_motif),
    ]
    datasets = [
        ("FARSEEING", model_farseeing, X_farseeing),
        ("FallAllD", model_fallalld, X_fallalld),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(8, 3.5), sharex=True, sharey="row", dpi=300
    )

    for r, (dataset_name, model, X) in enumerate(datasets):
        row_mins, row_maxs = [], []
        for c, (col_title, motif) in enumerate(col_specs):
            ax = axes[r, c]
            wins = extract_token_windows(model, X, motif, max_samples=max_samples)
            if len(wins) > 0:
                t = np.arange(wins.shape[1]) / freq
                for w in wins:
                    ax.plot(t, w, color="0.82", alpha=0.45, linewidth=0.8)

                mean = wins.mean(axis=0)
                ax.plot(
                    t,
                    mean,
                    color="black",
                    linewidth=2.0,
                    label="Mean pattern" if (r == 0 and c == 0) else None,
                )
                if r == 0 and c == 0:
                    ax.plot([], [], color="0.82", linewidth=1.0, label="Acc windows (g)")
                row_mins.append(np.min(wins))
                row_maxs.append(np.max(wins))

            ax.tick_params(labelsize=8)
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(0.8)

            if r == 0:
                ax.set_title(col_title, fontsize=10, y=1.2)
                ax.text(0.5, 1.03, motif, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", family='monospace')

            if c == 0:
                ax.annotate(dataset_name, xy=(-0.12, 0.5), xycoords="axes fraction",
                            ha="right", va="center", fontsize=11, rotation=90)

        if row_mins and row_maxs:
            lo = min(row_mins)
            hi = max(row_maxs)
            pad = 0.06 * (hi - lo + 1e-9)
            axes[r, 0].set_ylim(lo - pad, hi + pad)
            axes[r, 1].set_ylim(lo - pad, hi + pad)

    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    # fig.supylabel("Acceleration (g)", fontsize=12, x=0.11, y=0.6)
    fig.supxlabel("Time (s)", fontsize=12, x=0.57, y=0.14)
    fig.tight_layout(rect=(0.08, 0.07, 1, 1))
    return axes


def plot_motif_grid_topn(
    model,
    motif_df,
    X,
    n=3,
    freq=100,
    max_samples=40,
):
    """
    N×2 motif grid (rows = Top 1..N, cols = Fall / ADL) for a single dataset.
    motif_df must have 'token' and 'type' columns (type values 'Fall' / 'ADL').
    """
    fall_tokens = motif_df[motif_df["type"].str.lower() == "fall"]["token"].tolist()[:n]
    adl_tokens  = motif_df[motif_df["type"].str.lower() == "adl"]["token"].tolist()[:n]
    n_rows = max(len(fall_tokens), len(adl_tokens))

    fig, axes = plt.subplots(
        n_rows, 2, figsize=(8, n_rows * 1.9), sharex=True, dpi=300
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_headers = ["Fall motion pattern", "ADL motion pattern"]

    for r in range(n_rows):
        for c, tokens in enumerate([fall_tokens, adl_tokens]):
            ax = axes[r, c]
            ax.tick_params(labelsize=8)
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(0.8)

            if r == 0:
                ax.set_title(col_headers[c], fontsize=10, y=1.22)

            if r < len(tokens):
                motif = tokens[r]
                ax.text(
                    0.5, 1.03, motif, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", family="monospace",
                )
                wins = extract_token_windows(model, X, motif, max_samples=max_samples)
                if len(wins) > 0:
                    t = np.arange(wins.shape[1]) / freq
                    for w in wins:
                        ax.plot(t, w, color="0.82", alpha=0.45, linewidth=0.8)
                    mean = wins.mean(axis=0)
                    ax.plot(
                        t, mean, color="black", linewidth=2.0,
                        label="Mean pattern" if (r == 0 and c == 0) else None,
                    )
                    if r == 0 and c == 0:
                        ax.plot([], [], color="0.82", linewidth=1.0,
                                label="Representative windows (g)")

            if c == 0:
                ax.annotate(
                    f"Top {r + 1}", xy=(-0.12, 0.5), xycoords="axes fraction",
                    ha="right", va="center", fontsize=10, rotation=90,
                )

    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.supylabel("Acceleration (g)", fontsize=11, x=0.08)
    fig.supxlabel("Time (s)", fontsize=11, y=0.08)
    fig.tight_layout(rect=(0.06, 0.04, 1, 1))
    return axes


def get_accel_columns(df):
    # possible column naming schemes
    schemes = [
        ["ax", "ay", "az"],
        ["Acc AP", "Acc ML", "Acc V"],
        ["acc_x", "acc_y", "acc_z"],
        ["x", "y", "z"],
    ]
    for cols in schemes:
        if all(c in df.columns for c in cols):
            return df[cols].values
    raise ValueError("No valid accelerometer columns found.")


# Define feature engineering logic
def calculate_magnitude(df):
    accel_data = get_accel_columns(df)
    raw_mag = np.sqrt(np.sum(accel_data**2, axis=1))
    return raw_mag / 9.81  # Convert to g-units


def adl_label_extractor(df, p):
    name = os.path.splitext(os.path.basename(p))[0].lower()
    if "ataxia" in name:
        df["label"] = 0
    elif "psp" in name:
        df["label"] = 1
    else:
        df["label"] = 2
    return df


# Create the loader function
load_fall_subject = make_csv_loader(
    feature_cols=["mag"],
    label_col="label",
    new_features={"mag": calculate_magnitude},  #
)

# loader for tri-axial data
load_fall_subject_3d = make_csv_loader(
    feature_cols=["acc_x", "acc_y", "acc_z"], label_col="label"
)

load_adl_subject = make_csv_loader(
    feature_cols=["mag"],
    label_extractor=adl_label_extractor,
    label_col="label",
    new_features={"mag": calculate_magnitude},
)

load_adl_subject_3d = make_csv_loader(
    feature_cols=["Acc AP", "Acc ML", "Acc V"],
    label_col="label",
    new_features={"label": lambda df: 0},
)


def fall_id_extractor(p: str) -> str:
    name = os.path.splitext(os.path.basename(p))[0]
    s = name.split("-")[0].split("_")[1]
    return s


def adl_id_extractor(p: str) -> str:
    name = os.path.splitext(os.path.basename(p))[0]
    s = name.split("_")[0]
    return s


def filter_active_windows(X, y, threshold=1.4):
    """
    X: (N, 3, 1000)
    Filters out windows where the magnitude never exceeds 'threshold'.
    """
    mag_zone = np.sqrt(np.sum(X[:, :, 100:200] ** 2, axis=1)) / 9.81
    # 2. Get the max per window in that zone
    max_vals = mag_zone.max(axis=1)
    active_idx = max_vals >= threshold
    print(f"Filtering: {len(X)} windows -> {np.sum(active_idx)} active windows.")
    return X[active_idx], y[active_idx]
