import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, pairwise_distances
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from hdbscan.validity import validity_index
from dtaidistance import dtw
import plotly.express as px
from PIL import Image
from pyod.models.iforest import IForest

def plot_time_series_with_rolling(df, x_col, window_length=12, include_median=False, include_std=False):
    columns = df.drop(columns=[x_col]).columns
    nrows = len(columns)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(18, 3 * nrows))

    if nrows == 1:
        axes = [axes]

    for i, column in enumerate(columns):
        sns.lineplot(x=df[x_col], y=df[column], ax=axes[i], color='dodgerblue', label=column)

        rolling_mean = df[column].rolling(window=window_length, center=True).mean()
        sns.lineplot(x=df[x_col], y=rolling_mean, ax=axes[i], label='Moving Mean', color='red')

        if include_median:
            rolling_median = df[column].rolling(window=window_length, center=True).median()
            sns.lineplot(x=df[x_col], y=rolling_median, ax=axes[i], label='Moving Median', color='orange')

        if include_std:
            rolling_std = df[column].rolling(window=window_length, center=True).std()
            sns.lineplot(x=df[x_col], y=rolling_std, ax=axes[i], label='Moving Std Dev', color='green')

        axes[i].legend()
        axes[i].set_title(f'Trend for {column}')

    plt.tight_layout()
    plt.show()
  

def plot_distributions_and_qq(df, x_col, figsize=(18, 20)):
    """
    Generate histogram + KDE and Q-Q plots for each numerical column in the DataFrame,
    excluding the specified x_col.

    Parameters:
    - df: pd.DataFrame
    - x_col: column to exclude (typically a time or ID column)
    - figsize: tuple, size of the overall plot
    """
    columns = df.drop(columns=[x_col]).columns
    nrows = len(columns)

    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)

    # If only one row, make sure axes is 2D
    if nrows == 1:
        axes = axes.reshape(1, 2)

    for i, column in enumerate(columns):
        # --- Histogram with KDE ---
        sns.histplot(data=df, x=column, kde=True, ax=axes[i, 0], color='dodgerblue')
        axes[i, 0].set_title(f'Distribution: {column}', fontsize=10)
        axes[i, 0].set_xlabel(column)
        axes[i, 0].set_ylabel("Frequency")

        # --- Q-Q Plot ---
        probplot(df[column], dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(f'Q-Q Plot: {column}', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_acf_pacf_matrix(df, x_col, lags=10, method='ywm', figsize=(18, 20)):
    """

    Generate ACF and PACF plots for each time series in the DataFrame.
    Parameters:
    - df: pd.DataFrame
    - x_col: column to exclude (typically the time index)
    - lags: number of lags to display
    - method: PACF calculation method (default: 'ywm')
    - figsize: tuple, size of the overall plot
    """
    df_plot = df.drop(columns=[x_col])
    columns = df_plot.columns
    nrows = len(columns)

    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)

    # If only one row, reshape axes
    if nrows == 1:
        axes = axes.reshape(1, 2)

    for i, column in enumerate(columns):
        # --- ACF ---
        plot_acf(
            df_plot[column],
            ax=axes[i, 0],
            lags=lags,
            title=f"ACF: {column}"
        )

        # --- PACF ---
        plot_pacf(
            df_plot[column],
            ax=axes[i, 1],
            lags=lags,
            title=f"PACF: {column}",
            method=method
        )

    plt.tight_layout()
    plt.show()


    
def stationarity_test(df, exclude_col, significance_level=0.05, sort_by='ADF p-value'):
    """
    Perform ADF and KPSS stationarity tests on each time series column in the DataFrame.

    Parameters:
    - df: pd.DataFrame
    - exclude_col: column to exclude (e.g. time index)
    - significance_level: threshold for p-value to declare stationarity
    - sort_by: column to sort the result table by

    Returns:
    - pd.DataFrame with test statistics and stationarity interpretation
    """
    df_test = df.drop(columns=[exclude_col]) if exclude_col in df.columns else df
    results = []

    for col in df_test.columns:
        series = df_test[col]

        # ADF test
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(series.dropna(), autolag='AIC')
        except:
            adf_stat, adf_p = None, None

        # KPSS test
        try:
            kpss_stat, kpss_p, _, _ = kpss(series.dropna(), regression='c', nlags='auto')
        except:
            kpss_stat, kpss_p = None, None

        results.append({
            "Variable": col,
            "ADF p-value": adf_p,
            "ADF stationary?": "✅ Yes" if adf_p is not None and adf_p < significance_level else "❌ No",
            "KPSS p-value": kpss_p,
            "KPSS stationary?": "✅ Yes" if kpss_p is not None and kpss_p > significance_level else "❌ No"
        })

    df_stationarity = pd.DataFrame(results)

    if sort_by in df_stationarity.columns:
        df_stationarity = df_stationarity.sort_values(by=sort_by)

    return df_stationarity

def pairwise_correlation_plot(
    df,
    method='pearson',
    height=1.6,
    sig_level=0.05,
    show_sig_marker=True,
    diag_plot='kde',
    lower_plot='scatter'
):
    """
    Create a pairwise correlation plot with annotated coefficients.

    Parameters:
    - df: pd.DataFrame with only numerical columns
    - method: correlation method ('pearson', 'spearman', 'kendall')
    - height: float, size of each subplot
    - sig_level: significance threshold for annotation (e.g., 0.05)
    - show_sig_marker: whether to append significance marker (e.g., "***")
    - diag_plot: 'kde' or 'hist' for the diagonal plot
    - lower_plot: type of plot in the lower triangle (e.g., 'scatter', 'reg')
    """
    
    # Select appropriate correlation function
    corr_methods = {
        'pearson': pearsonr,
        'spearman': spearmanr,
    }
    corr_func = corr_methods.get(method.lower())
    if corr_func is None:
        raise ValueError(f"Unsupported method: {method}")

    def annotate_corr(x, y, **kws):
        r, p = corr_func(x, y)
        ax = plt.gca()
        sig = "***" if show_sig_marker and p < sig_level else ""
        ax.annotate(f"{r:.3f}{sig}", xy=(0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center', fontsize=12)

    g = sns.PairGrid(df, height=height)
    
    # Map plots
    if lower_plot == 'scatter':
        g.map_lower(sns.scatterplot)
    elif lower_plot == 'reg':
        g.map_lower(sns.regplot)
    
    g.map_upper(annotate_corr)
    
    if diag_plot == 'kde':
        g.map_diag(sns.kdeplot, lw=2)
    elif diag_plot == 'hist':
        g.map_diag(sns.histplot)

    # Set column titles on top row
    for i, col in enumerate(df.columns):
        g.axes[0, i].set_title(col, fontsize=10)

    g.set(xlabel="", ylabel="")  # remove default labels
    plt.tight_layout()
    plt.show()



def tsne_plot(df, cluster_labels, title='t-SNE Visualization', save_path='figure.png'):

    tsne_3d = TSNE(n_components=3, perplexity=30, init='pca', random_state=42)
    tsne_results = tsne_3d.fit_transform(df)
    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    df_tsne['cluster_name'] = cluster_labels

    # Colors
    unique_clusters = df_tsne['cluster_name'].unique()
    palette = sns.color_palette("coolwarm", n_colors=len(unique_clusters))
    cluster_colors = {}
    for i, name in enumerate(unique_clusters):
        if 'anomaly' in str(name).lower():
          cluster_colors[name] = (1.0, 0.0, 0.0)  # rosso
        else:
          cluster_colors[name] = palette[i]
    df_tsne['color'] = df_tsne['cluster_name'].map(cluster_colors)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for name in unique_clusters:
        mask = df_tsne['cluster_name'] == name
        ax.scatter(
            df_tsne.loc[mask, 'TSNE1'],
            df_tsne.loc[mask, 'TSNE2'],
            df_tsne.loc[mask, 'TSNE3'],
            c=[cluster_colors[name]],
            label=name,
            alpha=0.7,
            s=40
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('TSNE1')
    ax.set_ylabel('TSNE2')
    ax.set_zlabel('TSNE3')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def grid_search_dbscan_silhouette(X, metric, eps_values=np.arange(1, 2, 0.01), min_samples_values=range(3, 8)):
    results = []

    VI = None
    if metric == 'mahalanobis':
        cov = np.cov(X.T)
        cov += np.eye(cov.shape[0]) * 1e-6  # regolarizzazione
        VI = np.linalg.inv(cov)

    for eps in eps_values:
        for min_samples in min_samples_values:
            if metric == 'mahalanobis':
                dbscan = DBSCAN(eps=eps, min_samples=min_samples,
                                metric='mahalanobis', metric_params={'VI': VI})
            else:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

            labels = dbscan.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters >= 2 and len(set(labels)) < len(X):
                sil_score = silhouette_score(X, labels)
            else:
                sil_score = -1

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'silhouette_score': sil_score
            })

    results_df = pd.DataFrame(results).sort_values(by='silhouette_score', ascending=False)
    best_valid = results_df[results_df['silhouette_score'] > -1].head(1)

    if not best_valid.empty:
        best = best_valid.iloc[0]
        print(f"Migliori parametri: eps={best['eps']}, min_samples={best['min_samples']}, silhouette={best['silhouette_score']:.3f}")
    else:
        print("Nessun clustering valido trovato (almeno 2 cluster).")

    return results_df


def elbow_method(X, min_samples_list=range(3, 9), metric='euclidean', VI=None):

    if metric == 'mahalanobis':
        if VI is None:
            cov = np.cov(X.T)
            cov += np.eye(cov.shape[0]) * 1e-6  # Regolarization
            VI = np.linalg.inv(cov)

    plt.figure(figsize=(12, 6))

    for k in min_samples_list:
        kwargs = {'metric': metric}
        if metric == 'mahalanobis':
            kwargs['metric_params'] = {'VI': VI}

        neigh = NearestNeighbors(n_neighbors=k, **kwargs)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)

        kth_distances = np.sort(distances[:, k - 1])
        plt.plot(kth_distances, label=f'min_samples = {k}')

    plt.title("Elbow Method")
    plt.xlabel("Punti ordinati per distanza")
    plt.ylabel("Distanza al k-esimo vicino più vicino")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def grid_dbscan_dbcv(X, metric, eps_range=np.arange(0.015, 0.030, 0.001), min_samples_range=range(3, 8)):
    results = []

    if metric == 'mahalanobis':
        VI = np.linalg.inv(np.cov(X.T))
        distance_matrix = pairwise_distances(X, metric='mahalanobis', VI=VI)
    else:
        VI = None
        distance_matrix = pairwise_distances(X, metric=metric)

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = dbscan.fit_predict(distance_matrix)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_outliers = np.sum(labels == -1)

            # Calcolo del DBCV solo se ci sono almeno 2 cluster
            dbcv = validity_index(distance_matrix, labels) if n_clusters >= 2 else -1

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'dbcv': dbcv,
                'n_clusters': n_clusters,
                'n_outliers': n_outliers
            })

    results_df = pd.DataFrame(results).sort_values(by='dbcv', ascending=False)
    best = results_df.iloc[0]
    print(f"Migliori parametri trovati: eps={best['eps']}, min_samples={best['min_samples']}, dbcv={best['dbcv']:.3f}")
    return results_df


def tsne_plot_3d(X, cluster_labels, title='t-SNE 3D Visualization'):

    tsne = TSNE(n_components=3, perplexity=30, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    df_tsne['cluster_name'] = cluster_labels

    # Plot 3D
    fig = px.scatter_3d(
        df_tsne,
        x='TSNE1',
        y='TSNE2',
        z='TSNE3',
        color='cluster_name',
        title=title,
        opacity=0.7
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        legend_title_text='Cluster',
        scene=dict(
            xaxis_title='TSNE1',
            yaxis_title='TSNE2',
            zaxis_title='TSNE3'
        )
    )

    fig.show()



def train_iforest(data, features, *, contamination = 0.05, n_estimators = 300, random_state = 42):
    model = IForest(
        n_estimators = n_estimators,
        max_samples = "auto",
        contamination = contamination,
        random_state = random_state,
    )
    model.fit(data[features])

    labels = pd.Series(model.predict(data[features]), index = data.index, name = "outlier_iforest")
    scores = pd.Series(model.decision_function(data[features]), index = data.index, name = "anomaly_score")

    return model, labels, scores


def plot_anomaly_distribution(scores: pd.Series):
    plt.figure(figsize=(6, 3))
    sns.histplot(scores, binwidth = 0.005, color = "steelblue")
    plt.axvline(0, color = "red", linestyle = "--", label = "Anomaly threshold")
    plt.title("Distribuzione dell'anomaly score")
    plt.xlabel("Anomaly score")
    plt.ylabel("Frequenza")
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_feature_importance(model, features):
    importances = model.feature_importances_
    order = importances.argsort()[::-1]

    plt.figure(figsize=(6, 3))
    plt.barh(
        [features[i] for i in order],
        importances[order],
        color = "steelblue")
    plt.xlabel("Importanza")
    plt.title("Feature importance - Isolation Forest")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()



def plot_time_series_outliers(
    data,
    features,
    outlier_mask):
    n_rows = len(features)
    _, ax = plt.subplots(nrows=n_rows, ncols=1, figsize=(15, 2 * n_rows))

    for i, col in enumerate(features):
        sns.lineplot(x=data.index, y=data[col], ax=ax[i], color="dodgerblue")
        # punti outlier
        ax[i].scatter(
            data.index[outlier_mask == 1],
            data.loc[outlier_mask == 1, col],
            color="red",
            s=20,
            label="Outlier",
        )
        ax[i].set_title(f"Serie temporale: {col}")
        ax[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()




def tsne_embedding(
    data,
    features,
    *,
    n_components = 3,
    perplexity = 50,
    learning_rate = "auto",
    max_iter = 1_000,
    random_state = 42):

    X = data[features].values
    tsne = TSNE(
        n_components = n_components,
        perplexity = perplexity,
        learning_rate = learning_rate,
        init = "random",
        max_iter = max_iter,
        random_state = random_state,
    )
    return tsne.fit_transform(X)



def plot_tsne_3d(
    embedding,
    outlier_mask,
    *,
    title = "t-SNE (Isolation Forest outliers)",
    point_size = 14,
    alpha = 0.85):

    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = "3d")

    colors = ["crimson" if o else "steelblue" for o in outlier_mask]
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c = colors,
        s = point_size,
        alpha = alpha)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    plt.title(title)
    plt.tight_layout()
    plt.show()



def run_dbscan(
    data,
    features,
    *,
    eps = 0.021,
    min_samples = 5):

    dbscan = DBSCAN(eps = eps, min_samples = min_samples)
    labels = pd.Series(
        dbscan.fit_predict(data[features]),
        index = data.index,
        name = "cluster")

    is_anomaly = (labels == -1).astype(int).rename("anomaly_DBscan_euclidean")

    label_mapping = {lbl: f"cluster_{lbl}" for lbl in labels.unique() if lbl != -1}
    label_mapping[-1] = "anomaly_DBscan_euclidean"
    name_lbl = labels.map(label_mapping).rename("cluster_name")

    return labels, is_anomaly, name_lbl



def compare_anomaly_score_dist(
    scores_iforest,
    outlier_mask_dbscan,
    *,
    binwidth = 0.005,
    title = "Anomaly score distribution (DBSCAN - euclidean)"):

    plt.figure(figsize = (6, 3))
    sns.histplot(scores_iforest, binwidth = binwidth, color = "steelblue")
    sns.histplot(
        scores_iforest[outlier_mask_dbscan == 1],
        binwidth = binwidth,
        color = "crimson",
        label = "Outlier DBSCAN",
    )
    plt.axvline(0, color = "red", linestyle = "--", label = "Threshold I-Forest = 0")
    plt.title(title)
    plt.xlabel("Anomaly score (Isolation Forest decision_function)")
    plt.ylabel("Frequence")
    plt.legend()
    plt.tight_layout()
    plt.show()



def run_dbscan_mahalanobis(data_scaled, features, eps=1.597, min_samples=4, index=None):

    if isinstance(data_scaled, pd.DataFrame):
        X   = data_scaled[features].values
        idx = data_scaled.index
    else:  # ndarray
        X   = data_scaled
        idx = index if index is not None else pd.RangeIndex(len(X))

    VI = np.linalg.inv(np.cov(X.T))

    dbscan = DBSCAN(
        eps = eps,
        min_samples = min_samples,
        metric = "mahalanobis",
        metric_params = {"VI": VI},
    )
    lbl = dbscan.fit_predict(X)

    labels = pd.Series(lbl, index = idx, name = "cluster_mah")
    is_anom = (labels == -1).astype(int).rename("anomaly_mah")

    mapping = {l: f"cluster_{l}" for l in labels.unique() if l != -1}
    mapping[-1] = "anomaly_mah"
    name_lbl = labels.map(mapping).rename("cluster_name_mah")

    return labels, is_anom, name_lbl