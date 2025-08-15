"""
Advanced Statistical Analysis for VBD Rankings
Provides rigorous statistical methods for combining VBD metrics beyond simple weighted averages
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_vbd_correlations(df: pd.DataFrame, vbd_columns: List[str] = None) -> pd.DataFrame:
    """
    Analyze correlations between VBD methods
    
    Args:
        df: DataFrame with VBD calculations
        vbd_columns: List of VBD column names to analyze
        
    Returns:
        Correlation matrix
    """
    if vbd_columns is None:
        vbd_columns = ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']
    
    # Calculate both Pearson and Spearman correlations
    pearson_corr = df[vbd_columns].corr(method='pearson')
    spearman_corr = df[vbd_columns].corr(method='spearman')
    
    logging.info("VBD Method Correlations:")
    logging.info(f"Pearson:\n{pearson_corr}")
    logging.info(f"Spearman:\n{spearman_corr}")
    
    return pearson_corr, spearman_corr

def calculate_vbd_statistics(df: pd.DataFrame, vbd_columns: List[str] = None) -> Dict:
    """
    Calculate comprehensive statistics for VBD methods
    
    Returns:
        Dictionary with statistical measures
    """
    if vbd_columns is None:
        vbd_columns = ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']
    
    stats_dict = {}
    
    for col in vbd_columns:
        stats_dict[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'var': df[col].var(),
            'skewness': stats.skew(df[col]),
            'kurtosis': stats.kurtosis(df[col]),
            'coefficient_of_variation': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
        }
    
    return stats_dict

def pca_analysis(df: pd.DataFrame, vbd_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Principal Component Analysis of VBD methods
    
    Returns:
        principal_components, explained_variance_ratio, first_pc_score
    """
    if vbd_columns is None:
        vbd_columns = ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']
    
    # Standardize the data
    scaler = StandardScaler()
    vbd_scaled = scaler.fit_transform(df[vbd_columns])
    
    # Apply PCA
    pca = PCA()
    principal_components = pca.fit_transform(vbd_scaled)
    
    # First principal component explains the most variance
    first_pc_variance = pca.explained_variance_ratio_[0]
    
    logging.info(f"PCA Results:")
    logging.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logging.info(f"First PC explains {first_pc_variance:.1%} of variance")
    logging.info(f"Component loadings:\n{pd.DataFrame(pca.components_, columns=vbd_columns)}")
    
    return principal_components, pca.explained_variance_ratio_, first_pc_variance

def variance_weighted_consensus(df: pd.DataFrame, vbd_columns: List[str] = None) -> np.ndarray:
    """
    Weight VBD methods inversely to their variance (more stable = higher weight)
    """
    if vbd_columns is None:
        vbd_columns = ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']
    
    # Calculate inverse variance weights
    variances = df[vbd_columns].var()
    inv_var_weights = 1 / variances
    weights = inv_var_weights / inv_var_weights.sum()
    
    logging.info(f"Inverse variance weights: {weights.to_dict()}")
    
    # Calculate weighted consensus
    consensus = (df[vbd_columns] * weights).sum(axis=1)
    
    return consensus, weights

def information_weighted_consensus(df: pd.DataFrame, vbd_columns: List[str] = None) -> np.ndarray:
    """
    Weight methods based on information content (entropy-based)
    """
    if vbd_columns is None:
        vbd_columns = ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']
    
    weights = {}
    
    for col in vbd_columns:
        # Convert to ranks for entropy calculation
        ranks = rankdata(df[col], method='ordinal')
        
        # Calculate normalized entropy
        prob_dist = np.bincount(ranks) / len(ranks)
        prob_dist = prob_dist[prob_dist > 0]  # Remove zeros
        entropy = -np.sum(prob_dist * np.log2(prob_dist))
        
        # Higher entropy = more information = higher weight
        weights[col] = entropy
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    logging.info(f"Information-based weights: {weights}")
    
    # Calculate weighted consensus
    weight_series = pd.Series(weights)[vbd_columns]
    consensus = (df[vbd_columns] * weight_series).sum(axis=1)
    
    return consensus, weights

def rank_aggregation_borda(df: pd.DataFrame, vbd_columns: List[str] = None) -> np.ndarray:
    """
    Borda count rank aggregation method
    """
    if vbd_columns is None:
        vbd_columns = ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']
    
    n_players = len(df)
    
    # Convert VBD scores to ranks (higher VBD = better rank)
    borda_scores = np.zeros(n_players)
    
    for col in vbd_columns:
        # Rank players (1 = best)
        ranks = rankdata(-df[col], method='ordinal')  # Negative for descending
        # Borda points = n_players - rank + 1
        borda_points = n_players - ranks + 1
        borda_scores += borda_points
    
    return borda_scores

def kemeny_optimal_ranking(df: pd.DataFrame, vbd_columns: List[str] = None) -> np.ndarray:
    """
    Simplified Kemeny optimal ranking (computationally expensive for large datasets)
    Uses median rank as approximation
    """
    if vbd_columns is None:
        vbd_columns = ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']
    
    # Convert to ranks matrix
    ranks_matrix = np.zeros((len(df), len(vbd_columns)))
    
    for i, col in enumerate(vbd_columns):
        ranks_matrix[:, i] = rankdata(-df[col], method='ordinal')  # Higher VBD = better rank
    
    # Kemeny approximation: median rank
    median_ranks = np.median(ranks_matrix, axis=1)
    
    # Convert back to scores (lower rank = higher score)
    kemeny_scores = len(df) - median_ranks + 1
    
    return kemeny_scores

def ensemble_consensus(df: pd.DataFrame, vbd_columns: List[str] = None, methods: List[str] = None) -> pd.DataFrame:
    """
    Calculate multiple consensus methods and analyze their agreement
    """
    if vbd_columns is None:
        vbd_columns = ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']
    
    if methods is None:
        methods = ['pca', 'variance_weighted', 'information_weighted', 'borda', 'kemeny']
    
    results = df.copy()
    
    # Calculate each consensus method
    if 'pca' in methods:
        pca_components, _, _ = pca_analysis(df, vbd_columns)
        results['VBD_PCA'] = pca_components[:, 0]  # First principal component
    
    if 'variance_weighted' in methods:
        var_consensus, var_weights = variance_weighted_consensus(df, vbd_columns)
        results['VBD_VARIANCE_WEIGHTED'] = var_consensus
    
    if 'information_weighted' in methods:
        info_consensus, info_weights = information_weighted_consensus(df, vbd_columns)
        results['VBD_INFO_WEIGHTED'] = info_consensus
    
    if 'borda' in methods:
        borda_scores = rank_aggregation_borda(df, vbd_columns)
        results['VBD_BORDA'] = borda_scores
    
    if 'kemeny' in methods:
        kemeny_scores = kemeny_optimal_ranking(df, vbd_columns)
        results['VBD_KEMENY'] = kemeny_scores
    
    return results

def analyze_consensus_stability(df: pd.DataFrame, consensus_columns: List[str]) -> Dict:
    """
    Analyze how well different consensus methods agree with each other
    """
    stability_metrics = {}
    
    # Calculate rank correlations between consensus methods
    corr_matrix = df[consensus_columns].corr(method='spearman')
    stability_metrics['rank_correlations'] = corr_matrix
    
    # Calculate coefficient of variation for each player across methods
    consensus_cv = df[consensus_columns].std(axis=1) / df[consensus_columns].mean(axis=1).abs()
    stability_metrics['player_stability'] = {
        'mean_cv': consensus_cv.mean(),
        'median_cv': consensus_cv.median(),
        'max_cv': consensus_cv.max(),
        'players_with_high_variance': len(consensus_cv[consensus_cv > 0.5])
    }
    
    # Top 50 player ranking stability
    top_50_indices = df.nlargest(50, consensus_columns[0]).index
    top_50_cv = consensus_cv.loc[top_50_indices]
    stability_metrics['top_50_stability'] = {
        'mean_cv': top_50_cv.mean(),
        'median_cv': top_50_cv.median()
    }
    
    return stability_metrics

def comprehensive_vbd_analysis(df: pd.DataFrame) -> Dict:
    """
    Run complete statistical analysis of VBD methods
    """
    vbd_columns = ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']
    
    logging.info("Starting comprehensive VBD statistical analysis...")
    
    analysis_results = {}
    
    # 1. Basic statistics
    analysis_results['basic_stats'] = calculate_vbd_statistics(df, vbd_columns)
    
    # 2. Correlation analysis
    pearson_corr, spearman_corr = analyze_vbd_correlations(df, vbd_columns)
    analysis_results['correlations'] = {'pearson': pearson_corr, 'spearman': spearman_corr}
    
    # 3. PCA analysis
    pca_components, explained_var, first_pc_var = pca_analysis(df, vbd_columns)
    analysis_results['pca'] = {
        'explained_variance_ratio': explained_var,
        'first_pc_variance': first_pc_var
    }
    
    # 4. Generate ensemble consensus
    consensus_df = ensemble_consensus(df, vbd_columns)
    consensus_columns = ['VBD_PCA', 'VBD_VARIANCE_WEIGHTED', 'VBD_INFO_WEIGHTED', 'VBD_BORDA', 'VBD_KEMENY']
    
    # 5. Stability analysis
    analysis_results['stability'] = analyze_consensus_stability(consensus_df, consensus_columns)
    
    logging.info("VBD statistical analysis completed")
    
    return analysis_results, consensus_df

def recommend_optimal_consensus(analysis_results: Dict) -> Dict:
    """
    Recommend the best consensus method based on statistical analysis
    """
    recommendations = {}
    
    # Check if methods are highly correlated (suggests redundancy)
    avg_correlation = analysis_results['correlations']['spearman'].values[np.triu_indices_from(analysis_results['correlations']['spearman'], k=1)].mean()
    
    if avg_correlation > 0.95:
        recommendations['method'] = 'simple_average'
        recommendations['reasoning'] = f"High correlation ({avg_correlation:.3f}) suggests methods are very similar. Simple average is sufficient."
    elif analysis_results['pca']['first_pc_variance'] > 0.85:
        recommendations['method'] = 'pca'
        recommendations['reasoning'] = f"First PC explains {analysis_results['pca']['first_pc_variance']:.1%} of variance. PCA captures most information."
    elif analysis_results['stability']['player_stability']['mean_cv'] < 0.3:
        recommendations['method'] = 'ensemble_average'
        recommendations['reasoning'] = f"Low variance across methods (CV={analysis_results['stability']['player_stability']['mean_cv']:.2f}). Ensemble average recommended."
    else:
        recommendations['method'] = 'variance_weighted'
        recommendations['reasoning'] = "Moderate disagreement between methods. Variance weighting balances stability and information content."
    
    return recommendations