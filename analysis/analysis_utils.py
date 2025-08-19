import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ExperimentAnalyzer:
    """Utility class for loading and analyzing RL experiments"""
    
    def __init__(self, base_dir: str = '../experiments'):
        self.base_dir = Path(base_dir)
        self.experiments = {}
    
    def load_experiment(self, experiment_folder: str) -> Dict[str, Any]:
        """Load a single experiment's data"""
        exp_path = self.base_dir / experiment_folder
        data_path = exp_path / 'data'
        
        if not exp_path.exists():
            print(f"Warning: Experiment folder {experiment_folder} not found")
            return None
        
        experiment_data = {
            'name': experiment_folder,
            'path': str(exp_path),
            'config': None,
            'training_metrics': None,
            'training_log': None,
            'summary': None
        }
        
        config_file = data_path / 'config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                experiment_data['config'] = json.load(f)
        
        csv_file = data_path / 'training_metrics.csv'
        if csv_file.exists():
            experiment_data['training_metrics'] = pd.read_csv(csv_file)
        
        log_file = data_path / 'training_log.json'
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_data = []
                for line in f:
                    try:
                        log_data.append(json.loads(line.strip()))
                    except:
                        continue
                if log_data:
                    experiment_data['training_log'] = pd.DataFrame(log_data)
        
        summary_file = data_path / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                experiment_data['summary'] = json.load(f)
        
        return experiment_data
    
    def load_experiments(self, experiment_folders: List[str]) -> Dict[str, Any]:
        """Load multiple experiments"""
        loaded_experiments = {}
        
        for folder in experiment_folders:
            print(f"Loading {folder}...")
            exp_data = self.load_experiment(folder)
            if exp_data:
                loaded_experiments[folder] = exp_data
            else:
                print(f"Failed to load")
        
        self.experiments = loaded_experiments
        return loaded_experiments
    
    def get_experiment_info(self) -> pd.DataFrame:
        """Get summary info about loaded experiments"""
        info_data = []
        
        for name, exp in self.experiments.items():
            info = {'experiment': name}
            
            if exp['config']:
                config = exp['config']
                info['environment'] = config.get('env', {}).get('id', 'Unknown')
                info['algorithm'] = config.get('algorithm', 'Unknown')
                
                train_cfg = config.get('train', {})
                info['learning_rate'] = train_cfg.get('learning_rate', 'Unknown')
                info['gamma'] = train_cfg.get('gamma', 'Unknown')
                info['num_episodes'] = train_cfg.get('num_episodes', 'Unknown')
            
            if exp['training_metrics'] is not None:
                metrics = exp['training_metrics']
                info['total_episodes'] = len(metrics)
                if 'reward' in metrics.columns:
                    info['avg_reward'] = metrics['reward'].mean()
                    info['max_reward'] = metrics['reward'].max()
                    info['final_reward'] = metrics['reward'].iloc[-1] if len(metrics) > 0 else None
            
            info_data.append(info)
        
        return pd.DataFrame(info_data)


def plot_training_curves(experiments: Dict[str, Any], 
                        metrics: List[str] = ['reward', 'loss'],
                        figsize: Tuple[int, int] = (15, 10),
                        smooth_window: int = 10):
    """Plot training curves for multiple experiments"""
    
    available_metrics = set()
    for exp in experiments.values():
        if exp['training_metrics'] is not None:
            available_metrics.update(exp['training_metrics'].columns)
    
    available_metrics = [m for m in metrics if m in available_metrics]

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 2, figsize=figsize)
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
    colors = sns.color_palette("husl", len(experiments))
    
    for i, metric in enumerate(available_metrics):
        ax_raw = axes[i, 0]
        ax_smooth = axes[i, 1]
        
        for j, (name, exp) in enumerate(experiments.items()):
            if exp['training_metrics'] is not None and metric in exp['training_metrics'].columns:
                data = exp['training_metrics']
                episodes = range(len(data))
                values = data[metric]
                
                ax_raw.plot(episodes, values, alpha=0.7, color=colors[j], 
                           label=name, linewidth=1)
                
                if len(values) >= smooth_window:
                    smoothed = values.rolling(window=smooth_window, min_periods=1).mean()
                    ax_smooth.plot(episodes, smoothed, color=colors[j], 
                                  label=name, linewidth=2)
        
        ax_raw.set_title(f'{metric.capitalize()} (Raw)')
        ax_raw.set_xlabel('Episode')
        ax_raw.set_ylabel(metric.capitalize())
        ax_raw.legend()
        ax_raw.grid(True, alpha=0.3)
        
        ax_smooth.set_title(f'{metric.capitalize()} (Smoothed, window={smooth_window})')
        ax_smooth.set_xlabel('Episode')
        ax_smooth.set_ylabel(metric.capitalize())
        ax_smooth.legend()
        ax_smooth.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_performance_comparison(experiments: Dict[str, Any]):
    """Create performance comparison charts"""
    
    performance_data = []
    
    for name, exp in experiments.items():
        if exp['training_metrics'] is not None and 'reward' in exp['training_metrics'].columns:
            rewards = exp['training_metrics']['reward']
            
            perf = {
                'experiment': name,
                'mean_reward': rewards.mean(),
                'std_reward': rewards.std(),
                'max_reward': rewards.max(),
                'min_reward': rewards.min(),
                'final_reward': rewards.iloc[-1] if len(rewards) > 0 else 0,
                'episodes': len(rewards)
            }
            
            # Calculate final performance (last 10% of episodes)
            final_10_pct = max(1, len(rewards) // 10)
            perf['final_performance'] = rewards.iloc[-final_10_pct:].mean()
            
            performance_data.append(perf)
    
    if not performance_data:
        print("No reward data found in experiments")
        return None
    
    perf_df = pd.DataFrame(performance_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    ax = axes[0, 0]
    bars = ax.bar(range(len(perf_df)), perf_df['mean_reward'], 
                  yerr=perf_df['std_reward'], capsize=5, alpha=0.7)
    ax.set_title('Mean Reward Comparison')
    ax.set_ylabel('Mean Reward')
    ax.set_xticks(range(len(perf_df)))
    ax.set_xticklabels(perf_df['experiment'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.bar(range(len(perf_df)), perf_df['final_performance'], alpha=0.7)
    ax.set_title('Final Performance (Last 10% Episodes)')
    ax.set_ylabel('Average Reward')
    ax.set_xticks(range(len(perf_df)))
    ax.set_xticklabels(perf_df['experiment'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.bar(range(len(perf_df)), perf_df['max_reward'], alpha=0.7, color='green')
    ax.set_title('Maximum Reward Achieved')
    ax.set_ylabel('Max Reward')
    ax.set_xticks(range(len(perf_df)))
    ax.set_xticklabels(perf_df['experiment'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    cv = (perf_df['std_reward'] / perf_df['mean_reward'].abs()) * 100
    ax.bar(range(len(perf_df)), cv, alpha=0.7, color='orange')
    ax.set_title('Learning Stability (Coeff. of Variation)')
    ax.set_ylabel('CV (%)')
    ax.set_xticks(range(len(perf_df)))
    ax.set_xticklabels(perf_df['experiment'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return perf_df


def analyze_learning_curves(experiments: Dict[str, Any], window_size: int = 50):
    """Analyze learning curve characteristics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = sns.color_palette("husl", len(experiments))
    
    ax = axes[0, 0]
    for i, (name, exp) in enumerate(experiments.items()):
        if exp['training_metrics'] is not None and 'reward' in exp['training_metrics'].columns:
            rewards = exp['training_metrics']['reward']
            episodes = range(len(rewards))
            
            rolling_mean = rewards.rolling(window=window_size, min_periods=1).mean()
            rolling_std = rewards.rolling(window=window_size, min_periods=1).std()
            
            ax.plot(episodes, rolling_mean, color=colors[i], label=name, linewidth=2)
            ax.fill_between(episodes, 
                           rolling_mean - rolling_std,
                           rolling_mean + rolling_std,
                           color=colors[i], alpha=0.2)
    
    ax.set_title(f'Learning Curves with Confidence Bands (window={window_size})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for i, (name, exp) in enumerate(experiments.items()):
        if exp['training_metrics'] is not None and 'reward' in exp['training_metrics'].columns:
            rewards = exp['training_metrics']['reward']
            episodes = range(1, len(rewards) + 1)
            
            cumulative_best = rewards.expanding().max()
            ax.semilogx(episodes, cumulative_best, color=colors[i], label=name, linewidth=2)
    
    ax.set_title('Best Reward Achieved vs Episode (Log Scale)')
    ax.set_xlabel('Episode (log scale)')
    ax.set_ylabel('Best Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for i, (name, exp) in enumerate(experiments.items()):
        if exp['training_metrics'] is not None and 'reward' in exp['training_metrics'].columns:
            rewards = exp['training_metrics']['reward']
            ax.hist(rewards, bins=30, alpha=0.6, label=name, color=colors[i])
    
    ax.set_title('Reward Distribution')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for i, (name, exp) in enumerate(experiments.items()):
        if exp['training_metrics'] is not None and 'reward' in exp['training_metrics'].columns:
            rewards = exp['training_metrics']['reward']
            
            improvement_rate = rewards.rolling(window=window_size, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
            
            episodes = range(len(improvement_rate))
            ax.plot(episodes, improvement_rate, color=colors[i], label=name, linewidth=2)
    
    ax.set_title(f'Learning Rate (Reward Improvement per Episode, window={window_size})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Improvement Rate')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_hyperparameters(experiments: Dict[str, Any]):
    """Analyze the relationship between hyperparameters and performance"""
    
    hyperparam_data = []
    
    for name, exp in experiments.items():
        if exp['config'] is None:
            continue
            
        config = exp['config']
        train_cfg = config.get('train', {})
        model_cfg = config.get('model', {})
        
        hyperparams = {
            'experiment': name,
            'learning_rate': train_cfg.get('learning_rate', None),
            'gamma': train_cfg.get('gamma', None),
            'hidden_dim': model_cfg.get('hidden_dim', None),
            'batch_size': train_cfg.get('batch_size', None),
            'epsilon_decay': train_cfg.get('epsilon_decay', None),
        }
        
        if exp['training_metrics'] is not None and 'reward' in exp['training_metrics'].columns:
            rewards = exp['training_metrics']['reward']
            hyperparams['mean_reward'] = rewards.mean()
            hyperparams['max_reward'] = rewards.max()
            hyperparams['final_performance'] = rewards.iloc[-len(rewards)//10:].mean() if len(rewards) > 10 else rewards.mean()
            hyperparams['stability'] = rewards.std()
        
        hyperparam_data.append(hyperparams)
    
    if not hyperparam_data:
        print("No hyperparameter data available")
        return None
    
    df = pd.DataFrame(hyperparam_data)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'mean_reward' in numeric_cols:
        numeric_cols.remove('mean_reward')
    
    if len(numeric_cols) > 1 and 'mean_reward' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        correlations = []
        for col in numeric_cols:
            if df[col].nunique() > 1:
                corr = df[[col, 'mean_reward']].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations.append((col, corr))
        
        if correlations:
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            params, corr_values = zip(*correlations)
            
            ax = axes[0]
            colors = ['red' if x < 0 else 'green' for x in corr_values]
            bars = ax.barh(params, corr_values, color=colors, alpha=0.7)
            ax.set_title('Hyperparameter Correlation with Mean Reward')
            ax.set_xlabel('Correlation Coefficient')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            for i, (param, corr) in enumerate(correlations):
                ax.text(corr + 0.01 if corr >= 0 else corr - 0.01, i, 
                       f'{corr:.3f}', va='center', 
                       ha='left' if corr >= 0 else 'right')
        
        if correlations:
            best_param = correlations[0][0]
            ax = axes[1]
            
            plot_df = df.dropna(subset=[best_param, 'mean_reward'])
            
            if len(plot_df) > 1:
                scatter = ax.scatter(plot_df[best_param], plot_df['mean_reward'], 
                                   s=100, alpha=0.7, c=range(len(plot_df)), cmap='viridis')
                ax.set_xlabel(best_param.replace('_', ' ').title())
                ax.set_ylabel('Mean Reward')
                ax.set_title(f'Mean Reward vs {best_param.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
                
                for i, row in plot_df.iterrows():
                    ax.annotate(row['experiment'][:10] + '...', 
                               (row[best_param], row['mean_reward']),
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    return df


def generate_statistical_summary(experiments: Dict[str, Any]):
    """Generate comprehensive statistical summary"""
    
    print("Statistical Summary Report")
    print("=" * 80)
    print(f"Total Experiments Analyzed: {len(experiments)}")
    print(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_rewards = []
    experiment_stats = []
    
    for name, exp in experiments.items():
        if exp['training_metrics'] is not None and 'reward' in exp['training_metrics'].columns:
            rewards = exp['training_metrics']['reward']
            all_rewards.extend(rewards.tolist())
            
            stats = {
                'Experiment': name,
                'Episodes': len(rewards),
                'Mean Reward': rewards.mean(),
                'Std Reward': rewards.std(),
                'Min Reward': rewards.min(),
                'Max Reward': rewards.max(),
                'Median Reward': rewards.median(),
                'Q1 Reward': rewards.quantile(0.25),
                'Q3 Reward': rewards.quantile(0.75),
                'Final 10% Mean': rewards.iloc[-len(rewards)//10:].mean() if len(rewards) > 10 else rewards.mean()
            }
            experiment_stats.append(stats)
    
    if experiment_stats:
        stats_df = pd.DataFrame(experiment_stats)
        
        print("Individual Experiment Statistics:")
        print("-" * 80)
        print(stats_df.round(3).to_string(index=False))
        
        print("\nOverall Statistics Across All Experiments:")
        print("-" * 80)
        overall_stats = {
            'Total Episodes': stats_df['Episodes'].sum(),
            'Best Mean Reward': stats_df['Mean Reward'].max(),
            'Worst Mean Reward': stats_df['Mean Reward'].min(),
            'Average Mean Reward': stats_df['Mean Reward'].mean(),
            'Most Stable (Lowest Std)': stats_df.loc[stats_df['Std Reward'].idxmin(), 'Experiment'],
            'Highest Peak': stats_df['Max Reward'].max(),
            'Best Final Performance': stats_df['Final 10% Mean'].max()
        }
        
        for key, value in overall_stats.items():
            print(f"{key}: {value}")
        
        print("\nRecommendations:")
        print("-" * 80)
        
        best_exp = stats_df.loc[stats_df['Final 10% Mean'].idxmax(), 'Experiment']
        print(f"• Best Overall Performance: {best_exp}")
        
        stable_exp = stats_df.loc[stats_df['Std Reward'].idxmin(), 'Experiment']
        print(f"• Most Stable Learning: {stable_exp}")
        
        highest_exp = stats_df.loc[stats_df['Max Reward'].idxmax(), 'Experiment']
        print(f"• Highest Potential: {highest_exp}")
        
        return stats_df
    else:
        print("No reward data available for statistical analysis")
        return None