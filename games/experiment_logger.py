import json
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import uuid


class ExperimentLogger:
    """
    Experiment logging system for RL model comparison and visualization.
    
    Provides structured data storage for:
    - Model configuration and hyperparameters
    - Training progress and metrics
    - Evaluation results
    - System information and reproducibility data
    - Statistical summaries for comparison
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.experiment_dir = os.path.join(base_dir, f"{experiment_name}_{self.experiment_id}")
        self.data_dir = os.path.join(self.experiment_dir, "data")
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.plots_dir = os.path.join(self.experiment_dir, "plots")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.metadata = {}
        self.config = {}
        self.training_data = []
        self.evaluation_data = []
        self.system_info = {}
        
        self.metadata_file = os.path.join(self.data_dir, "metadata.json")
        self.config_file = os.path.join(self.data_dir, "config.json")
        self.training_file = os.path.join(self.data_dir, "training_log.json")
        self.evaluation_file = os.path.join(self.data_dir, "evaluation_log.json")
        self.summary_file = os.path.join(self.data_dir, "summary.json")
        self.csv_file = os.path.join(self.data_dir, "training_metrics.csv")
        
        self._init_metadata()
    
    def _init_metadata(self):
        """Initialize experiment metadata"""
        import platform
        import torch
        
        self.metadata = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "status": "running",
            "version": "1.0",
            "git_commit": self._get_git_commit(),
        }
        
        self.system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash for reproducibility"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                 capture_output=True, text=True, cwd=os.getcwd())
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def _create_config_hash(self, config: Dict) -> str:
        """Create unique hash for configuration for comparison"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration and hyperparameters"""
        self.config = config.copy()
        self.config["config_hash"] = self._create_config_hash(config)
        self._save_json(self.config, self.config_file)
    
    def log_episode(self, episode_data: Dict[str, Any]):
        """
        Log episode training data with metrics
        
        Expected fields:
        - episode: int
        - total_steps: int
        - episode_steps: int
        - reward: float
        - loss: float (optional)
        - epsilon: float (optional)
        - additional metrics...
        """
        episode_entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            **episode_data
        }
        
        self.training_data.append(episode_entry)
        
        self._save_training_data()
        self._update_csv_log()
    
    def log_evaluation(self, eval_data: Dict[str, Any]):
        """Log evaluation results"""
        eval_entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            **eval_data
        }
        
        self.evaluation_data.append(eval_entry)
        self._save_json(self.evaluation_data, self.evaluation_file)
    
    def log_checkpoint(self, checkpoint_path: str, episode: int, metrics: Dict[str, Any]):
        """Log checkpoint information with associated metrics"""
        checkpoint_info = {
            "path": checkpoint_path,
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        if "checkpoints" not in self.metadata:
            self.metadata["checkpoints"] = []
        
        self.metadata["checkpoints"].append(checkpoint_info)
        self._save_metadata()
    
    def finalize_experiment(self, final_metrics: Optional[Dict[str, Any]] = None):
        """Finalize experiment and generate summary statistics"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.metadata.update({
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "status": "completed",
            "total_episodes": len(self.training_data),
            "final_metrics": final_metrics or {}
        })
        
        summary = self._generate_summary()
        self._save_json(summary, self.summary_file)
        self._save_metadata()
        
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate experiment summary for comparison"""
        if not self.training_data:
            return {"error": "No training data available"}
        
        df = pd.DataFrame(self.training_data)
        
        reward_stats = {
            "mean": float(df['reward'].mean()),
            "std": float(df['reward'].std()),
            "min": float(df['reward'].min()),
            "max": float(df['reward'].max()),
            "median": float(df['reward'].median()),
            "q25": float(df['reward'].quantile(0.25)),
            "q75": float(df['reward'].quantile(0.75)),
        }
        
        window_size = min(100, len(df) // 5) if len(df) > 20 else len(df)
        df['reward_smooth'] = df['reward'].rolling(window=window_size, min_periods=1).mean()
        
        final_performance = df['reward_smooth'].tail(window_size).mean()
        initial_performance = df['reward_smooth'].head(window_size).mean()
        improvement = final_performance - initial_performance
        
        final_episodes = df.tail(window_size)
        stability = {
            "final_mean": float(final_episodes['reward'].mean()),
            "final_std": float(final_episodes['reward'].std()),
            "coefficient_of_variation": float(final_episodes['reward'].std() / final_episodes['reward'].mean()) if final_episodes['reward'].mean() != 0 else float('inf')
        }
        
        total_steps = df['total_steps'].max() if 'total_steps' in df.columns else 0
        steps_per_episode = df['episode_steps'].mean() if 'episode_steps' in df.columns else 0
        
        loss_stats = {}
        if 'loss' in df.columns and df['loss'].notna().any():
            loss_data = df['loss'].dropna()
            loss_stats = {
                "mean": float(loss_data.mean()),
                "std": float(loss_data.std()),
                "final_mean": float(loss_data.tail(window_size).mean()),
                "convergence_rate": self._calculate_convergence_rate(loss_data)
            }
        
        return {
            "experiment_metadata": self.metadata,
            "system_info": self.system_info,
            "config_hash": self.config.get("config_hash"),
            "training_summary": {
                "total_episodes": len(df),
                "total_steps": int(total_steps),
                "average_steps_per_episode": float(steps_per_episode),
                "training_time_seconds": self.metadata.get("duration_seconds", 0)
            },
            "performance_metrics": {
                "reward_statistics": reward_stats,
                "stability_metrics": stability,
                "improvement": float(improvement),
                "final_performance": float(final_performance),
                "learning_efficiency": float(final_performance / (total_steps / 1000)) if total_steps > 0 else 0,
            },
            "loss_statistics": loss_stats,
            "convergence_analysis": {
                "episodes_to_convergence": self._estimate_convergence_episodes(df),
                "performance_plateau": self._detect_plateau(df['reward_smooth']),
            },
            "comparative_metrics": {
                "area_under_curve": float(np.trapz(df['reward_smooth'])),
                "sample_efficiency": self._calculate_sample_efficiency(df),
                "robustness_score": self._calculate_robustness_score(df),
            }
        }
    
    def _calculate_convergence_rate(self, loss_data: pd.Series) -> float:
        """Calculate loss convergence rate"""
        if len(loss_data) < 10:
            return 0.0
        
        x = np.arange(len(loss_data))
        try:
            log_loss = np.log(loss_data + 1e-8)
            slope, _ = np.polyfit(x, log_loss, 1)
            return float(-slope)
        except:
            return 0.0
    
    def _estimate_convergence_episodes(self, df: pd.DataFrame) -> int:
        """Estimate number of episodes needed for convergence"""
        if len(df) < 50:
            return len(df)
        
        reward_smooth = df['reward_smooth'].values
        window = len(reward_smooth) // 4
        
        for i in range(window, len(reward_smooth) - window):
            before = reward_smooth[i-window:i].mean()
            after = reward_smooth[i:i+window].mean()
            if abs(after - before) / (abs(before) + 1e-8) < 0.1:  # 10% change threshold
                return i
        
        return len(df)
    
    def _detect_plateau(self, reward_smooth: pd.Series) -> Dict[str, Any]:
        """Detect performance plateau"""
        if len(reward_smooth) < 20:
            return {"detected": False}
        
        # Check last 25% of training for plateau
        tail_length = len(reward_smooth) // 4
        tail_data = reward_smooth.tail(tail_length).values
        
        x = np.arange(len(tail_data))
        slope, _ = np.polyfit(x, tail_data, 1)
        
        return {
            "detected": abs(slope) < 0.01,  # Very small slope indicates plateau
            "slope": float(slope),
            "plateau_length": tail_length,
            "plateau_value": float(tail_data.mean())
        }
    
    def _calculate_sample_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate sample efficiency metric"""
        if len(df) < 10:
            return 0.0
        
        total_steps = df['total_steps'].max()
        final_performance = df['reward'].tail(20).mean()
        max_performance = df['reward'].max()
        
        efficiency = (final_performance / max_performance) / (total_steps / 100000) if total_steps > 0 else 0
        return float(min(efficiency, 10.0))  # Cap at reasonable value
    
    def _calculate_robustness_score(self, df: pd.DataFrame) -> float:
        """Calculate robustness score based on performance consistency"""
        if len(df) < 10:
            return 0.0
        
        final_rewards = df['reward'].tail(50).values if len(df) > 50 else df['reward'].values
        mean_reward = final_rewards.mean()
        std_reward = final_rewards.std()
        
        cv = std_reward / (abs(mean_reward) + 1e-8)
        robustness = 1.0 / (1.0 + cv)  # Transform to 0-1 scale
        
        return float(robustness)
    
    def _save_training_data(self):
        """Save training data to JSON file"""
        self._save_json(self.training_data, self.training_file)
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        self._save_json(self.metadata, self.metadata_file)
    
    def _save_json(self, data: Any, filepath: str):
        """Save data to JSON file with pretty formatting"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    
    def _update_csv_log(self):
        """Update CSV log for compatibility and quick analysis"""
        if not self.training_data:
            return
        
        df = pd.DataFrame(self.training_data)
        
        csv_columns = [
            'timestamp', 'episode', 'total_steps', 'episode_steps', 
            'reward', 'loss', 'epsilon', 'best_reward', 'worst_reward',
            'rolling_avg', 'rolling_std'
        ]
        
        existing_columns = [col for col in csv_columns if col in df.columns]
        csv_df = df[existing_columns]
        
        csv_df.to_csv(self.csv_file, index=False)
    
    def get_experiment_path(self) -> str:
        """Get the experiment directory path"""
        return self.experiment_dir
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current experiment summary"""
        return self._generate_summary()


class ExperimentComparator:
    """
    Utility class for comparing multiple experiments and generating visualizations
    """
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = experiments_dir
    
    def load_experiment_summaries(self, experiment_paths: List[str]) -> List[Dict[str, Any]]:
        """Load summary data from multiple experiments"""
        summaries = []
        
        for path in experiment_paths:
            summary_file = os.path.join(path, "data", "summary.json")
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    summary['experiment_path'] = path
                    summaries.append(summary)
        
        return summaries
    
    def create_comparison_table(self, summaries: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a comparison table of key metrics"""
        comparison_data = []
        
        for summary in summaries:
            metadata = summary.get('experiment_metadata', {})
            performance = summary.get('performance_metrics', {})
            training = summary.get('training_summary', {})
            
            row = {
                'experiment_name': metadata.get('experiment_name', 'Unknown'),
                'experiment_id': metadata.get('experiment_id', 'Unknown'),
                'config_hash': summary.get('config_hash', 'Unknown'),
                'total_episodes': training.get('total_episodes', 0),
                'total_steps': training.get('total_steps', 0),
                'training_time_hours': training.get('training_time_seconds', 0) / 3600,
                'final_performance': performance.get('final_performance', 0),
                'mean_reward': performance.get('reward_statistics', {}).get('mean', 0),
                'reward_std': performance.get('reward_statistics', {}).get('std', 0),
                'improvement': performance.get('improvement', 0),
                'learning_efficiency': performance.get('learning_efficiency', 0),
                'robustness_score': summary.get('comparative_metrics', {}).get('robustness_score', 0),
                'sample_efficiency': summary.get('comparative_metrics', {}).get('sample_efficiency', 0),
            }
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def export_comparison_data(self, summaries: List[Dict[str, Any]], output_dir: str):
        """Export comparison data in multiple formats for visualization"""
        os.makedirs(output_dir, exist_ok=True)
        
        comparison_df = self.create_comparison_table(summaries)
        
        comparison_df.to_csv(os.path.join(output_dir, "experiments_comparison.csv"), index=False)
        
        comparison_df.to_json(os.path.join(output_dir, "experiments_comparison.json"), 
                             orient='records', indent=2)
        
        with open(os.path.join(output_dir, "detailed_summaries.json"), 'w') as f:
            json.dump(summaries, f, indent=2, default=str)
        
        agg_stats = {
            'total_experiments': len(summaries),
            'best_performance': comparison_df['final_performance'].max(),
            'average_performance': comparison_df['final_performance'].mean(),
            'most_efficient': comparison_df.loc[comparison_df['learning_efficiency'].idxmax()]['experiment_name'],
            'most_robust': comparison_df.loc[comparison_df['robustness_score'].idxmax()]['experiment_name'],
        }
        
        with open(os.path.join(output_dir, "aggregated_stats.json"), 'w') as f:
            json.dump(agg_stats, f, indent=2)
        
        print(f"Comparison data exported to {output_dir}")
        return comparison_df