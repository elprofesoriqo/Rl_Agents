import logging
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Logging class for training metrics and progress."""
    
    def __init__(self, config):
        """Initialize logger with configuration."""
        self.config = config
        self.log_dir = Path(config.logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_file_logging()
        
        if config.logging.tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        else:
            self.tb_writer = None
        
        if config.logging.excel:
            self.excel_path = self.run_dir / "metrics.xlsx"
            self.metrics_data = []
        else:
            self.excel_path = None
            self.metrics_data = None
        
        self.step = 0
        self.episode = 0
        
        self.info("Logger initialized")
        self.info(f"Run directory: {self.run_dir}")
    
    def _setup_file_logging(self):
        """Set up file and console logging."""
        self.logger = logging.getLogger("PPOCarRacing")
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_level = getattr(logging, self.config.logging.console_level.upper())
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler(self.run_dir / "training.log")
        file_level = getattr(logging, self.config.logging.file_level.upper())
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to TensorBoard and Excel."""
        if step is None:
            step = self.step
        
        # Log to TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
                elif isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        self.tb_writer.add_scalar(key, value.item(), step)
        
        # Log to Excel
        if self.metrics_data is not None:
            metric_entry = {"step": step, "timestamp": datetime.now()}
            metric_entry.update(metrics)
            self.metrics_data.append(metric_entry)
    
    def log_episode(self, episode_metrics: Dict[str, Any]):
        """Log episode-specific metrics."""
        self.episode += 1
        episode_metrics["episode"] = self.episode
        
        self.info(f"Episode {self.episode}: {episode_metrics}")
        self.log_metrics(episode_metrics)
    
    def log_training_step(self, training_metrics: Dict[str, Any]):
        """Log training step metrics."""
        self.step += 1
        self.log_metrics(training_metrics, self.step)
    
    def log_model_info(self, model: torch.nn.Module):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log model architecture to file
        with open(self.run_dir / "model_architecture.txt", "w") as f:
            f.write(str(model))
            f.write(f"\n\nTotal parameters: {total_params:,}")
            f.write(f"\nTrainable parameters: {trainable_params:,}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        if self.tb_writer:
            self.tb_writer.add_hparams(hparams, {})
    
    def save_excel_metrics(self):
        """Save metrics to Excel file."""
        if self.metrics_data and self.excel_path:
            df = pd.DataFrame(self.metrics_data)
            
            # Create Excel writer with multiple sheets
            with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
                # Main metrics sheet
                df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # Summary statistics
                numeric_cols = df.select_dtypes(include=['number']).columns
                summary = df[numeric_cols].describe()
                summary.to_excel(writer, sheet_name='Summary')
                
                # Episode-wise metrics if available
                episode_metrics = df[df['episode'].notna()] if 'episode' in df.columns else None
                if episode_metrics is not None and not episode_metrics.empty:
                    episode_metrics.to_excel(writer, sheet_name='Episodes', index=False)
            
            self.info(f"Metrics saved to {self.excel_path}")
    
    def log_system_info(self):
        """Log system information."""
        import psutil
        import platform
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": f"{psutil.virtual_memory().total / 1e9:.1f} GB",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info["cuda_version"] = torch.version.cuda
            system_info["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                system_info[f"gpu_{i}_name"] = gpu_props.name
                system_info[f"gpu_{i}_memory"] = f"{gpu_props.total_memory / 1e9:.1f} GB"
        
        self.info("System Information:")
        for key, value in system_info.items():
            self.info(f"  {key}: {value}")
        
        # Save to file
        with open(self.run_dir / "system_info.txt", "w") as f:
            for key, value in system_info.items():
                f.write(f"{key}: {value}\n")
    
    def close(self):
        """Close logger and save final metrics."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.metrics_data:
            self.save_excel_metrics()
        
        self.info("Logger closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
