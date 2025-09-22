from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Logging class for training metrics and progress."""

    def __init__(self, config: Any) -> None:
        """Initialize logger with configuration."""
        self.config = config
        self.log_dir = Path(config.logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.log_dir / f'run_{timestamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._setup_file_logging()

        if config.logging.tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / 'tensorboard'))
        else:
            self.tb_writer = None

        if config.logging.excel:
            self.excel_path = self.run_dir / 'metrics.xlsx'
            self.metrics_data = []
        else:
            self.excel_path = None
            self.metrics_data = None

        self.txt_metrics_path = self.run_dir / 'metrics.txt'
        self.txt_metrics_file = None
        self._init_txt_metrics()

        self.step = 0
        self.episode = 0

        self.info('Logger initialized')
        self.info(f'Run directory: {self.run_dir}')
        self.info(f'TXT metrics file: {self.txt_metrics_path}')

    def _setup_file_logging(self) -> None:
        """Set up file and console logging."""
        self.logger = logging.getLogger('PPOCarRacing')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_level = getattr(logging, self.config.logging.console_level.upper())
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(self.run_dir / 'training.log')
        file_level = getattr(logging, self.config.logging.file_level.upper())
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

    def _init_txt_metrics(self) -> None:
        """Initialize TXT metrics file with headers."""
        try:
            self.txt_metrics_file = open(self.txt_metrics_path, 'w', encoding='utf-8')
            self.txt_metrics_file.write('# Training Metrics Log\n')
            self.txt_metrics_file.write(f'# Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            self.txt_metrics_file.write(
                '# Format: timestamp | step | episode | metric_name=value | metric_name=value | ...\n'
            )
            self.txt_metrics_file.write('#' + '=' * 80 + '\n')
            self.txt_metrics_file.flush()
        except Exception as e:
            self.warning(f'Failed to initialize TXT metrics file: {e}')
            self.txt_metrics_file = None

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to TensorBoard, Excel, and TXT file."""
        if step is None:
            step = self.step

        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.tb_writer.add_scalar(key, value.item(), step)

        if self.metrics_data is not None:
            metric_entry = {'step': step, 'timestamp': datetime.now()}
            metric_entry.update(metrics)
            self.metrics_data.append(metric_entry)
            if len(self.metrics_data) % 100 == 0:
                self.save_excel_metrics()

        self._log_metrics_to_txt(metrics, step)

    def _log_metrics_to_txt(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to TXT file in real-time."""
        if self.txt_metrics_file is None:
            return

        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            episode = metrics.get('episode', self.episode)

            metric_pairs = []
            for key, value in metrics.items():
                if key not in ['step', 'timestamp', 'episode']:
                    if isinstance(value, float):
                        metric_pairs.append(f'{key}={value:.4f}')
                    elif isinstance(value, torch.Tensor) and value.numel() == 1:
                        metric_pairs.append(f'{key}={value.item():.4f}')
                    else:
                        metric_pairs.append(f'{key}={value}')

            line = f'{timestamp} | step={step} | episode={episode} | {" | ".join(metric_pairs)}\n'
            self.txt_metrics_file.write(line)
            self.txt_metrics_file.flush()

        except Exception as e:
            self.warning(f'Failed to write metrics to TXT file: {e}')

    def log_model_info(self, model: torch.nn.Module) -> None:
        """Log model architecture information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.info(f'Model parameters: {total_params:,} total, {trainable_params:,} trainable')
        with open(self.run_dir / 'model_architecture.txt', 'w') as f:
            f.write(str(model))
            f.write(f'\n\nTotal parameters: {total_params:,}')
            f.write(f'\nTrainable parameters: {trainable_params:,}')

    def log_hyperparameters(self, hparams: dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        if self.tb_writer:
            self.tb_writer.add_hparams(hparams, {})

    def save_excel_metrics(self) -> None:
        """Save metrics to Excel file."""
        if self.metrics_data and self.excel_path:
            try:
                df = pd.DataFrame(self.metrics_data)
                with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Metrics', index=False)
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        summary = df[numeric_cols].describe()
                        summary.to_excel(writer, sheet_name='Summary')
                    episode_metrics = df[df['episode'].notna()] if 'episode' in df.columns else None
                    if episode_metrics is not None and not episode_metrics.empty:
                        episode_metrics.to_excel(writer, sheet_name='Episodes', index=False)
                if len(self.metrics_data) % 1000 == 0:
                    self.info(f'Metrics saved to {self.excel_path} ({len(self.metrics_data)} entries)')
            except Exception as e:
                self.warning(f'Failed to save Excel metrics: {e}')

    def log_system_info(self) -> None:
        """Log system information."""
        import platform
        import psutil

        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': f'{psutil.virtual_memory().total / 1e9:.1f} GB',
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            system_info['cuda_version'] = torch.version.cuda
            system_info['gpu_count'] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                system_info[f'gpu_{i}_name'] = gpu_props.name
                system_info[f'gpu_{i}_memory'] = f'{gpu_props.total_memory / 1e9:.1f} GB'

        self.info('System Information:')
        for key, value in system_info.items():
            self.info(f'  {key}: {value}')

        with open(self.run_dir / 'system_info.txt', 'w') as f:
            for key, value in system_info.items():
                f.write(f'{key}: {value}\n')

    def close(self) -> None:
        """Close logger and save final metrics."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.metrics_data:
            self.save_excel_metrics()
        if self.txt_metrics_file:
            try:
                self.txt_metrics_file.write(f'\n# Training ended: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                self.txt_metrics_file.close()
                self.info(f'TXT metrics saved: {self.txt_metrics_path}')
            except Exception as e:
                self.warning(f'Failed to close TXT metrics file: {e}')
        self.info('Logger closed')

    def __enter__(self) -> Logger:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object | None
    ) -> None:
        self.close()
