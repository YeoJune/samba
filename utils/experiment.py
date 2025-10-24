"""
Experiment utilities for logging, checkpointing, and result tracking
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import wandb


class ExperimentLogger:
    """간단하고 깔끔한 실험 로깅 클래스"""
    
    def __init__(self, exp_name: str, save_dir: str = "experiments", use_wandb: bool = False, 
                 wandb_project: str = "samba", config: Optional[Dict] = None):
        """
        Args:
            exp_name: 실험 이름
            save_dir: 저장 디렉토리
            use_wandb: W&B 사용 여부
            wandb_project: W&B 프로젝트명
            config: 실험 설정 (dict)
        """
        self.exp_name = exp_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(save_dir) / f"{exp_name}_{self.timestamp}"
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_file = self.exp_dir / "train.log"
        self.metrics_file = self.exp_dir / "metrics.json"
        
        # 디렉토리 생성
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Config 저장
        if config:
            with open(self.exp_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
        
        # W&B 초기화
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=f"{exp_name}_{self.timestamp}",
                config=config
            )
        
        # 메트릭 저장소
        self.metrics_history = []
        
        print(f"✓ Experiment initialized: {self.exp_dir}")
    
    def log(self, message: str, print_msg: bool = True):
        """로그 메시지 기록"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        if print_msg:
            print(log_line)
        
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """메트릭 로깅 (로컬 + W&B)"""
        # 메트릭에 prefix 추가
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # 메트릭 저장
        metrics_entry = {"step": step, **metrics}
        self.metrics_history.append(metrics_entry)
        
        # JSON 파일로 저장
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # W&B 로깅
        if self.use_wandb:
            wandb.log({**metrics, "step": step})
    
    def save_checkpoint(self, model, optimizer, epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False, filename: Optional[str] = None):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': self.timestamp
        }
        
        # 파일명 설정
        if filename is None:
            filename = f"checkpoint_epoch{epoch}.pt"
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        self.log(f"✓ Checkpoint saved: {save_path}", print_msg=False)
        
        # Best model 저장
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.log(f"✓ Best model saved: {best_path}")
        
        return save_path
    
    def save_final_results(self, results: Dict[str, Any]):
        """최종 결과 저장"""
        results_file = self.exp_dir / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.log(f"✓ Final results saved: {results_file}")
    
    def finish(self):
        """실험 종료"""
        if self.use_wandb:
            wandb.finish()
        self.log("Experiment finished")
        print(f"\n{'='*80}")
        print(f"Experiment completed: {self.exp_dir}")
        print(f"{'='*80}")


class CheckpointManager:
    """체크포인트 저장 및 관리"""
    
    def __init__(self, checkpoint_dir: str, max_keep: int = 3, save_interval: int = 1):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            max_keep: 최대 보관 개수 (best 제외)
            save_interval: 저장 주기 (epoch)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.save_interval = save_interval
        self.best_metric = float('inf')
        self.checkpoints = []
    
    def should_save(self, epoch: int) -> bool:
        """저장 여부 판단"""
        return (epoch + 1) % self.save_interval == 0
    
    def is_best(self, metric: float, lower_is_better: bool = True) -> bool:
        """Best model 여부 판단"""
        if lower_is_better:
            is_better = metric < self.best_metric
        else:
            is_better = metric > self.best_metric
        
        if is_better:
            self.best_metric = metric
        return is_better
    
    def save(self, model, optimizer, epoch: int, metrics: Dict[str, float], 
             is_best: bool = False):
        """체크포인트 저장 및 관리"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Regular checkpoint 저장
        filename = f"checkpoint_epoch{epoch}.pt"
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        self.checkpoints.append(save_path)
        
        # Best model 저장
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (metric: {self.best_metric:.4f})")
        
        # 오래된 체크포인트 삭제
        self._cleanup()
    
    def _cleanup(self):
        """최대 개수 초과 시 오래된 체크포인트 삭제"""
        if len(self.checkpoints) > self.max_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
    
    def load_checkpoint(self, model, optimizer, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def load_best(self, model, optimizer=None):
        """Best model 로드"""
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best model not found: {best_path}")
        
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Best model loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['metrics']


class MetricsTracker:
    """메트릭 추적 및 통계"""
    
    def __init__(self):
        self.history = {}
        self.current = {}
    
    def update(self, metrics: Dict[str, float]):
        """메트릭 업데이트"""
        self.current = metrics
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_best(self, metric_name: str, lower_is_better: bool = True) -> float:
        """최고 성능 메트릭 반환"""
        if metric_name not in self.history:
            return float('inf') if lower_is_better else float('-inf')
        
        if lower_is_better:
            return min(self.history[metric_name])
        else:
            return max(self.history[metric_name])
    
    def get_current(self, metric_name: str) -> Optional[float]:
        """현재 메트릭 반환"""
        return self.current.get(metric_name)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """메트릭 요약 통계"""
        summary = {}
        for key, values in self.history.items():
            summary[key] = {
                'last': values[-1] if values else None,
                'best': min(values) if values else None,
                'avg': sum(values) / len(values) if values else None
            }
        return summary
