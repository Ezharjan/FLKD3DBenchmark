import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

DEFAULT_OUTPUT_ROOT = Path("./outputs")
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    return value


def _asdict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: _sanitize_value(v) for k, v in obj.items()}
    if hasattr(obj, "_asdict"):
        return {k: _sanitize_value(v) for k, v in obj._asdict().items()}
    if hasattr(obj, "__dict__"):
        return {k: _sanitize_value(v) for k, v in vars(obj).items()}
    return {"value": _sanitize_value(obj)}


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[f"{key}.{inner_key}"] = _sanitize_value(inner_value)
        else:
            flat[key] = _sanitize_value(value)
    return flat


def _build_run_name(log_dir: Optional[str], name_hint: Optional[str]) -> str:
    if log_dir:
        return log_dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if name_hint:
        return f"{name_hint}_{timestamp}"
    return f"run_{timestamp}"


def setup_logger(log_file: Path, logger_name: str, console: bool = True) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


@dataclass
class Experiment:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    metrics_file: Path
    config_file: Path
    logger: logging.Logger
    run_name: str

    def log(self, message: str) -> None:
        self.logger.info(message)

    def log_args(self, args: Any) -> Dict[str, Any]:
        args_dict = _asdict(args)
        self.logger.info("PARAMETER ...")
        self.logger.info(args_dict)
        self.write_config(args_dict)
        return args_dict

    def write_config(self, config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        config_dict = _asdict(config)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        return config_dict

    def record_metrics(self, split: str, **metrics: Any) -> Dict[str, Any]:
        payload = {"timestamp": datetime.utcnow().isoformat() + "Z", "split": split}
        payload.update(_flatten_metrics(metrics))
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return payload


def prepare_experiment(
    subdirs: Union[str, Sequence[str]],
    log_dir: Optional[str] = None,
    name_hint: Optional[str] = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    logger_name_prefix: str = "Model",
    log_filename: str = "train.log",
    console: bool = True,
) -> Experiment:
    if isinstance(subdirs, str):
        subdirs = [subdirs]

    run_name = _build_run_name(log_dir, name_hint)
    run_dir = Path(output_root).joinpath(*subdirs, run_name)
    checkpoints_dir = run_dir.joinpath("checkpoints")
    logs_dir = run_dir.joinpath("logs")
    metrics_file = run_dir.joinpath("metrics.jsonl")
    config_file = run_dir.joinpath("config.json")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir.joinpath(log_filename)
    logger = setup_logger(log_file, f"{logger_name_prefix}-{run_name}", console=console)

    return Experiment(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        metrics_file=metrics_file,
        config_file=config_file,
        logger=logger,
        run_name=run_name,
    )


def load_config(run_dir: Path) -> Dict[str, Any]:
    config_path = Path(run_dir).joinpath("config.json")
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_metrics_line(run_dir: Path, split: str, **metrics: Any) -> Dict[str, Any]:
    metrics_path = Path(run_dir).joinpath("metrics.jsonl")
    payload = {"timestamp": datetime.utcnow().isoformat() + "Z", "split": split}
    payload.update(_flatten_metrics(metrics))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return payload
