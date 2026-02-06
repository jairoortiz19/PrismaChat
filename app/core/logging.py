import sys
from pathlib import Path
from loguru import logger
from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()

    # Remove default logger
    logger.remove()

    # Console logging
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {extra[session_id]} | <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
        filter=lambda record: record["extra"].setdefault("session_id", "no-session"),
    )

    # Create logs directory
    log_path = Path(settings.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File logging - general
    logger.add(
        log_path / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra[session_id]} | {message}",
        level=settings.log_level,
        rotation="00:00",  # Rotate at midnight
        retention="30 days",
        compression="zip",
        filter=lambda record: record["extra"].setdefault("session_id", "no-session"),
    )

    # File logging - errors only
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra[session_id]} | {message}",
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        compression="zip",
        filter=lambda record: record["extra"].setdefault("session_id", "no-session"),
    )

    logger.info("Logging configured successfully")


def get_logger(session_id: str = "no-session"):
    return logger.bind(session_id=session_id)
