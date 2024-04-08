from pathlib import Path
import os

DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_DATA_DIR = "./data"

OUTPUT_DIR = Path(os.environ.get("FEWSHOT_DOMAIN_ADAPT_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
DATA_DIR = Path(os.environ.get("FEWSHOT_DOMAIN_ADAPT_DATA_DIR", DEFAULT_DATA_DIR))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)