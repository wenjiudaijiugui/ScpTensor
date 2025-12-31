from typing import Literal
from .structures import ScpContainer, Assay, ScpMatrix

def _clean_cols():
    pass

def reader(data_path: str, groups: dict[str, str], batch: dict[str, str], software: Literal["DIA-NN", "Spectronaut"]) -> ScpContainer:
    """main logit"""
    ...