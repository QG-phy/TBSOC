from pydantic import BaseModel
from typing import List, Optional

class FitConfig(BaseModel):
    posfile: str = 'POSCAR'
    winfile: str = 'wannier90.win'
    hrfile: str = 'wannier90_hr.dat' # Changed default to match expected
    kpfile: str = 'KPOINTS'
    eigfile: str = 'EIGENVAL'
    Efermi: Optional[float] = None
    weight_sigma: float = 2.0
    lambdas: List[float]

class FitStatus(BaseModel):
    status: str # "idle", "running", "completed", "error"
    progress: float
    message: str
    result: Optional[dict] = None

class TBBandsRequest(BaseModel):
    lambdas: List[float]
