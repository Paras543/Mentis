from __future__ import annotations
from dataclasses import dataclass,field,asdict
from typing import Any


@dataclass
class Finding:
    """THe class Finding ensures that the it will find the single issue
    Atrributes we are using here that are 
    Check_name: THis is will check the identifier through which the warning has came 
    severity: This will check the check the servity incase of this is info,warning or critical
    message: Human readable descriptions 
    column_names: By which coloumn the warning has arrived 
    details: The details of the scan result will be displayed here 
    suggestion: The list of suggestion that will imorove this code 


    """


    check_name:str
    severity: str
    message:str
    column_names: list[str] = field(default_factory=list)
    details: dict[str,Any] = field(default_factory=dict)
    suggestion: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation of this finding."""
        return asdict(self)
    

@dataclass
class ColumnProfile:
    """
    This will check each and every coloumn 
    Attributes we are considering here is the 
    name: name of the coloumn
    is_constant: whether the column has unique single value
    missing_value: will calculate the null values 
    missing_value_pct: will calculate the percentage of the missing null values 
    memory_usage_byte: How much memory is taken
    unique_count: Number of unique non-null values.
    unique_pct: Fraction of unique values relative to row count.

    """


    name: str
    dtype: str
    role: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    is_constant: bool
    memory_usage_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass 
class ScanResult:
    """
    This provides the entire result we done till now
    """

    


    
