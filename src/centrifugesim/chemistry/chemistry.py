from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple, Union
import logging
import math
import ast

# Required by the user specification
try:
    from centrifugesim.chemistry.maxwellian_rates import (
        build_rate_table_and_interpolator,
        kelvin_from_eV
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import 'build_rate_table_and_interpolator' from "
        "'centrifugesim.chemistry.maxwellian_rates'. "
        "Ensure centrifugesim is installed and importable."
    ) from e


Number = Union[int, float]
log = logging.getLogger(__name__)


# ----------------------- Safe expression compiler -----------------------

_ALLOWED_FUNCS = {
    'exp': math.exp, 'log': math.log, 'log10': math.log10, 'sqrt': math.sqrt,
    'pow': pow, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
    'asinh': math.asinh, 'acosh': math.acosh, 'atanh': math.atanh,
}

_ALLOWED_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Load,
    ast.Name, ast.Call, ast.Pow, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd, ast.LShift, ast.RShift  # (shifts disallowed below)
}

def _assert_safe_ast(node: ast.AST) -> None:
    """Recursively assert the AST uses only allowed, safe constructs."""
    if type(node) not in _ALLOWED_NODES:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")
    # Disallow bit-shifts (we listed them above just so isinstance checks succeed)
    if isinstance(node, (ast.LShift, ast.RShift)):
        raise ValueError("Bit-shift operators are not allowed.")
    for child in ast.iter_child_nodes(node):
        _assert_safe_ast(child)
    # For calls, only allow math-like functions
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("Only basic math functions are allowed: " + ", ".join(sorted(_ALLOWED_FUNCS)))
    # Names will be validated at runtime

def _compile_rate_expression(expr: str) -> Callable[[float], float]:
    """
    Compile a safe expression of Te (Kelvin) to a callable f(Te_K)->k [m^3/s].
    Accepts '^' as exponent, normalizes to '**'.
    Provides names: Te, Te_K, Te_eV, pi, e and math funcs listed above.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("Empty EXPRESSION.")
    normalized = expr.replace('^', '**')
    tree = ast.parse(normalized, mode='eval')
    _assert_safe_ast(tree)
    code = compile(tree, "<rate_expr>", "eval")

    def f(Te_K: float) -> float:
        # numeric guards
        Te = float(Te_K)
        env = {
            **_ALLOWED_FUNCS,
            'Te': Te,
            'Te_K': Te,
            'Te_eV': Te / 11604.518121,  # 1 eV in K
            'pi': math.pi,
            'e': math.e,
        }
        # Disallow any unexpected names
        for name in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}:
            if name not in env:
                raise ValueError(f"Unknown symbol in EXPRESSION: {name}")
        val = eval(code, {'__builtins__': {}}, env)
        return float(val)
    return f


# ----------------------- Data model -----------------------

@dataclass(frozen=True)
class ReactionRate:
    """
    Container for a single reaction's rate data and interpolator.
    """
    name: str
    path: Path
    rtype: str                       # internal storage (IONIZATION, RECOMBINATION, ELASTIC, ...)
    process: Optional[str]           # human-readable reaction string
    k_grid_m3s: "list[float] | tuple | 'np.ndarray'"
    k_interp: Callable[[float], float]
    meta: dict

    @property
    def type(self) -> str:
        """Alias so callers can use reaction.type, as you requested."""
        return self.rtype


class Chemistry:
    """
    Load Maxwellian-averaged reaction rates (e.g., ionization / recombination / elastic)
    from:
      * cross-section files              -> builds rates via build_rate_table_and_interpolator(...)
      * analytic rate files (expressions)-> evaluates safe expressions

    Parameters
    ----------
    cross_sections_dir : str | Path | None
        Defaults to CWD/'cross_sections'.
    rate_coefficients_dir : str | Path | None
        Directory for analytic rate files. Defaults to CWD/'rate_coefficients'.
    Te_min_eV : float
        Lower bound of electron temperature (eV). Default 0.05 eV.
    Te_max_eV : float
        Upper bound of electron temperature (eV). Default 25.0 eV.
    n_T : int
        Number of temperature grid points. Default 300.
    spacing : {"log","lin"}
        Temperature grid spacing. Default "log".

    Attributes
    ----------
    Te_min_eV, Te_max_eV : float
    Te_min_K,  Te_max_K  : float
    Te_grid_K            : list[float] | np.ndarray
    reactions            : dict[str, ReactionRate]
    """

    def __init__(
        self,
        cross_sections_dir: Optional[Union[str, Path]] = None,
        rate_coefficients_dir: Optional[Union[str, Path]] = None,
        *,
        Te_min_eV: float = 0.05,
        Te_max_eV: float = 25.0,
        n_T: int = 300,
        spacing: str = "log",
    ) -> None:
        # Store user-configurable eV bounds
        self.Te_min_eV: float = float(Te_min_eV)
        self.Te_max_eV: float = float(Te_max_eV)

        # Derived Kelvin bounds
        self.Te_min_K: float = kelvin_from_eV(self.Te_min_eV)
        self.Te_max_K: float = kelvin_from_eV(self.Te_max_eV)

        # Grid construction parameters
        self._n_T: int = int(n_T)
        self._spacing: str = spacing

        # Resolve directories relative to runtime CWD by default
        if cross_sections_dir is None:
            cross_sections_dir = Path.cwd() / "cross_sections"
        if rate_coefficients_dir is None:
            rate_coefficients_dir = Path.cwd() / "rate_coefficients"
        self.cross_sections_dir: Path = Path(cross_sections_dir)
        self.rate_coefficients_dir: Path = Path(rate_coefficients_dir)

        # Placeholders populated during load
        self.Te_grid_K = None  # will become array-like
        self.reactions: Dict[str, ReactionRate] = {}

        self._load_all()

    # ------------------------- Public API ---------------------------------

    def names(self) -> Iterable[str]:
        return self.reactions.keys()

    def __len__(self) -> int:
        return len(self.reactions)

    def __contains__(self, name: str) -> bool:
        return name in self.reactions

    def __getitem__(self, name: str) -> ReactionRate:
        return self.reactions[name]

    def get_interp(self, name: str) -> Callable[[float], float]:
        return self.reactions[name].k_interp

    def rate(self, name: str, Te_K: Number) -> float:
        return float(self.reactions[name].k_interp(float(Te_K)))

    # ------------------------- Internal -----------------------------------

    def _build_te_grid(self):
        """Build a Te grid if none has been set yet."""
        if self.Te_grid_K is not None:
            return
        lo, hi, n = float(self.Te_min_K), float(self.Te_max_K), int(self._n_T)
        if self._spacing == "log":
            if lo <= 0:
                raise ValueError("Te_min_K must be > 0 for log spacing.")
            log_lo, log_hi = math.log(lo), math.log(hi)
            step = (log_hi - log_lo) / (n - 1)
            self.Te_grid_K = [math.exp(log_lo + i * step) for i in range(n)]
        elif self._spacing == "lin":
            step = (hi - lo) / (n - 1)
            self.Te_grid_K = [lo + i * step for i in range(n)]
        else:
            raise ValueError("spacing must be 'log' or 'lin'")

    def _parse_cs_header(self, path: Path) -> tuple[str, Optional[str], dict]:
        """
        Parse the lightweight header in a cross-section .txt file to extract:
          - rtype   : first non-empty line (e.g., IONIZATION)
          - process : line starting with 'PROCESS:'
        Stops at the row of dashes '----'.
        Returns (rtype, process, extra_meta).
        """
        rtype = None
        process = None
        extra: dict = {}
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    if s.startswith('-'):
                        break
                    if rtype is None:
                        rtype = s.upper()
                        continue
                    if s.upper().startswith("PROCESS:"):
                        # keep text before first comma if present
                        proc = s.split(":", 1)[1].strip()
                        process = proc.split(",", 1)[0].strip() or None
                    elif ':' in s:
                        # stash any other header keys
                        key, val = s.split(':', 1)
                        extra[key.strip().upper()] = val.strip()
        except Exception as e:
            log.debug("Header parse failed for %s: %s", path.name, e)
        rtype = rtype or "UNKNOWN"
        return rtype, process, extra

    def _parse_rate_file(self, path: Path) -> dict:
        """
        Parse a key:value analytic rate file. Expected keys:
        TYPE, NAME, PROCESS, UNITS (optional), EXPRESSION (or EXPR).
        """
        data: dict = {}
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    # allow bare EXPRESSION = <expr> too
                    if "=" in line:
                        k, v = line.split("=", 1)
                        data[k.strip().upper()] = v.strip()
                    continue
                k, v = line.split(":", 1)
                data[k.strip().upper()] = v.strip()
        expr = data.get("EXPRESSION") or data.get("EXPR")
        if not expr:
            raise ValueError(f"Missing EXPRESSION in {path.name}")
        rtype = (data.get("TYPE") or "UNKNOWN").upper()
        name = data.get("NAME") or path.stem
        process = data.get("PROCESS")
        units = data.get("UNITS") or "m^3/s"
        return {"name": name, "rtype": rtype, "process": process, "expr": expr, "units": units}

    def _load_all(self) -> None:
        """
        Scan directories and build all reactions.
        Ensures Te_grid_K is identical across all reactions.
        """
        # 1) Cross sections
        if self.cross_sections_dir.exists() and self.cross_sections_dir.is_dir():
            cs_files = sorted(self.cross_sections_dir.glob("*.txt"))
        else:
            cs_files = []

        shared_Te_tuple: Optional[Tuple] = None

        for path in cs_files:
            name = path.stem
            rtype, process, header_meta = self._parse_cs_header(path)
            Te_grid_K, k_grid_m3s, k_interp, meta = build_rate_table_and_interpolator(
                path,
                Te_min_K=self.Te_min_K,
                Te_max_K=self.Te_max_K,
                n_T=self._n_T,
                spacing=self._spacing,
            )

            if self.Te_grid_K is None:
                self.Te_grid_K = Te_grid_K
                try:
                    shared_Te_tuple = tuple(Te_grid_K)
                except TypeError:
                    shared_Te_tuple = None
                log.debug("Set shared Te_grid_K from %s", path.name)
            else:
                # enforce identical grid
                if shared_Te_tuple is not None:
                    try:
                        if tuple(Te_grid_K) != shared_Te_tuple:
                            raise ValueError(
                                f"Te_grid_K mismatch for '{path.name}'. "
                                "All reactions must share the exact same Te grid."
                            )
                    except TypeError:
                        if not self._grids_equal(self.Te_grid_K, Te_grid_K):
                            raise ValueError(
                                f"Te_grid_K mismatch for '{path.name}'. "
                                "All reactions must share the exact same Te grid."
                            )

            meta_all = {**meta, **header_meta, "source": "cross_section"}
            self.reactions[name] = ReactionRate(
                name=name,
                path=path,
                rtype=rtype,
                process=process,
                k_grid_m3s=k_grid_m3s,
                k_interp=k_interp,
                meta=meta_all,
            )

        # 2) Analytic rate coefficients
        if self.rate_coefficients_dir.exists() and self.rate_coefficients_dir.is_dir():
            rc_files = sorted(self.rate_coefficients_dir.glob("*.txt"))
        else:
            rc_files = []

        if not self.reactions and not rc_files:
            raise FileNotFoundError(
                f"No reactions found. Checked: {self.cross_sections_dir} and {self.rate_coefficients_dir}"
            )

        # Ensure we have a Te grid for expressions even if there were no cross sections
        if self.Te_grid_K is None:
            self._build_te_grid()
            shared_Te_tuple = tuple(self.Te_grid_K) if self.Te_grid_K is not None else None

        for path in rc_files:
            info = self._parse_rate_file(path)
            expr_func = _compile_rate_expression(info["expr"])

            # k_interp: direct evaluation
            def k_interp_local(Te_K: float, _f=expr_func) -> float:
                return float(_f(Te_K))

            # k_grid sampled on the shared grid
            k_grid = [k_interp_local(T) for T in self.Te_grid_K]

            name = info["name"]
            if name in self.reactions:
                log.warning("Reaction '%s' already exists; analytic file %s will overwrite.", name, path.name)
            self.reactions[name] = ReactionRate(
                name=name,
                path=path,
                rtype=info["rtype"],
                process=info["process"],
                k_grid_m3s=k_grid,
                k_interp=k_interp_local,
                meta={
                    "source": "analytic_expression",
                    "expr": info["expr"],
                    "units": info["units"],
                },
            )

        if not self.reactions:
            raise RuntimeError("No reactions loaded.")

    @staticmethod
    def _grids_equal(a, b) -> bool:
        try:
            if len(a) != len(b):
                return False
            for x, y in zip(a, b):
                if x != y:
                    return False
            return True
        except Exception:
            return False