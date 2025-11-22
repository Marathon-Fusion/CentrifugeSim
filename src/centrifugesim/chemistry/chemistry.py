from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple, Union, Set, List
import logging
import math
import ast
import re

import numpy as np

from centrifugesim import constants
from centrifugesim.geometry.geometry import Geometry
from centrifugesim.fluids.neutral_fluid import NeutralFluidContainer
from centrifugesim.fluids.electron_fluid import ElectronFluidContainer
from centrifugesim.chemistry.chemistry_helper import compute_dTe_inelastic

try:
    from centrifugesim.chemistry.maxwellian_rates import (
        build_rate_table_and_interpolator,
        kelvin_from_eV,
        eval_rate_map_numba
    )
except Exception as e:
    raise ImportError(
        "Failed to import 'centrifugesim.chemistry.maxwellian_rates'. "
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
class SpontaneousRate:
    """
    Container for a spontaneous emission transition.
    Process: Reactant -> Product + hν (photon ignored for fluid density)
    Rate: dn/dt = -A * n
    """
    reactant: str
    product: str
    A: float  # Einstein coefficient [1/s]

@dataclass(frozen=True)
class ReactionRate:
    """
    Container for a single reaction's rate data and interpolator.
    """
    name: str
    path: Path
    rtype: str                       # internal storage (IONIZATION, RECOMBINATION, EXCITATION, ELASTIC, ...)
    process: [str]                  # human-readable reaction string
    k_grid_m3s: "list[float] | tuple | 'np.ndarray'"
    k_interp: Callable[[float], float]
    meta: dict

    # --- Pre-computed kinetic parameters ---
    
    # Exponents for rate calc: R = k * (ne ** ne_order) * Product(n_s ** order for s in neutral_reactants)
    ne_order: int
    
    # Map of {species_name: order} for neutral reactants (e.g. {'H(1S)': 1})
    neutral_reactants: Dict[str, int]
    
    # Net change per reaction event: 
    # 'ne': +1/-1 (net electron change)
    # 'species': {'H(1S)': -1, 'H(2P)': +1, ...} (net neutral species change)
    net_changes_ne: int
    net_changes_species: Dict[str, int]

    delta_E_eV: float = 0.0          # optional energy change per reaction (eV)

    @property
    def type(self) -> str:
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
        spontaneous_dir: Optional[Union[str, Path]] = None,
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
        if spontaneous_dir is None:
            spontaneous_dir = Path.cwd() / "spontaneous_emission"

        self.cross_sections_dir: Path = Path(cross_sections_dir)
        self.rate_coefficients_dir: Path = Path(rate_coefficients_dir)
        self.spontaneous_dir: Path = Path(spontaneous_dir)

        # Placeholders populated during load
        self.Te_grid_K = None  # will become array-like
        self.reactions: Dict[str, ReactionRate] = {}
        self.spontaneous_reactions: List[SpontaneousRate] = []

        # Unique list of species (e.g. "H", "H+", "Ar") excluding "E", "hν"
        self.species_list: List[str] = []

        self._load_all()
        self._load_spontaneous()

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
        found_species = set()
        
        # 1) Cross sections
        cs_files = sorted(self.cross_sections_dir.glob("*.txt")) if self.cross_sections_dir.is_dir() else []
        shared_Te_tuple = None

        for path in cs_files:
            name = path.stem
            rtype, process, header_meta = self._parse_cs_header(path)
            
            if process:
                found_species.update(self._extract_species_names(process))

            Te_grid_K, k_grid_m3s, k_interp, meta = build_rate_table_and_interpolator(
                path, Te_min_K=self.Te_min_K, Te_max_K=self.Te_max_K,
                n_T=self._n_T, spacing=self._spacing,
            )
            delta_E_eV = self._extract_delta_E_eV_from_file(path)

            if self.Te_grid_K is None:
                self.Te_grid_K = Te_grid_K
                try:
                    shared_Te_tuple = tuple(Te_grid_K)
                except TypeError: shared_Te_tuple = None
            else:
                if shared_Te_tuple and tuple(Te_grid_K) != shared_Te_tuple:
                     raise ValueError(f"Te_grid_K mismatch for '{path.name}'.")

            ne_ord, n_react, dn_e, dn_s = self._analyze_stoichiometry(process)
            
            meta_all = {**meta, **header_meta, "source": "cross_section"}
            self.reactions[name] = ReactionRate(
                name=name, path=path, rtype=rtype, process=process,
                k_grid_m3s=k_grid_m3s, k_interp=k_interp, meta=meta_all,
                ne_order=ne_ord, neutral_reactants=n_react,
                net_changes_ne=dn_e, net_changes_species=dn_s,
                delta_E_eV=delta_E_eV
            )

        # 2) Analytic rate coefficients
        rc_files = sorted(self.rate_coefficients_dir.glob("*.txt")) if self.rate_coefficients_dir.is_dir() else []
        if not self.reactions and not rc_files:
             raise FileNotFoundError(f"No reactions found.")
        
        if self.Te_grid_K is None:
            self._build_te_grid()

        for path in rc_files:
            info = self._parse_rate_file(path)
            if info["process"]:
                found_species.update(self._extract_species_names(info["process"]))

            expr_func = _compile_rate_expression(info["expr"])
            def k_interp_local(Te_K: float, _f=expr_func) -> float: return float(_f(Te_K))
            k_grid = [k_interp_local(T) for T in self.Te_grid_K]

            name = info["name"]
            ne_ord, n_react, dn_e, dn_s = self._analyze_stoichiometry(info["process"])
            
            delta_E_eV = self._extract_delta_E_eV_from_file(path)

            self.reactions[name] = ReactionRate(
                name=name, path=path, rtype=info["rtype"], process=info["process"],
                k_grid_m3s=k_grid, k_interp=k_interp_local,
                meta={"source": "analytic", "expr": info["expr"], "units": info["units"]},
                ne_order=ne_ord, neutral_reactants=n_react,
                net_changes_ne=dn_e, net_changes_species=dn_s,
                delta_E_eV=delta_E_eV
            )
        
        self.species_list = sorted(list(found_species))

    def _load_spontaneous(self) -> None:
        """
        Loads spontaneous emission rates from 'spontaneous_emission.txt'
        Format: H(3S) -> H(2P), A = 6.32e6
        """
        filename = "spontaneous_emission.txt"
        path = self.spontaneous_dir / filename
        
        if not path.exists():
            log.info(f"No spontaneous emission file found at {path}")
            return

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith("#") or s.startswith("["):
                         # Skip comments or source tags like 
                        continue
                    
                    # Expected: "H(3S) -> H(2P), A = 6.32e6"
                    # 1. Split reaction and A coeff
                    if "," not in s:
                        continue
                    
                    reaction_part, meta_part = s.split(",", 1)
                    
                    # 2. Parse A coefficient
                    # Look for A = ...
                    m = re.search(r"A\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)", meta_part)
                    if not m:
                        continue
                    A_val = float(m.group(1))
                    
                    # 3. Parse Reactant -> Product
                    if "->" not in reaction_part:
                        continue
                    
                    lhs, rhs = reaction_part.split("->", 1)
                    reactant = lhs.strip()
                    product = rhs.strip()
                    
                    self.spontaneous_reactions.append(
                        SpontaneousRate(reactant=reactant, product=product, A=A_val)
                    )
                    
                    # Add to species list if not present (ignoring photons if any)
                    if reactant not in self.species_list: self.species_list.append(reactant)
                    if product not in self.species_list: self.species_list.append(product)

        except Exception as e:
            log.error(f"Failed to load spontaneous emission file: {e}")

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

    @staticmethod
    def _extract_delta_E_eV_from_file(path: Path) -> float:
        """
        Return ionization threshold energy (eV) if present in a 'PARAM' line like:
            'PARAM.:  E = 13.6 eV, complete set'
        Otherwise return 0.0. Elastic/recombination entries typically lack 'E =' and will yield 0.0.
        """
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = raw.strip()
                    # Focus on the PARAM line only
                    if line.upper().startswith("PARAM"):
                        m = re.search(
                            r"\bE\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)\s*eV\b",
                            line,
                        )
                        if m:
                            return float(m.group(1))
        except Exception:
            # If file read fails, just return 0.0
            pass
            
        # Fallback if no matching PARAM line was found in the file
        return 0.0

    def _extract_species_names(self, process: str) -> Set[str]:
        """Return set of unique species names (ignoring e, hv, and IONS)."""
        if not process: return set()
        ignore = {'e', 'e-', 'electron', 'hv', 'hν'}
        species = set()
        
        tokens = re.split(r"->|,|\s+\+\s+", process)
        
        for tok in tokens:
            t = tok.strip()
            if not t: continue
            
            # 1. Ignore electrons and photons
            if t.lower() in ignore:
                continue
                
            # 2. Ignore Ions (any token containing '+')
            # This handles "H+" or "Ar+" so they aren't added as neutral species.
            if '+' in t:
                continue
                
            species.add(t)
        return species

    def _analyze_stoichiometry(self, process: str) -> Tuple[int, Dict[str, int], int, Dict[str, int]]:
        """
        Returns:
          ne_order (int): Power of ne in rate eq.
          neutral_reactants (dict): {species_name: power} for neutrals in rate eq.
          net_changes_ne (int): Change in electron count per event.
          net_changes_species (dict): Change in count per species per event.
        """
        if not process or not isinstance(process, str):
             return 0, {}, 0, {}

        try:
            left_str, right_str = process.split("->", 1)
        except ValueError:
            left_str, right_str = process, ""
        
        def _parse_side(side_str):
            # We separate 'strict electrons' (e) from 'ions' (+)
            counts = {'e_strict': 0, 'ions': 0, 'species': {}}
            tokens = re.split(r"\s+\+\s+|\s*,\s*", side_str.strip())
            for tok in tokens:
                t = tok.strip()
                if not t: continue
                t_lower = t.lower()
                
                if t_lower in {"e", "e-", "electron"}:
                    counts['e_strict'] += 1
                elif "hv" in t_lower or "hν" in t_lower:
                    pass 
                else:
                    if '+' in t:
                        # It is an ion (e.g. H+). 
                        # It affects Kinetics (LHS) but is not a 'neutral species' 
                        # and is not an 'electron' for stoichiometry.
                        counts['ions'] += 1
                    else:
                        # It is a neutral species (H(1S), etc)
                        c = counts['species'].get(t, 0)
                        counts['species'][t] = c + 1
            return counts

        left = _parse_side(left_str)
        right = _parse_side(right_str)

        # 1. Kinetic Order (LHS only)
        # Rate ~ [e]^(e_strict + ions) * [neutrals]
        ne_order = left['e_strict'] + left['ions']
        neutral_reactants = left['species']

        # 2. Stoichiometric Change (Right - Left)
        # Strictly track electrons for the electron fluid update
        delta_ne = right['e_strict'] - left['e_strict']
        
        delta_species = {}
        all_species = set(left['species']) | set(right['species'])
        for s in all_species:
            change = right['species'].get(s, 0) - left['species'].get(s, 0)
            if change != 0:
                delta_species[s] = change

        return ne_order, neutral_reactants, delta_ne, delta_species

    def compute_collision_frequencies(
        self,
        geom: Geometry,
        electron_fluid: ElectronFluidContainer,
        neutral_fluid: NeutralFluidContainer
    ) -> Tuple[float, Dict[str, Dict[str, np.ndarray]]]:
        """
        Pre-calculates rate coefficients and collision frequencies.
        Checks BOTH Electron generation frequency AND Neutral depletion frequency 
        to determine the safe timestep.
        
        Returns:
            max_freq (float): Maximum frequency (1/s) for dt limiting.
            precomputed_data (dict): Cached maps for the solver loop.
        """
        if not hasattr(neutral_fluid, 'str_states_list') or not hasattr(neutral_fluid, 'list_nn_grid'):
            raise AttributeError("NeutralFluidContainer must have str_states_list and list_nn_grid.")

        mask = geom.mask.astype(np.int8)
        Te = electron_fluid.Te_grid
        ne = electron_fluid.ne_grid  # <--- Need ne for depletion check
        
        state_map = {name: i for i, name in enumerate(neutral_fluid.str_states_list)}
        
        max_freq = 0.0
        precomputed_data = {}

        relevant_reactions = [
            r for r in self.reactions.values()
            if r.rtype in {"IONIZATION", "RECOMBINATION", "EXCITATION"} 
        ]

        for rxn in relevant_reactions:
            # 1. Evaluate Rate Coefficient k(Te)
            k_map = eval_rate_map_numba(
                mask.astype(np.int64),
                np.asarray(self.Te_grid_K),
                np.asarray(rxn.k_grid_m3s),
                np.asarray(Te),
            )

            # 2. Build Neutral Reactants Term (term_nn)
            term_nn = np.ones_like(Te) 
            valid_reaction = True
            for spec, order in rxn.neutral_reactants.items():
                if spec not in state_map:
                    valid_reaction = False
                    break
                idx = state_map[spec]
                n_s = neutral_fluid.list_nn_grid[idx]
                if order == 1: term_nn *= n_s
                else:          term_nn *= (n_s ** order)
            
            if not valid_reaction:
                zeros = np.zeros_like(k_map)
                precomputed_data[rxn.name] = {'freq': zeros, 'k_map': zeros, 'term_nn': zeros}
                continue

            # 3. Build Electron Term (term_ne) for Depletion Check
            # (How aggressive is this reaction toward the neutral?)
            if rxn.ne_order == 0:   term_ne = np.ones_like(ne)
            elif rxn.ne_order == 1: term_ne = ne
            else:                   term_ne = ne ** rxn.ne_order

            # 4. Calculate Frequencies
            
            # Freq A: Electron Fluid Evolution timescale (e.g. ionization freq)
            # This drives how fast ne changes.
            freq_electron_ev = k_map * term_nn

            # Freq B: Neutral Depletion timescale
            # This drives how fast the neutral state dies.
            # Only relevant if the reaction actually consumes a neutral reactant.
            freq_neutral_loss = np.zeros_like(k_map)
            if rxn.neutral_reactants:
                freq_neutral_loss = k_map * term_ne

            # 5. Update Max Frequency (Check BOTH)
            # We mask outside regions to avoid artifacts
            
            # Check Electron Limit
            current_max_e = np.max(np.where(mask > 0, freq_electron_ev, 0.0))
            if current_max_e > max_freq:
                max_freq = current_max_e
                
            # Check Neutral Depletion Limit (CRITICAL for excited states)
            current_max_n = np.max(np.where(mask > 0, freq_neutral_loss, 0.0))
            if current_max_n > max_freq:
                max_freq = current_max_n

            # 6. Store Data (We strictly save freq_electron_ev for the Rate equation R = freq * ne)
            precomputed_data[rxn.name] = {
                'freq': freq_electron_ev, 
                'k_map': k_map,
                'term_nn': term_nn
            }

        return max_freq, precomputed_data

    def do_electron_inelastic_collisions_safe_but_slow(
        self,
        geom: Geometry,
        electron_fluid: ElectronFluidContainer,
        neutral_fluid: NeutralFluidContainer,
        dt: float,
        precomputed_data: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        """
        Unified generalized inelastic collision loop.
        Uses precomputed 'freq' (k * term_nn) to optimize performance.
        """
        mask = geom.mask.astype(np.int8)
        ne = electron_fluid.ne_grid
        Te = electron_fluid.Te_grid

        if not hasattr(neutral_fluid, 'str_states_list') or not hasattr(neutral_fluid, 'list_nn_grid'):
             raise AttributeError("NeutralFluidContainer must have str_states_list and list_nn_grid.")

        state_map = {name: i for i, name in enumerate(neutral_fluid.str_states_list)}

        delta_ne_total = np.zeros_like(ne, dtype=float)
        delta_Te_total = np.zeros_like(Te, dtype=float)
        delta_nn_states: Dict[int, np.ndarray] = {}
        
        relevant_reactions = [
            r for r in self.reactions.values()
            if r.rtype in {"IONIZATION", "RECOMBINATION", "EXCITATION"} 
        ]

        for rxn in relevant_reactions:
            if rxn.name not in precomputed_data:
                continue
            
            # 1. Retrieve precomputed frequency
            # freq = k(Te) * term_nn
            freq_map = precomputed_data[rxn.name]['freq']

            # 2. Build Electron Term (term_ne)
            if rxn.ne_order == 0:
                term_ne = 1.0
            elif rxn.ne_order == 1:
                term_ne = ne
            elif rxn.ne_order == 2:
                term_ne = ne * ne
            elif rxn.ne_order == 3:
                term_ne = ne * ne * ne
            else:
                term_ne = ne ** rxn.ne_order

            # 3. Calculate Total Rate R
            # R = (k * term_nn) * term_ne
            R = freq_map * term_ne
            events = R * dt 

            # 4. Apply Changes
            
            # A) Electrons
            if rxn.net_changes_ne != 0:
                delta_ne_total += float(rxn.net_changes_ne) * events
            
            # B) Neutral Species
            for spec, change in rxn.net_changes_species.items():
                if spec not in state_map:
                    continue
                
                idx = state_map[spec]
                if idx not in delta_nn_states:
                    delta_nn_states[idx] = np.zeros_like(ne, dtype=float)
                
                delta_nn_states[idx] += float(change) * events

            # C) Energy
            if rxn.delta_E_eV != 0.0:
                dE_J = float(constants.kb * kelvin_from_eV(float(rxn.delta_E_eV)))

                delta_Te_total += compute_dTe_inelastic(
                    mask.astype(np.int8),
                    ne,
                    R,
                    -float(dE_J), 
                    float(dt),
                    float(electron_fluid.ne_floor),
                )

        # Commit fields
        electron_fluid.Te_grid += delta_Te_total
        electron_fluid.ne_grid += delta_ne_total
        
        # Update individual state grids
        for idx, delta_grid in delta_nn_states.items():
            neutral_fluid.list_nn_grid[idx] += delta_grid
            np.maximum(neutral_fluid.list_nn_grid[idx], 0.0, out=neutral_fluid.list_nn_grid[idx])

        # Finalize Neutral Fluid
        neutral_fluid.compute_nn_grid_from_states()
        neutral_fluid.update_rho()
        neutral_fluid.update_p()
        
    def do_electron_inelastic_collisions(
        self,
        geom: Geometry,
        electron_fluid: ElectronFluidContainer,
        neutral_fluid: NeutralFluidContainer,
        dt: float,
        precomputed_data: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        """
        Unified generalized inelastic collision loop with ADAPTIVE SUB-CYCLING.
        
        Logic:
          1. Uses global 'dt' (externally set, potentially large).
          2. For each reaction, calculates a local stiffness (max_freq).
          3. Determines required sub-steps N to maintain stability (freq * dt_sub < 0.1).
          4. Loops N times, re-evaluating density terms to ensure strict mass conservation
             and depletion physics, while keeping k(Te) frozen for performance.
        """
        mask = geom.mask.astype(np.int8)
        
        # Direct access to mutable arrays for in-place updates (Conservation requirement)
        ne = electron_fluid.ne_grid
        Te = electron_fluid.Te_grid # Used for Energy update only (rates are frozen)
        
        if not hasattr(neutral_fluid, 'str_states_list') or not hasattr(neutral_fluid, 'list_nn_grid'):
             raise AttributeError("NeutralFluidContainer must have str_states_list and list_nn_grid.")

        state_map = {name: i for i, name in enumerate(neutral_fluid.str_states_list)}
        
        relevant_reactions = [
            r for r in self.reactions.values()
            if r.rtype in {"IONIZATION", "RECOMBINATION", "EXCITATION"} 
        ]

        for rxn in relevant_reactions:
            if rxn.name not in precomputed_data:
                continue
            
            # 1. Retrieve frozen Rate Coefficient
            # We trust k(Te) varies slowly compared to density depletion
            data = precomputed_data[rxn.name]
            k_map = data['k_map'] 
            
            # 2. Determine Stiffness (Number of Sub-steps)
            # We need to estimate the CURRENT max frequency to pick N.
            # We can use the pre-calculated 'freq' (k*nn) and 'term_nn' from the precompute step
            # as a good-enough estimate for the start of the timestep.
            
            # Retrieve cached frequency (Ionization/Generation timescale)
            freq_gen_est = data['freq'] 
            
            # Estimate Depletion frequency (Loss timescale) roughly
            # freq_loss ~ k_map * ne (approx). 
            # We do a quick check to see if we need to account for fast neutral loss.
            freq_loss_est = np.zeros_like(freq_gen_est)
            if rxn.neutral_reactants:
                 # Quick estimate using current ne
                 # Optimization: if ne_order==1, use ne. Else use ne^order.
                 if rxn.ne_order == 1:
                     term_ne_est = ne
                 else:
                     term_ne_est = ne ** rxn.ne_order
                 freq_loss_est = k_map * term_ne_est

            # Find maximum freq across the domain
            max_freq_gen = np.max(np.where(mask > 0, freq_gen_est, 0.0))
            max_freq_loss = np.max(np.where(mask > 0, freq_loss_est, 0.0))
            
            local_max_freq = max(max_freq_gen, max_freq_loss)
            
            # Determine N steps. Target: freq * dt_sub ~ 0.1
            if local_max_freq > 1e-20:
                # If dt * freq = 10, we need 100 steps.
                n_steps = int(math.ceil(dt * local_max_freq / 0.1))
            else:
                n_steps = 1
            
            # Force at least 1 step
            if n_steps < 1: n_steps = 1
            
            dt_sub = dt / float(n_steps)

            # 3. Sub-cycling Loop
            for _ in range(n_steps):
                
                # --- A. Re-evaluate Density Terms (CRITICAL for Mass Conservation) ---
                
                # Electron Term
                if rxn.ne_order == 0:
                    term_ne = 1.0
                elif rxn.ne_order == 1:
                    term_ne = ne
                elif rxn.ne_order == 2:
                    term_ne = ne * ne
                else:
                    term_ne = ne ** rxn.ne_order

                # Neutral Term
                term_nn = 1.0
                valid_reaction = True
                
                # We iterate neutral reactants and pull the *current* grid values
                for spec, order in rxn.neutral_reactants.items():
                    # spec is guaranteed in state_map due to precompute check, but we check for safety
                    idx = state_map[spec]
                    n_s = neutral_fluid.list_nn_grid[idx]
                    
                    if order == 1:
                        term_nn = term_nn * n_s
                    else:
                        term_nn = term_nn * (n_s ** order)

                # Rate [events / m^3 / s]
                # Note: k_map is FROZEN, but terms are FRESH.
                R = k_map * term_ne * term_nn
                
                events = R * dt_sub

                # --- B. Apply Changes Immediately (In-Place) ---
                
                # 1. Electrons
                if rxn.net_changes_ne != 0:
                    # Direct update to fluid array
                    ne += float(rxn.net_changes_ne) * events
                
                # 2. Neutrals
                for spec, change in rxn.net_changes_species.items():
                    idx = state_map[spec]
                    neutral_fluid.list_nn_grid[idx] += float(change) * events
                    
                    # Safety Clamp: Because we sub-cycle, we track depletion well,
                    # but fp precision can still cause -1e-20.
                    # We clamp ONLY the species that changed.
                    if change < 0:
                        np.maximum(neutral_fluid.list_nn_grid[idx], 0.0, out=neutral_fluid.list_nn_grid[idx])

                # 3. Energy
                if rxn.delta_E_eV != 0.0:
                    dE_J = float(constants.kb * kelvin_from_eV(float(rxn.delta_E_eV)))
                    
                    # We update Te immediately. 
                    # Note: compute_dTe_inelastic usually returns a delta.
                    # We add it to the main Te grid immediately.
                    dTe = compute_dTe_inelastic(
                        mask.astype(np.int8),
                        ne,
                        R,
                        -float(dE_J), 
                        float(dt_sub), # Use sub-step dt
                        float(electron_fluid.ne_floor),
                    )
                    Te += dTe

        # Finalize Neutral Fluid (Macroscopic moments) after all reactions are done
        neutral_fluid.compute_nn_grid_from_states()
        neutral_fluid.update_rho()
        neutral_fluid.update_p()


    def do_electron_inelastic_collisions_exponential_integrator(
        self,
        geom: Geometry,
        electron_fluid: ElectronFluidContainer,
        neutral_fluid: NeutralFluidContainer,
        dt: float,
        precomputed_data: Dict[str, Dict[str, np.ndarray]],
        accumulate_ionization: bool = False,
        accumulate_recombination: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Unified generalized inelastic collision loop with ADAPTIVE SUB-CYCLING.
        Includes PROBABILISTIC (Exponential) correction for depletion stability.
        
        Returns:
            If accumulate_* flags are True, returns (delta_ni_ionization, delta_ni_recombination).
            Otherwise returns None.
        """
        mask = geom.mask.astype(np.int8)
        
        # Direct access to mutable arrays
        ne = electron_fluid.ne_grid
        Te = electron_fluid.Te_grid
        
        # Initialize Accumulators (if requested)
        # We allocate them regardless to keep logic simple, or check flags to save RAM.
        # Given this is a heavy loop, let's only allocate if needed.
        delta_ni_ionization = None
        delta_ni_recombination = None
        
        if accumulate_ionization:
            delta_ni_ionization = np.zeros_like(ne, dtype=float)
        if accumulate_recombination:
            delta_ni_recombination = np.zeros_like(ne, dtype=float)
        
        if not hasattr(neutral_fluid, 'str_states_list') or not hasattr(neutral_fluid, 'list_nn_grid'):
             raise AttributeError("NeutralFluidContainer must have str_states_list and list_nn_grid.")

        state_map = {name: i for i, name in enumerate(neutral_fluid.str_states_list)}
        
        relevant_reactions = [
            r for r in self.reactions.values()
            if r.rtype in {"IONIZATION", "RECOMBINATION", "EXCITATION"} 
        ]

        for rxn in relevant_reactions:
            if rxn.name not in precomputed_data:
                continue
            
            # 1. Retrieve frozen Rate Coefficient
            data = precomputed_data[rxn.name]
            k_map = data['k_map']
            
            # 2. Estimate Stiffness for N steps
            freq_gen_est = data['freq'] 
            freq_loss_est = np.zeros_like(freq_gen_est)
            
            if rxn.neutral_reactants:
                 if rxn.ne_order == 1:
                     term_ne_est = ne
                 else:
                     term_ne_est = ne ** rxn.ne_order
                 freq_loss_est = k_map * term_ne_est

            local_max_freq = max(
                np.max(np.where(mask > 0, freq_gen_est, 0.0)),
                np.max(np.where(mask > 0, freq_loss_est, 0.0))
            )
            
            # Target 0.2 probability per step
            if local_max_freq > 1e-20:
                n_steps = int(math.ceil(dt * local_max_freq / 0.2))
            else:
                n_steps = 1
            if n_steps < 1: n_steps = 1
            
            dt_sub = dt / float(n_steps)

            # Identify Depletion Species
            depletion_spec = None
            depletion_stoich = 0.0
            
            for spec in rxn.neutral_reactants:
                net_change = rxn.net_changes_species.get(spec, 0)
                if net_change < 0:
                    depletion_spec = spec
                    depletion_stoich = abs(net_change)
                    break 

            # 3. Sub-cycling Loop
            for _ in range(n_steps):
                
                # --- A. Re-evaluate Density Terms ---
                if rxn.ne_order == 0:   term_ne = 1.0
                elif rxn.ne_order == 1: term_ne = ne
                elif rxn.ne_order == 2: term_ne = ne * ne
                else:                   term_ne = ne ** rxn.ne_order

                term_nn = 1.0
                depletion_n_grid = None
                
                for spec, order in rxn.neutral_reactants.items():
                    idx = state_map[spec]
                    n_s = neutral_fluid.list_nn_grid[idx]
                    
                    if spec == depletion_spec:
                        depletion_n_grid = n_s 
                    
                    if order == 1: term_nn = term_nn * n_s
                    else:          term_nn = term_nn * (n_s ** order)

                # Linear Rate [events / m^3 / s]
                R = k_map * term_ne * term_nn
                
                # --- B. Calculate Events (Linear vs Exponential) ---
                if depletion_spec and depletion_n_grid is not None:
                    # PROBABILISTIC UPDATE
                    valid_mask = depletion_n_grid > 1e-30
                    
                    freq_loss = np.zeros_like(R)
                    np.divide(R, depletion_n_grid, out=freq_loss, where=valid_mask)
                    
                    decay_rate = freq_loss * depletion_stoich
                    prob = 1.0 - np.exp(-decay_rate * dt_sub)
                    
                    n_consumed = depletion_n_grid * prob
                    events = n_consumed / depletion_stoich
                    
                else:
                    # Standard Linear Update
                    events = R * dt_sub


                # --- C. Apply Changes Immediately ---
                
                # 1. Electrons & Accumulation
                if rxn.net_changes_ne != 0:
                    change_ne = float(rxn.net_changes_ne) * events
                    ne += change_ne
                    
                    # --- ACCUMULATION LOGIC ---
                    if accumulate_ionization and rxn.rtype == "IONIZATION":
                        # change_ne is positive for ionization
                        delta_ni_ionization += change_ne
                    
                    elif accumulate_recombination and rxn.rtype == "RECOMBINATION":
                        # change_ne is negative for recombination. 
                        # We usually want the MAGNITUDE for the particle weigher.
                        delta_ni_recombination += np.abs(change_ne)
                
                # 2. Neutrals
                for spec, change in rxn.net_changes_species.items():
                    idx = state_map[spec]
                    neutral_fluid.list_nn_grid[idx] += float(change) * events
                    if change < 0:
                        np.maximum(neutral_fluid.list_nn_grid[idx], 0.0, out=neutral_fluid.list_nn_grid[idx])

                # 3. Energy
                if rxn.delta_E_eV != 0.0:
                    dE_J = float(constants.kb * kelvin_from_eV(float(rxn.delta_E_eV)))
                    R_effective = events / dt_sub
                    dTe = compute_dTe_inelastic(
                         mask.astype(np.int8),
                         ne,
                         R_effective,
                         -float(dE_J),
                         float(dt_sub),
                         float(electron_fluid.ne_floor)
                    )
                    Te += dTe

        # Finalize
        neutral_fluid.compute_nn_grid_from_states()
        neutral_fluid.update_rho()
        neutral_fluid.update_p()
        
        # Return accumulators if they exist
        if delta_ni_ionization is not None or delta_ni_recombination is not None:
            return delta_ni_ionization, delta_ni_recombination
        return None
        

    def do_spontaneous_emission(
        self,
        geom: Geometry,
        neutral_fluid: NeutralFluidContainer,
        dt: float,
        ) -> None:
        """
        Updates neutral fluid states based on spontaneous emission (Einstein A coefficients).
        Uses exponential decay formula to ensure stability if A*dt > 1.
        """
        if not self.spontaneous_reactions:
            return

        if not hasattr(neutral_fluid, 'str_states_list') or not hasattr(neutral_fluid, 'list_nn_grid'):
             raise AttributeError("NeutralFluidContainer must have str_states_list and list_nn_grid.")

        state_map = {name: i for i, name in enumerate(neutral_fluid.str_states_list)}
        
        # Accumulate changes to apply all at once (Operator Splitting)
        # This prevents order-dependence (e.g. A->B and B->C in same step)
        delta_nn_states: Dict[int, np.ndarray] = {}

        for rxn in self.spontaneous_reactions:
            # Check if both species exist in the simulation
            if rxn.reactant not in state_map or rxn.product not in state_map:
                continue
            
            idx_r = state_map[rxn.reactant]
            idx_p = state_map[rxn.product]
            
            n_reactant = neutral_fluid.list_nn_grid[idx_r]
            
            # Physics: Exponential decay probability P = 1 - exp(-A*dt)
            # This prevents n_reactant from going negative if A is large.
            decay_factor = 1.0 - math.exp(-rxn.A * dt)
            
            # Density changing state
            delta_n = n_reactant * decay_factor
            
            # Initialize buffers if needed
            if idx_r not in delta_nn_states:
                delta_nn_states[idx_r] = np.zeros_like(n_reactant)
            if idx_p not in delta_nn_states:
                delta_nn_states[idx_p] = np.zeros_like(n_reactant)
            
            # Apply conservation
            delta_nn_states[idx_r] -= delta_n
            delta_nn_states[idx_p] += delta_n

        # Apply updates
        for idx, delta in delta_nn_states.items():
            neutral_fluid.list_nn_grid[idx] += delta
            
        # Recompute macroscopic moments if any changes occurred
        if delta_nn_states:
            neutral_fluid.compute_nn_grid_from_states()
            neutral_fluid.update_rho()
            neutral_fluid.update_p()