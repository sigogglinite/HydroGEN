"""
hbond_core.py - calculations behind HydroGEN (hbond_gui.py)

  * CIF reading (cell, symmetry, atom sites with occupancy, _chemical_formula_sum)
  * local environments (all neighbours out to 6 A, distances, angles)
  * bond-valence analysis
      - Gagne & Hawthorne (2015) parameters for cation-O bonds,
        Brese & O'Keeffe (1991) for other anions (values in bv_params.txt)
      - oxidation states from charge balance of _chemical_formula_sum, then
        best BV fit per site; V4+/V5+ estimate after Cooper et al. (2019)
      - neighbour contributions weighted by the neighbour's site occupancy
        (never by the atom's own occupancy)
      - bonds below the sum cutoff (default 0.06 vu) but above the list cutoff
        (default 0.03 vu) are listed but not summed; per-bond overrides allowed
  * O classification (water / hydroxyl / acceptor classes) from the BV sum
  * inferred H-bond candidates for water / hydroxyl donors (forbidden cone,
    A-D-C pass angle, graded A-D-C penalties)
  * hypothetical H positions (D-H 0.918 A, H-D-H 108 deg, each H nearest its acceptor)
  * BV sums including H two ways: GRG (Gagne & Hawthorne 2015 H-O parameters) and
    Ferraris & Ivaldi (1988) from D...A
  * H...H contacts between different donor O atoms (red < 2.0 A, moderate < 2.1 A)
  * whole-structure scheme search (2 H-bonds per water, 1 per hydroxyl) ranked by
    BV compatibility; a site far from its ideal BVS is penalised steeply (power 4)
  * <name>_hypothetical[N].cif output with the chosen H atoms
  * geometric H-bond search when H atoms are present in the CIF

Every tunable number lives in the Settings dataclass below.
"""

import math
import os
import re
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations, product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _default_param_file():
    """bv_params.txt next to the script - or, in the .exe, next to the .exe
    (so it can be edited) with the copy bundled inside the .exe as fallback."""
    places = []
    if getattr(sys, "frozen", False):
        places.append(os.path.dirname(os.path.abspath(sys.executable)))
        places.append(getattr(sys, "_MEIPASS", HERE))
    places.append(HERE)
    for d in places:
        p = os.path.join(d, "bv_params.txt")
        if os.path.exists(p):
            return p
    return os.path.join(places[0], "bv_params.txt")


DEFAULT_PARAM_FILE = _default_param_file()


# ============================================================================
# Settings
# ============================================================================

@dataclass
class Settings:
    # --- environments / bond valence
    env_radius: float = 6.0          # A, neighbours listed for every site
    min_bond_length: float = 1.0     # A, shorter contacts to non-H atoms are ignored
    sum_cutoff: float = 0.06         # vu, bonds >= this are summed
    list_cutoff: float = 0.03        # vu, bonds >= this are listed (not summed if < sum_cutoff)
    # --- O classification by BV sum WITHOUT bonds to H (vu)
    water_min: float = 0.10
    water_max: float = 0.70
    hydroxyl_max: float = 1.40
    probable_max: float = 1.85
    possible_max: float = 1.95
    # N classes from the BV sum without H (N3-): NH4+ below nh4_max, NH3 below nh3_max, NH2 below nh2_max
    nh4_max: float = 0.35
    nh3_max: float = 1.25
    nh2_max: float = 2.25
    nh4_cation_min: float = 3.0      # A; ammonium N and a partly occupied cation closer than this exclude each other
    # --- inferred H bonds: candidate tests
    da_min: float = 2.4              # A, donor-acceptor search shell
    da_max: float = 3.4
    cone_angle: float = 70.0         # deg, full vertex angle of the forbidden cone D->C
    adc_pass_angle: float = 60.0     # deg, candidate fails if any A-D-C is not above this
    # --- A-D-C angle penalty (applied in scoring, not pass/fail)
    adc_soft_angle: float = 90.0     # below this: -adc_soft_w per degree
    adc_hard_angle: float = 80.0     # below this additionally: -adc_hard_w * (degrees below)^2
    adc_soft_w: float = 1.0
    adc_hard_w: float = 2.0
    adc_min_occ: float = 0.5         # cation positions occupied less than this do not block H (A-D-C)
    # --- hypothetical H geometry
    dh_length: float = 0.918         # A, D-H distance used for the GRG bond valences of hypothetical H
    geom_dh_length: float = 0.82     # A, D-H distance at which hypothetical H are PLACED (H...H, cone,
                                     #    H...cation tests, tables, CIF) - X-ray-like (cf. DFIX 0.82);
                                     #    best agreement with located H in tests (0.78-0.92 tried)
    split_acceptors: bool = True     # an H may be split over two alternative (disordered) acceptor positions
    grg_norm_located: bool = True   # GRG valences of LOCATED H use dh_length along their D-H direction
                                     # (X-ray D-H ~0.8 A would give O-H valences >> 1)
    hdh_angle: float = 108.0         # deg, H1-D-H2 angle for water
    # --- H...A bond valence, Ferraris & Ivaldi (1988): s = (D...A / d0)^-n + c
    fi_d0: float = 2.17
    fi_n: float = 8.2
    fi_c: float = 0.06
    # --- scoring
    pair_min_angle: float = 90.0     # deg, A1-D-A2: no penalty at/above this ...
    pair_hard_angle: float = 70.0    # ... sliding (quadratic) to pair_soft_penalty at this angle ...
    pair_soft_penalty: float = -100.0
    pair_angle_penalty: float = -1000.0   # ... and this (growing by itself per 10 deg) below pair_hard_angle
    pair_max_angle: float = 145.0    # deg, A1-D-A2: no penalty at/below this ...
    pair_max_hard: float = 160.0     # ... sliding to pair_soft_penalty here, then pair_angle_penalty (+ per 10 deg)
    h_flex: float = 0.0              # A; optional: the H...H PENALTY uses d + h_flex per hypothetical H
                                     # (flags, counts and impossibility always use the real distance)
    bv_scale: float = 0.10           # vu; a site costs (|BVS - target| outside band / scale) ** power
    bv_power: float = 4.0
    occ_weight: bool = True          # each anion's BV cost x its site occupancy (low occupancy: less certain)
    mix_balance: bool = False        # mixed-valence ('mixed, best BV fit') sites: fit with overall charge balance
    nh4_gr: bool = True              # NH4+ bonded as a whole (García-Rodríguez et al. 2000); its H geometry only
    sub_require_link: bool = True    # subschemes: invalid when the network links fewer directions than it could
    score_base: float = 1000.0      # every total starts here: one major (-1000) violation makes it negative
    w_bv: float = 5.0              # weight of the BV score in the total (high: F&I balance dominates)
    donor_band: float = 0.15         # vu; F&I target band for water (0 +- band) and hydroxyl (1 +- band)
    group_band: float = 0.6          # vu; F&I band for NH4+, NH3 and H3O+ (their X-H valences are uncertain)
    donor_minor: float = 0.05       # vu beyond the band with only a minor penalty
    minor_penalty: float = 1.0       # size of that minor penalty (reached at band + minor)
    hh_red: float = 1.95             # A, H...H between different donors below this: red flag (harsh)
    hh_red_penalty: float = -1000.0  # at hh_red; grows further below it
    hh_mod: float = 2.1              # A, sliding scale from 0 here ...
    hh_mod_penalty: float = -100.0   # ... to this at hh_red (quadratic)
    dis_conflict_penalty: float = -10.0   # H...H < hh_mod between two disordered donors whose occupancies
                                          # sum to <= 1 (may never coexist): donor conflict, not geometry
    acc_cone_factor: float = 1.5     # water acceptor cone: base diameter = factor x H...H of that water
                                     # (an H-bond arriving inside it costs hh_red_penalty)
    hh_severe: float = 1.75          # A, schemes with an inferred H...H below this are hidden ...
    show_severe: bool = False        # ... unless this is on
    ignore_located: bool = False     # treat H atoms in the CIF as absent (predict every H)
    rank_by: str = "fi"              # 'fi' (Ferraris & Ivaldi), 'grg' or 'both' (mean)
    # --- GRG: each D-H may change by up to +- dh_range
    dh_range: float = 0.12           # A
    dh_step: float = 0.01            # A
    # --- disorder: sites that cannot be present at the same time
    split_dist: float = 2.2          # A, two partly occupied sites closer than this with occ sum <= 1
    group_dist: float = 3.5          # A, different CIF disorder groups of one assembly within this
    disorder_mode: str = "donor"     # disordered water/hydroxyl: 'donor' | 'acceptor' (acceptor only) | 'exclude'
    flip_penalty: float = -100.0     # per located H that must flip away from a disordered acceptor
    # --- impossible hypothetical H (options dropped unless show_severe)
    h_min_cation: float = 1.8        # A, H closer than this to a cation
    h_min_anion: float = 1.4         # A, H closer than this to an anion other than its parent
    # --- located H (H atoms in the CIF): parent and H-bond criteria
    loc_max_dh: float = 1.30
    loc_max_ha: float = 2.70
    loc_max_da: float = 3.50
    loc_min_angle: float = 120.0
    # --- scheme search
    max_options: int = 30            # best options kept per donor
    exhaustive_limit: int = 2000000  # enumerate every combination up to this many, else beam search + refinement
    beam_width: int = 5000
    keep_schemes: int = 300
    # --- refinement (rigid rotation of inferred H about their parent O)
    refine_max_deg: float = 10.0     # maximum rotation of a water / hydroxyl
    refine_n_worst: int = 1          # donors refined in "worst-fitting" mode
    refine_sigma_bv: float = 0.05    # vu, weight of GRG BV deviations in the least-squares sum
    refine_sigma_hh: float = 0.05    # A, weight of H...H shortfall below hh_mod (quadratic + quartic)
    refine_hdh_deg: float = 2.0      # +- deg the H-D-H angle may open/close during refinement
    refine_w_sym: float = 1.0        # weight of the bonus for a water's two H-bonds having similar strength
    refine_auto_pick: bool = False   # refine the best scheme without H...H violations (or the least violating)
    refine_dh: bool = False          # also refine each inferred D-H length ...
    refine_dh_range: float = 0.06    # ... within +- this (A) = 2 sigma of a Gaussian restraint ...
    refine_dh_weight: float = 1.0    # ... costing weight x (change / sigma)^2
    # --- vanadium (Cooper et al. 2019): Vp = a*Vc + b
    v_slope: float = 1.538
    v_intercept: float = -2.692
    v_vc_state: int = 5              # oxidation state whose parameters give Vc


ANION_CHARGE = {"O": -2, "F": -1, "Cl": -1, "Br": -1, "I": -1,
                "S": -2, "Se": -2, "Te": -2, "N": -3}
ALWAYS_ANION = {"O", "F", "Cl", "Br", "I"}
FORMULA_TIE = 0.25      # v.u.: the formula charge balance only picks between states whose BV fits are this close
# element -> an O within this distance (A) makes the atom a cation, otherwise anion
CONDITIONAL_ANION = {"S": 2.0, "Se": 2.0, "Te": 2.0, "N": 1.6}

CLASS_WATER = "water"
CLASS_WATER_LOW = "water (BVS < min)"
CLASS_OH = "hydroxyl"
CLASS_PROBABLE = "probable acceptor"
CLASS_POSSIBLE = "possible acceptor"
CLASS_NOT = "not acceptor"
CLASS_POH = "partial hydroxyl"           # O disordered between hydroxyl and acceptor: H fraction x refined
CLASS_H3O = "hydronium (H3O+)"
CLASS_NH4 = "ammonium (NH4+)"
CLASS_NH3 = "ammine (NH3)"
CLASS_NH2 = "amide (NH2)"
DONOR_CLASSES = (CLASS_WATER, CLASS_WATER_LOW, CLASS_OH, CLASS_POH, CLASS_H3O, CLASS_NH4, CLASS_NH3, CLASS_NH2)
ALL_CLASSES = (CLASS_WATER, CLASS_WATER_LOW, CLASS_OH, CLASS_POH, CLASS_H3O, CLASS_PROBABLE, CLASS_POSSIBLE,
               CLASS_NOT, CLASS_NH4, CLASS_NH3, CLASS_NH2)
H_PER_CLASS = {CLASS_WATER: 2, CLASS_WATER_LOW: 2, CLASS_OH: 1, CLASS_POH: 1, CLASS_H3O: 3,
               CLASS_NH4: 4, CLASS_NH3: 3, CLASS_NH2: 2}
NON_ACCEPTOR_CLASSES = {CLASS_NH4, CLASS_NH3, CLASS_H3O}   # no free lone pair for an H-bond


def class_h(bv, i):
    """Number of H a site carries from its class (a partial hydroxyl: its refined H fraction x)."""
    c = bv.cls[i]
    if c == CLASS_POH:
        return getattr(bv, "poh_x", {}).get(i, 0.5)
    return H_PER_CLASS.get(c, 0)


def classify_nitrogen(v, s):
    """N classes from the BV sum without H (N3-: |V| = 3 shared between cations and N-H)."""
    if v < s.nh4_max:
        return CLASS_NH4
    if v < s.nh3_max:
        return CLASS_NH3
    if v < s.nh2_max:
        return CLASS_NH2
    return CLASS_PROBABLE


def _g(x):
    return "?" if x is None else ("%g" % x)


def formula_h_per_cell(st):
    """H atoms per unit cell from _chemical_formula_sum x Z (None if either is missing)."""
    if not st.formula or st.Z is None:
        return None
    return st.formula.get("H", 0.0) * st.Z


def model_formula(st, bv):
    """Formula per formula unit modelled from the sites (occupancy x multiplicity / Z; H atoms in the CIF
    are not counted - H come from the D/A classes instead). Returns ({element: n}, Z used)."""
    mult = site_multiplicities(st)
    cell = {}
    for i, a in enumerate(st.atoms):
        if a.el == "H":
            continue
        cell[a.el] = cell.get(a.el, 0.0) + a.occ * mult[i]
    cell["H"] = class_h_per_cell(st, bv)        # every site's class (acceptor-only / excluded sites included)
    z = st.Z
    if not z and st.formula:                     # no Z: scale on the anions of the given formula
        ref = next((e for e in ("O", "F", "Cl") if e in st.formula and cell.get(e)), None)
        z = cell[ref] / st.formula[ref] if ref else None
    z = z or 1.0
    return {e: n / z for e, n in cell.items() if n > 1e-6}, z


def formula_charge(st, bv, h_cell=None):
    """Charge balance of the model per formula unit: {'total', 'cations', 'anions', 'H', 'z'}.
    Cations: occupancy x multiplicity x assigned (average) valence; anions: their formal charge; H: +1
    each, from the D/A classes (or h_cell H per unit cell, e.g. those of a scheme). H atoms in the CIF are
    not counted separately (the classes stand for them)."""
    mult = site_multiplicities(st)
    cat = an = 0.0
    for i, a in enumerate(st.atoms):
        if a.el == "H":
            continue
        v = bv.valence.get(i) if isinstance(bv.valence, dict) else bv.valence[i]
        if v is None:
            continue
        if bv.roles[i] == "cation":
            cat += a.occ * mult[i] * v
        elif bv.roles[i] == "anion":
            an += a.occ * mult[i] * v
    h = class_h_per_cell(st, bv) if h_cell is None else h_cell
    _f, z = model_formula(st, bv)
    return {"total": (cat + an + h) / z, "cations": cat / z, "anions": an / z, "H": h / z, "z": z}


def scheme_formula(st, bv, hs):
    """Formula per formula unit from the sites with the H of a scheme (h_per_cell) instead of the classes."""
    f, z = model_formula(st, bv)
    nh = h_per_cell(st, hs) / z
    if nh > 1e-6:
        f["H"] = nh
    else:
        f.pop("H", None)
    return f, z


ATOMIC_WEIGHT = dict(zip(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu".split(),
    [1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.085,
     30.974, 32.06, 35.45, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693,
     63.546, 65.38, 69.723, 72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224, 92.906, 95.95,
     98.0, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.60, 126.90, 131.29, 132.91, 137.33,
     138.91, 140.12, 140.91, 144.24, 145.0, 150.36, 151.96, 157.25, 158.93, 162.50, 164.93, 167.26, 168.93,
     173.05, 174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2,
     208.98, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.04, 231.04, 238.03, 237.0, 244.0]))


def formula_weight(f):
    """Formula weight (g/mol) of {element: n}; None if an element is unknown."""
    try:
        return sum(ATOMIC_WEIGHT[e] * n for e, n in f.items())
    except KeyError:
        return None


def oxidation_state_text(st, bv):
    """'U6+, Se4+, Cu2+, V4.62+' - the (average) valence of each cation element, occupancy-weighted."""
    mult = site_multiplicities(st)
    tot = {}
    for i, a in enumerate(st.atoms):
        if a.el == "H" or bv.roles[i] != "cation" or bv.valence[i] is None:
            continue
        w = a.occ * mult[i]
        t = tot.setdefault(a.el, [0.0, 0.0])
        t[0] += w * bv.valence[i]
        t[1] += w
    out = []
    for e, (q, w) in tot.items():
        v = q / w if w else 0.0
        out.append("%s%s+" % (e, ("%d" % round(v)) if abs(v - round(v)) < 5e-3 else ("%.2f" % v)))
    return ", ".join(out)


def formula_text(f, order=None):
    order = list(dict.fromkeys(order or []))
    els = [e for e in order if e in f] + sorted(e for e in f if e not in order)
    return " ".join("%s%s" % (e, "" if abs(f[e] - 1) < 5e-3 else ("%.2f" % f[e]).rstrip("0").rstrip("."))
                    for e in els if f[e] > 5e-3)


def formula_comparison(st, bv, tol=0.02):
    """(given, modelled, status, differences): status 'agree' (all elements incl. H), 'noH' (agree
    apart from H, which the given formula lacks while the model has H) or 'differ'. None without a
    given formula."""
    if not st.formula:
        return None
    given = dict(st.formula)
    model, _z = model_formula(st, bv)
    diffs = []
    for e in sorted(set(given) | set(model)):
        g, m = given.get(e, 0.0), model.get(e, 0.0)
        if abs(g - m) > max(tol * max(abs(g), abs(m)), 0.02):
            diffs.append("%s %s vs %s" % (e, ("%.2f" % g).rstrip("0").rstrip("."), ("%.2f" % m).rstrip("0").rstrip(".")))
    if not diffs:
        return given, model, "agree", []
    non_h = [d for d in diffs if not d.startswith("H ")]
    if not non_h and given.get("H", 0.0) == 0 and model.get("H", 0.0) > 0:
        return given, model, "noH", []
    return given, model, "differ", diffs


def class_h_per_cell(st, bv):
    """H per cell implied by the O classes: (2 per water, 1 per hydroxyl) x occupancy x multiplicity."""
    mult = site_multiplicities(st)
    return sum(class_h(bv, i) * a.occ * mult[i] for i, a in enumerate(st.atoms))


def h_per_cell(st, hs):
    """H per unit cell in a set of H: hypothetical H count with their parent's occupancy;
    located H with their own CIF occupancy and position multiplicity."""
    mult = site_multiplicities(st)
    n = 0.0
    for h in hs:
        if not getattr(h, "primary", True):
            continue                     # extra H-bond branches of one located H
        if h.located:
            k = next((k for k, a in enumerate(st.atoms) if a.label == h.located), None)
            a = st.atoms[k] if k is not None else st.atoms[h.donor]
            n += a.occ * position_multiplicity(st, a.f)
        else:
            n += st.atoms[h.donor].occ * mult[h.donor] * getattr(h, "occf", 1.0)
    return n


def ordinal(n):
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return "%d%s" % (n, suf)


# ============================================================================
# CIF parsing
# ============================================================================

_TOKEN = re.compile(r"""'(.*?)'(?=\s|$)|"(.*?)"(?=\s|$)|(\S+)""")


def _tokenize(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            i += 1
            yield "\n".join(buf), True
            continue
        for m in _TOKEN.finditer(line):
            if m.group(3) is not None:
                tok = m.group(3)
                if tok.startswith("#"):
                    break
                yield tok, False
            else:
                yield (m.group(1) if m.group(1) is not None else m.group(2)), True
        i += 1


def _is_reserved(tok):
    low = tok.lower()
    return (tok.startswith("_") or low == "loop_" or low.startswith("data_")
            or low.startswith("save_") or low in ("global_", "stop_"))


def parse_cif(text):
    blocks, block = [], None
    toks = list(_tokenize(text))
    n, i = len(toks), 0
    while i < n:
        tok, quoted = toks[i]
        low = tok.lower()
        if not quoted and low.startswith("data_"):
            block = {"name": tok[5:], "items": {}, "loops": []}
            blocks.append(block)
            i += 1
        elif block is None:
            i += 1
        elif not quoted and low == "loop_":
            i += 1
            tags = []
            while i < n and not toks[i][1] and toks[i][0].startswith("_"):
                tags.append(toks[i][0].lower())
                i += 1
            vals = []
            while i < n and (toks[i][1] or not _is_reserved(toks[i][0])):
                vals.append(toks[i][0])
                i += 1
            if tags:
                k = len(tags)
                block["loops"].append((tags, [vals[j:j + k] for j in range(0, len(vals) - k + 1, k)]))
        elif not quoted and tok.startswith("_"):
            if i + 1 < n and (toks[i + 1][1] or not _is_reserved(toks[i + 1][0])):
                block["items"][low] = toks[i + 1][0]
                i += 2
            else:
                i += 1
        else:
            i += 1
    return blocks


def find_loop(block, *tags):
    for ltags, rows in block["loops"]:
        if any(t in ltags for t in tags):
            return {t: [r[c] for r in rows] for c, t in enumerate(ltags)}
    return None


_NUM = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def cif_number(s):
    if s is None:
        return None
    s = s.strip()
    if s in ("?", "."):
        return None
    m = _NUM.match(s)
    return float(m.group(0)) if m else None


_TWO_LETTER = set("""He Li Be Ne Na Mg Al Si Cl Ar Ca Sc Ti Cr Mn Fe Co Ni Cu Zn Ga Ge As Se
Br Kr Rb Sr Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy
Ho Er Tm Yb Lu Hf Ta Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa Np Pu Am Cm Bk Cf""".split())


def element_from_type(type_symbol):
    m = re.match(r"[A-Za-z]+", type_symbol or "")
    if not m:
        return None
    letters = m.group(0)
    two = letters[:2].capitalize()
    el = two if len(letters) >= 2 and two in _TWO_LETTER else letters[0].upper()
    return "H" if el == "D" else el


def element_from_label(label):
    m = re.match(r"[A-Za-z]+", label or "")
    if not m:
        return None
    letters = m.group(0)
    if len(letters) >= 2 and letters[1].islower() and letters[:2] in _TWO_LETTER:
        return letters[:2]
    el = letters[0].upper()
    return "H" if el == "D" else el


def parse_formula(s):
    """'Ca O6 S H4' or 'Na6 V10 O28' -> {'Ca': 1.0, 'O': 6.0, ...}; None if absent."""
    if not s or s.strip() in ("?", "."):
        return None
    out = {}
    for el, num in re.findall(r"([A-Z][a-z]?)\s*(\d*\.?\d*)", s):
        el = "H" if el == "D" else el
        out[el] = out.get(el, 0.0) + (float(num) if num not in ("", ".") else 1.0)
    return out or None


def _frac_value(s):
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def parse_symop(s):
    s = s.replace(" ", "").lower()
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError("Cannot parse symmetry operator: %r" % s)
    R, t = [], []
    for p in parts:
        row, const = [0.0, 0.0, 0.0], 0.0
        for sign, body in re.findall(r"([+-]?)([^+-]+)", p):
            sg = -1.0 if sign == "-" else 1.0
            m = re.fullmatch(r"([\d./]*)\*?([xyz])", body)
            if m:
                row["xyz".index(m.group(2))] += sg * (_frac_value(m.group(1)) if m.group(1) else 1.0)
                continue
            m = re.fullmatch(r"([xyz])/(\d+)", body)
            if m:
                row["xyz".index(m.group(1))] += sg / float(m.group(2))
                continue
            const += sg * _frac_value(body)
        R.append(row)
        t.append(const)
    return R, t


def is_identity(R, t):
    return all(abs(R[i][j] - (1.0 if i == j else 0.0)) < 1e-6 for i in range(3) for j in range(3)) \
        and all(abs(x) < 1e-6 for x in t)


def format_symop(R, t, shift):
    out = []
    for i in range(3):
        s = ""
        for j, v in enumerate("xyz"):
            c = R[i][j]
            if abs(c) < 1e-6:
                continue
            fr = Fraction(c).limit_denominator(12)
            mag = abs(fr)
            s += ("-" if fr < 0 else "+") + (v if mag == 1 else "%s%s" % (mag, v))
        const = Fraction(t[i] + shift[i]).limit_denominator(12)
        if const != 0:
            s += ("-" if const < 0 else "+") + str(abs(const))
        out.append(s.lstrip("+") or "0")
    return ",".join(out)


# ============================================================================
# Structure
# ============================================================================

class Atom:
    __slots__ = ("label", "el", "f", "occ", "uiso", "asm", "grp", "occ_cif")

    def __init__(self, label, el, f, occ, uiso=None, asm=".", grp="."):
        self.label, self.el, self.f, self.occ, self.uiso = label, el, f, occ, uiso
        self.asm, self.grp = asm, grp          # CIF disorder assembly / group ('.' = none)
        self.occ_cif = occ                     # as in the CIF (occ may be normalised, see normalize_occupancies)


class Neighbor:
    __slots__ = ("j", "d", "vec", "op", "shift")

    def __init__(self, j, d, vec, op, shift):
        self.j, self.d, self.vec, self.op, self.shift = j, d, vec, op, shift

    @property
    def key(self):
        return (self.j, round(self.vec[0], 2), round(self.vec[1], 2), round(self.vec[2], 2))


class Structure:
    def __init__(self, block, source=""):
        it = block["items"]
        self.name = block["name"]
        self.source = source
        try:
            self.a, self.b, self.c = (cif_number(it["_cell_length_" + k]) for k in "abc")
            self.alpha, self.beta, self.gamma = (
                cif_number(it["_cell_angle_" + k]) for k in ("alpha", "beta", "gamma"))
        except KeyError as e:
            raise ValueError("Missing cell parameter %s in data block '%s'" % (e, self.name))
        if None in (self.a, self.b, self.c, self.alpha, self.beta, self.gamma):
            raise ValueError("Incomplete cell parameters in data block '%s'" % self.name)
        ca, cb, cg = (math.cos(math.radians(x)) for x in (self.alpha, self.beta, self.gamma))
        sg = math.sin(math.radians(self.gamma))
        v = math.sqrt(max(0.0, 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg))
        self.M = np.array([[self.a, self.b * cg, self.c * cb],
                           [0.0, self.b * sg, self.c * (ca - cb * cg) / sg],
                           [0.0, 0.0, self.c * v / sg]])
        self.volume = self.a * self.b * self.c * v

        self.spacegroup = (it.get("_space_group_name_h-m_alt")
                           or it.get("_symmetry_space_group_name_h-m") or "?").strip()
        self.formula_text = it.get("_chemical_formula_sum", "?")
        self.formula = parse_formula(self.formula_text)
        self.res_text = it.get("_shelx_res_file") or None      # SHELXL .res embedded in the CIF
        self.hkl_text = it.get("_shelx_hkl_file") or None

        sym_tags = ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz",
                    "_space_group_symop.operation_xyz")
        loop = find_loop(block, *sym_tags)
        self.symops, self.sym_warning = [], None
        if loop:
            tag = next(t for t in sym_tags if t in loop)
            self.symops = [parse_symop(s) for s in loop[tag]]
        if not self.symops:
            self.symops = [([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.0, 0.0, 0.0])]
            self.sym_warning = "No symmetry operators found - using x,y,z only."

        loop = find_loop(block, "_atom_site_fract_x")
        if not loop or "_atom_site_label" not in loop:
            raise ValueError("No _atom_site loop with fractional coordinates in '%s'" % self.name)
        types = loop.get("_atom_site_type_symbol")
        flags = loop.get("_atom_site_calc_flag")
        occs = loop.get("_atom_site_occupancy")
        uisos = loop.get("_atom_site_u_iso_or_equiv")
        bisos = loop.get("_atom_site_b_iso_or_equiv")
        asms = loop.get("_atom_site_disorder_assembly")
        grps = loop.get("_atom_site_disorder_group")
        self.Z = cif_number(it.get("_cell_formula_units_z"))
        self.atoms = []
        for i, label in enumerate(loop["_atom_site_label"]):
            if flags and flags[i].lower() == "dum":
                continue
            f = [cif_number(loop["_atom_site_fract_" + k][i]) for k in "xyz"]
            if None in f:
                continue
            el = (element_from_type(types[i]) if types else None) or element_from_label(label)
            occ = cif_number(occs[i]) if occs else None
            u = cif_number(uisos[i]) if uisos else None
            if u is None and bisos:
                b = cif_number(bisos[i])
                u = b / (8 * math.pi ** 2) if b is not None else None
            asm = asms[i] if asms and asms[i] not in ("?", "") else "."
            grp = grps[i] if grps and grps[i] not in ("?", "") else "."
            self.atoms.append(Atom(label, el, f, 1.0 if occ is None else occ, u, asm, grp))
        self._env = {}
        self.occ_notes = normalize_occupancies(self)

    @property
    def title(self):
        t = "%s : %s" % (os.path.basename(self.source), self.name) if self.source else self.name
        return t + getattr(self, "tag", "")

    def symop_text(self, op, shift):
        R, t = self.symops[op]
        if is_identity(R, t) and tuple(shift) == (0, 0, 0):
            return "x,y,z"
        return format_symop(R, t, shift)

    def symop_code(self, op, shift):
        R, t = self.symops[op]
        if is_identity(R, t) and tuple(shift) == (0, 0, 0):
            return "."
        if all(-4 <= x <= 4 for x in shift):
            return "%d_%d%d%d" % (op + 1, 5 + shift[0], 5 + shift[1], 5 + shift[2])
        return "%d_%s" % (op + 1, ",".join(str(x) for x in shift))

    def environment(self, radius):
        """List (per asymmetric-unit atom) of Neighbor objects within radius, sorted by distance."""
        key = round(radius, 3)
        if key in self._env:
            return self._env[key]
        nat, nops = len(self.atoms), len(self.symops)
        F = np.array([a.f for a in self.atoms], dtype=float)
        R = np.array([op[0] for op in self.symops], dtype=float)
        T = np.array([op[1] for op in self.symops], dtype=float)
        G = (np.einsum("oij,aj->oai", R, F) + T[:, None, :]).reshape(-1, 3)
        img_atom = np.tile(np.arange(nat), nops)
        img_op = np.repeat(np.arange(nops), nat)
        rec = np.linalg.norm(np.linalg.inv(self.M), axis=1)   # |a*|, |b*|, |c*|
        nr = [int(math.ceil(radius * r)) for r in rec]
        shifts = np.array(list(product(*[range(-n, n + 1) for n in nr])), dtype=float)
        env = []
        for i in range(nat):
            d0 = G - F[i]
            base = -np.round(d0)
            D = (d0 + base)[:, None, :] + shifts[None, :, :]
            C = D @ self.M.T
            dist = np.sqrt((C * C).sum(axis=2))
            ii, ss = np.nonzero((dist <= radius) & (dist > 1e-3))
            seen, nbs = set(), []
            for a, s in zip(ii.tolist(), ss.tolist()):
                vec = C[a, s]
                k = (int(img_atom[a]), round(vec[0], 3), round(vec[1], 3), round(vec[2], 3))
                if k in seen:            # same position from a duplicate operator
                    continue
                seen.add(k)
                sh = base[a] + shifts[s]
                nbs.append(Neighbor(int(img_atom[a]), float(dist[a, s]), (float(vec[0]), float(vec[1]), float(vec[2])),
                                    int(img_op[a]), (int(round(sh[0])), int(round(sh[1])), int(round(sh[2])))))
            nbs.sort(key=lambda n: n.d)
            env.append(nbs)
        self._env[key] = env
        return env


def normalize_occupancies(st):
    """Occupancies above 1 (often left in a CIF to show that a site holds some of a heavier atom) are treated
    as a fully occupied site: one atom at a position -> occupancy 1; several atoms with the SAME coordinates
    whose occupancies add up to more than 1 -> scaled so that they add up to 1. The CIF's values are kept in
    atom.occ_cif. Returns notes (empty when nothing changed)."""
    notes, done = [], set()
    atoms = st.atoms
    for i, a in enumerate(atoms):
        a.occ_cif = a.occ
    for i, a in enumerate(atoms):
        if i in done:
            continue
        grp = [i]
        for j in range(i + 1, len(atoms)):
            if j in done or (atoms[j].el == "H") != (a.el == "H"):
                continue
            df = np.array(atoms[j].f, dtype=float) - np.array(a.f, dtype=float)
            df -= np.round(df)
            if np.linalg.norm(st.M @ df) < 0.01:
                grp.append(j)
        done.update(grp)
        tot = sum(atoms[k].occ for k in grp)
        if tot <= 1.0 + 1e-6:
            continue
        if len(grp) == 1:
            notes.append("%s %.3f → 1" % (a.label, a.occ))
            a.occ = 1.0
        else:
            txt = " + ".join("%s %.3f" % (atoms[k].label, atoms[k].occ) for k in grp)
            for k in grp:
                atoms[k].occ = atoms[k].occ / tot
            notes.append("%s = %.3f → scaled to 1 (%s)" % (txt, tot, " + ".join(
                "%s %.3f" % (atoms[k].label, atoms[k].occ) for k in grp)))
    return notes


def load_structures(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        blocks = parse_cif(fh.read())
    if not blocks:
        raise ValueError("No data blocks found in %s - is this a CIF file?" % os.path.basename(path))
    structures, errors = [], []
    for b in blocks:
        if find_loop(b, "_atom_site_fract_x") is None:
            continue
        try:
            structures.append(Structure(b, source=path))
        except ValueError as e:
            errors.append(str(e))
    if not structures:
        raise ValueError("No usable structure in %s.\n%s" % (os.path.basename(path), "\n".join(errors)))
    return structures


# ============================================================================
# Geometry helpers
# ============================================================================

def angle_between(u, v):
    nu = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
    nv = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))   # always 0-180


def neighbor_angles(st, i, radius, env_radius=6.0):
    """Angles X-i-Y between all neighbours of site i within `radius`."""
    nbs = [n for n in st.environment(env_radius)[i] if n.d <= radius]
    out = []
    for p, q in combinations(range(len(nbs)), 2):
        a, b = nbs[p], nbs[q]
        out.append((a, b, angle_between(a.vec, b.vec)))
    return out


# ============================================================================
# Bond-valence parameters
# ============================================================================

class BVParam:
    __slots__ = ("cation", "val", "anion", "anion_val", "ro", "b", "ref", "details")

    def __init__(self, cation, val, anion, anion_val, ro, b, ref, details):
        self.cation, self.val, self.anion, self.anion_val = cation, val, anion, anion_val
        self.ro, self.b, self.ref, self.details = ro, b, ref, details

    def valence(self, d):
        return math.exp((self.ro - d) / self.b)

    def __str__(self):
        return "%s%d+-%s: Ro=%.3f B=%.3f [%s]" % (self.cation, self.val, self.anion, self.ro, self.b, self.ref)

    @property
    def source(self):
        """Readable source: known reference code, or the details text of a 'user' row."""
        if self.ref == "user" and self.details:
            return self.details
        return REF_NAMES.get(self.ref, self.ref)


REF_NAMES = {"bs": "Gagne & Hawthorne 2015", "b": "Brese & O'Keeffe 1991",
             "a": "Brown & Altermatt 1985", "bc": "Brown 2002", "user": "user-preferred",
             "s": "García-Rodríguez et al. 2000"}


class BVParams:
    def __init__(self, path=DEFAULT_PARAM_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError("Bond-valence parameter file not found: %s" % path)
        self.path = path
        self.table = {}
        self.rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split(None, 7)
                if len(p) < 7:
                    continue
                prm = BVParam(p[0], int(p[1]), p[2], int(p[3]), float(p[4]), float(p[5]), p[6],
                              p[7].strip("'") if len(p) > 7 else "")
                self.rows.append(prm)
                self.table.setdefault((prm.cation, prm.val, prm.anion), []).append(prm)

    def get(self, cation, val, anion):
        """Preferred parameter: rows marked 'user' first; then cation-O -> G&H 2015, B&O'K, B&A;
        other anions -> B&O'K, B&A."""
        prefs = ("user", "bs", "b", "a") if anion == "O" else ("user", "b", "a")
        for v in (val, 9):
            rows = self.table.get((cation, v, anion), [])
            for ref in prefs:
                for r in rows:
                    if r.ref == ref:
                        return r
        for v in (val, 9):
            rows = self.table.get((cation, v, anion), [])
            if rows:
                return rows[0]
        return None

    def states(self, cation, anion):
        s = {r.val for r in self.rows if r.cation == cation and r.anion == anion and r.val != 9}
        if not s:
            s = {r.val for r in self.rows if r.cation == cation and r.val != 9}
        return s

    def hbond_valence(self, d_ha, acceptor_el):
        """Bond valence of an H...A bond of length d_ha. O: Brown (2002) range-dependent
        parameters; other acceptors: first H parameter in the file for that anion."""
        if acceptor_el == "O":
            rows = [r for r in self.table.get(("H", 1, "O"), []) if r.ref == "bc"]
            pick = None
            for r in rows:
                det = r.details.replace(" ", "")
                if d_ha < 1.05 and det.startswith("O-H<1.05"):
                    pick = r
                elif d_ha > 1.70 and det.startswith("1.70A<O-H"):
                    pick = r
                elif 1.05 <= d_ha <= 1.70 and det.startswith("1.05<O-H<1.70"):
                    pick = r
            if pick is not None:
                return pick.valence(d_ha), pick
        p = self.get("H", 1, acceptor_el)
        if p is None:
            return None, None
        return p.valence(d_ha), p


# ============================================================================
# Bond-valence analysis
# ============================================================================

class Bond:
    __slots__ = ("nb", "s", "occ", "too_short", "param", "included", "to_h", "blocked", "why")

    def __init__(self, nb, s, occ, too_short, param, to_h=False):
        self.nb, self.s, self.occ, self.too_short, self.param = nb, s, occ, too_short, param
        self.included = False
        self.to_h = to_h          # bond between an H atom and its anion
        self.blocked = False      # never summed by default (see why), unless overridden by hand
        self.why = ""             # why it is blocked: ammonium-cation, sites that cannot coexist, s > valence

    @property
    def weighted(self):
        return self.s * self.occ


class BVResult:
    pass


def assign_roles(st, env):
    roles = []
    for i, a in enumerate(st.atoms):
        if a.el in ALWAYS_ANION:
            roles.append("anion")
        elif a.el in CONDITIONAL_ANION:
            lim = CONDITIONAL_ANION[a.el]
            near_o = any(st.atoms[n.j].el == "O" and n.d <= lim for n in env[i])
            roles.append("cation" if near_o else "anion")
        else:
            roles.append("cation")
    return roles


def _is_included(i, bond, settings, overrides):
    k = (i, bond.nb.key)
    if overrides and k in overrides:
        return overrides[k]
    if bond.blocked:
        return False
    return (not bond.too_short) and bond.s >= settings.sum_cutoff


def _bond_list(st, env, roles, i, valence_of, params, settings, missing):
    """Bonds from site i to counter-ions. valence_of(k) -> valence of cation k (or None)."""
    a = st.atoms[i]
    out = []
    for nb in env[i]:
        j = nb.j
        if roles[i] == roles[j]:
            continue
        b = st.atoms[j]
        cat, an = (i, j) if roles[i] == "cation" else (j, i)
        v = valence_of(cat)
        if v is None:
            continue
        if isinstance(v, tuple):                 # mixed valence ("mix", v1, v2, fraction of v1)
            p1, p2 = (params.get(st.atoms[cat].el, w, st.atoms[an].el) for w in v[1:3])
            p = MixedParam(p1, p2, v[3]) if p1 is not None and p2 is not None else None
            if p is None:
                missing.add("%s%d+/%d+ - %s" % (st.atoms[cat].el, v[1], v[2], st.atoms[an].el))
                continue
        else:
            p = params.get(st.atoms[cat].el, v, st.atoms[an].el)
        if p is None:
            missing.add("%s%d+ - %s" % (st.atoms[cat].el, v, st.atoms[an].el))
            continue
        s = p.valence(nb.d)
        too_short = nb.d < settings.min_bond_length and "H" not in (a.el, b.el)
        if s < settings.list_cutoff and not too_short:
            continue
        bd = Bond(nb, s, b.occ, too_short, p, to_h=("H" in (a.el, b.el)))
        if not too_short and "H" not in (a.el, b.el):
            excl = getattr(st, "excl", None)
            if excl is not None and excl(i, j, nb.d):
                # two configurations of a disorder (alternatives / different PART groups): never together
                bd.blocked, bd.why = True, "not together (other disorder configuration)"
            elif a.occ < 0.999 and b.occ < 0.999:
                # two partly occupied atoms whose bond would exceed the valence of either one (typically a
                # 50/50 split, where occupancies cannot tell which atoms belong together): different
                # configurations. A partly occupied atom keeps its bonds to fully occupied neighbours.
                lim = min(abs(valence_value(v)), abs(ANION_CHARGE.get(st.atoms[an].el, 2)))
                if s > lim:
                    bd.blocked, bd.why = True, "s %.2f > valence %g (disorder: not together)" % (s, lim)
        out.append(bd)
    return out


class MixedParam:
    """Bond-valence parameter of a mixed-valence site: s = x * s(v1) + (1 - x) * s(v2). ro / b are the
    equivalent single-exponential values (exact when both b are equal)."""

    def __init__(self, p1, p2, x):
        self.p1, self.p2, self.x = p1, p2, float(x)
        self.cation, self.anion = p1.cation, p1.anion
        self.val = self.x * p1.val + (1 - self.x) * p2.val
        self.anion_val = p1.anion_val
        self.b = self.x * p1.b + (1 - self.x) * p2.b
        self.ro = self.b * math.log(max(1e-300, self.x * math.exp(p1.ro / self.b) +
                                        (1 - self.x) * math.exp(p2.ro / self.b)))
        self.ref = "mixed"

    def valence(self, d):
        return self.x * self.p1.valence(d) + (1 - self.x) * self.p2.valence(d)

    def __str__(self):
        return "%s %.0f%% %d+ / %.0f%% %d+ (R0 %.3f / %.3f, b %.3f / %.3f)" % (
            self.cation, 100 * self.x, self.p1.val, 100 * (1 - self.x), self.p2.val, self.p1.ro, self.p2.ro,
            self.p1.b, self.p2.b)

    @property
    def source(self):
        return "%s; %s" % (self.p1.source, self.p2.source)


def valence_text(v):
    """'3+' or, for a non-integer average, '3.50+'."""
    if v is None:
        return ""
    if abs(v - round(v)) < 1e-6:
        return "%d%s" % (abs(round(v)), "+" if v > 0 else "−")
    return "%.2f%s" % (abs(v), "+" if v > 0 else "−")


def valence_value(v):
    """Numeric (average) valence of an assigned valence (int, or ('mix', v1, v2, x))."""
    if isinstance(v, tuple):
        return v[3] * v[1] + (1 - v[3]) * v[2]
    return v


def _apply_valence_overrides(st, env, roles, params, settings, overrides, val, how, vov, notes,
                             forced=None, mixinfo=None):
    """Hand-set valences {site: spec}: {'v': n} one state; {'avg': q} average site charge (a mixture of
    the two states around q); {'mix': None or [v1, v2]} mixture with the BEST BV FIT (vanadium: the
    Cooper et al. 2019 estimate Vp).
    forced {site: q}: average charge of a 'mix' site fixed by the charge-balanced fit (analyse_bv).
    mixinfo (filled): {site: (best-fit average q*, misfit slope d(BVS - V)/dq, lowest, highest state)}."""
    forced = forced or {}
    atoms = st.atoms
    an_els = [a.el for a, r in zip(atoms, roles) if r == "anion"]
    main_an = "O" if "O" in an_els else (max(set(an_els), key=an_els.count) if an_els else "O")

    def bvs_for(i, v):
        bonds = _bond_list(st, env, roles, i, lambda k: v if k == i else None, params, settings, set())
        return _sum(i, bonds, settings, overrides)[0]

    def mix_of(states, q):
        if q in states:
            return int(q)
        lo = [s for s in states if s <= q]
        hi = [s for s in states if s >= q]
        if not lo or not hi:
            return int(min(states, key=lambda s: abs(s - q)))
        a, b = max(lo), min(hi)
        return ("mix", a, b, (b - q) / float(b - a))

    for i, spec in sorted(vov.items()):
        if not (0 <= i < len(atoms)) or roles[i] != "cation" or atoms[i].el == "H":
            continue
        el, lab = atoms[i].el, atoms[i].label
        states = sorted(params.states(el, main_an))
        if not states:
            continue
        if "v" in spec:
            if spec["v"] in states:
                val[i], how[i] = int(spec["v"]), "set by hand (%d+)" % spec["v"]
        elif "avg" in spec:
            q = float(spec["avg"])
            val[i] = mix_of(states, q)
            how[i] = "set by hand: average charge %+.3f" % q + (
                " = %.0f%% %d+ / %.0f%% %d+" % (100 * val[i][3], val[i][1], 100 * (1 - val[i][3]), val[i][2])
                if isinstance(val[i], tuple) else "")
        elif "mix" in spec:
            lo_s, hi_s = (states[0], states[-1]) if not spec.get("mix") else tuple(sorted(spec["mix"]))
            if el == "V" and spec.get("mix") is None:       # Cooper et al. (2019): Vp = a Vc + b
                vc = bvs_for(i, settings.v_vc_state)
                q = min(max(settings.v_slope * vc + settings.v_intercept, states[0]), states[-1])
                if mixinfo is not None:
                    mixinfo[i] = (q, 1.0, lo_s, hi_s)
                if i in forced:
                    val[i] = mix_of(states, forced[i])
                    how[i] = "mixed, charge-balanced fit: average %+.3f (Cooper et al. 2019 estimate Vp %.3f)" % (
                        forced[i], q)
                else:
                    val[i] = mix_of(states, q)
                    how[i] = "mixed, Cooper et al. 2019 estimate Vp %.3f" % q
            else:
                pairs = [tuple(spec["mix"])] if spec.get("mix") else list(zip(states, states[1:]))
                best = None
                for v1, v2 in pairs:
                    d1, d2 = bvs_for(i, v1) - v1, bvs_for(i, v2) - v2   # misfit with pure v1 / pure v2
                    den = d2 - d1
                    x = d2 / den if abs(den) > 1e-9 else 0.5
                    x = min(1.0, max(0.0, x))
                    resid = abs(x * d1 + (1 - x) * d2)
                    if best is None or resid < best[0] - 1e-9:
                        best = (resid, v1, v2, x)
                if best is None:
                    continue
                _r, v1, v2, x = best
                qbest = x * v1 + (1 - x) * v2
                if mixinfo is not None:
                    d1, d2 = bvs_for(i, v1) - v1, bvs_for(i, v2) - v2
                    slope = (d1 - d2) / float(v1 - v2) if v1 != v2 else 1.0
                    mixinfo[i] = (qbest, slope if abs(slope) > 0.05 else 0.05, lo_s, hi_s)
                if i in forced:
                    q = forced[i]
                    val[i] = mix_of(states, q)
                    how[i] = "mixed, charge-balanced fit: average %+.3f%s (best BV fit alone %+.3f)" % (
                        q, (" = %.0f%% %d+ / %.0f%% %d+" % (100 * val[i][3], val[i][1], 100 * (1 - val[i][3]), val[i][2])
                            if isinstance(val[i], tuple) else ""), qbest)
                else:
                    val[i] = ("mix", v1, v2, x) if 1e-6 < x < 1 - 1e-6 else (v1 if x >= 0.5 else v2)
                    how[i] = "mixed, best BV fit: %.0f%% %d+ / %.0f%% %d+ (average %+.3f)" % (
                        100 * x, v1, 100 * (1 - x), v2, qbest)
        notes.append("Valence of %s (hand-set): %s." % (lab, how[i]))


def _sum(i, bonds, settings, overrides):
    """(sum, listed-not-summed, number summed, sum without bonds to H)."""
    tot = weak = no_h = 0.0
    n = 0
    for bd in bonds:
        bd.included = _is_included(i, bd, settings, overrides)
        if bd.included:
            tot += bd.weighted
            n += 1
            if not bd.to_h:
                no_h += bd.weighted
        elif not bd.too_short:
            weak += bd.weighted
    return tot, weak, n, no_h


def classify_oxygen(v, s):
    if v < s.water_min:
        return CLASS_WATER_LOW
    if v < s.water_max:
        return CLASS_WATER
    if v < s.hydroxyl_max:
        return CLASS_OH
    if v < s.probable_max:
        return CLASS_PROBABLE
    if v < s.possible_max:
        return CLASS_POSSIBLE
    return CLASS_NOT


def _assign_valences(st, env, roles, params, settings, overrides, notes):
    atoms = st.atoms
    an_els = [a.el for a, r in zip(atoms, roles) if r == "anion"]
    main_an = "O" if "O" in an_els else (max(set(an_els), key=an_els.count) if an_els else "O")

    cand = {}
    for i, a in enumerate(atoms):
        if roles[i] != "cation":
            continue
        cand[i] = [1] if a.el == "H" else sorted(params.states(a.el, main_an))
        if not cand[i]:
            notes.append("No bond-valence parameters for %s (site %s) - its bonds are ignored." % (a.el, a.label))

    el_states = {}
    for i, c in cand.items():
        el_states.setdefault(atoms[i].el, set()).update(c)

    restrict, formula_avg = {}, {}
    f = st.formula
    if not f:
        notes.append("No _chemical_formula_sum: oxidation states chosen by best BV fit only.")
    else:
        anion_q, fixed, multi, ok = 0.0, 0.0, {}, True
        absent = sorted(el for el, n in f.items() if n > 0 and el != "H" and not any(a.el == el for a in atoms))
        if absent:
            notes.append("The formula lists %s, which the atom list does not contain - the formula does not "
                         "describe this model, so its charge balance is not used (oxidation states by "
                         "best BV fit only)." % ", ".join(absent))
            ok = False
        for el, n in (f.items() if ok else ()):
            r_el = {roles[i] for i, a in enumerate(atoms) if a.el == el}
            if r_el == {"anion"} or (not r_el and el in ALWAYS_ANION):
                anion_q += n * ANION_CHARGE[el]
            elif len(r_el) > 1:
                notes.append("%s occurs both as cation and anion - formula charge balance skipped." % el)
                ok = False
            elif el == "H":
                fixed += n
            else:
                states = sorted(el_states.get(el) or params.states(el, main_an))
                if not states:
                    notes.append("Unknown oxidation states for %s - formula charge balance skipped." % el)
                    ok = False
                elif len(states) == 1:
                    fixed += n * states[0]
                else:
                    multi[el] = (n, states)
        missing_el = sorted({a.el for a in atoms} - set(f))
        if missing_el:
            notes.append("Elements in the atom list but not in the formula: %s." % ", ".join(missing_el))
        if ok:
            target = -anion_q - fixed
            tol = 0.01 * abs(anion_q) + 0.02
            if not multi:
                notes.append("Formula charge balance: cations %+.2f, anions %+.2f (residual %+.2f)."
                             % (fixed, anion_q, fixed + anion_q))
            else:
                elems = list(multi)
                for e in elems:
                    formula_avg[e] = target / multi[e][0] if len(elems) == 1 else None
                ncomb = 1
                for e in elems:
                    ncomb *= len(multi[e][1])
                combos = []
                if ncomb <= 50000:
                    for vs in product(*[multi[e][1] for e in elems]):
                        if abs(sum(multi[e][0] * v for e, v in zip(elems, vs)) - target) <= tol:
                            combos.append(vs)
                if combos:
                    for k, e in enumerate(elems):
                        restrict[e] = sorted({c[k] for c in combos})
                    notes.append("Formula charge balance allows: " + "; ".join(
                        "%s %s" % (e, "/".join("%d+" % v for v in restrict[e])) for e in elems))
                elif len(elems) == 1:
                    e = elems[0]
                    n, states = multi[e]
                    avg = target / n
                    lo = [s for s in states if s <= avg]
                    hi = [s for s in states if s >= avg]
                    pair = sorted({x for x in ((max(lo) if lo else None), (min(hi) if hi else None)) if x is not None})
                    restrict[e] = pair or states
                    notes.append("Formula requires average %s valence %.3f (mixed valence?) - "
                                 "each %s site chosen by best BV fit among %s."
                                 % (e, avg, e, "/".join("%d+" % v for v in restrict[e])))
                else:
                    notes.append("No integer oxidation-state combination balances the formula - "
                                 "best BV fit used for %s." % ", ".join(elems))

    def bvs_for(i, v):
        tmp = set()
        bonds = _bond_list(st, env, roles, i, lambda k: v if k == i else None, params, settings, tmp)
        return _sum(i, bonds, settings, overrides)[0]

    # The bond valences are the primary evidence for each site's oxidation state: the state whose BV sum
    # matches it best wins. The formula charge balance only decides between states the BVs cannot tell
    # apart (fits within FORMULA_TIE v.u. of the best) - a formula may be incomplete (missing H, a
    # placeholder, a different model), so it never overrides a clear BV fit.
    val, how, fits, overruled = {}, {}, {}, {}
    for i, c in cand.items():
        if not c:
            val[i], how[i] = None, "no parameters"
            continue
        if len(c) == 1:
            val[i], how[i], fits[i] = c[0], "only state", {}
            continue
        e = atoms[i].el
        yl = _uranyl(st, env, i) if e == "U" and 6 in c else None
        if yl:
            # a uranyl ion: U6+ (U5+ uranyl is very rare in minerals); its BV sum can still come out low when
            # the structure is poorly refined, so the geometry decides here (set by hand if it really is U5+)
            val[i], fits[i] = 6, {v: bvs_for(i, v) for v in c}
            how[i] = "uranyl ion (%s) - U6+ (BV fits %s)" % (yl, ", ".join("%d+:%.2f" % (v, fits[i][v]) for v in c))
            continue
        fits[i] = {v: bvs_for(i, v) for v in c}
        dev = {v: abs(fits[i][v] - v) for v in c}
        best = min(c, key=lambda v: dev[v])
        txt = ", ".join("%d+:%.2f" % (v, fits[i][v]) for v in c)
        allowed = [v for v in restrict.get(e, []) if v in c]
        if not allowed or best in allowed:
            val[i] = best
            how[i] = "best BV fit (%s)%s" % (txt, "; formula agrees" if allowed else "")
            continue
        alt = min(allowed, key=lambda v: dev[v])
        if dev[alt] - dev[best] <= FORMULA_TIE:
            val[i] = alt
            how[i] = "formula charge balance - BV fits too close to decide (%s)" % txt
        else:
            val[i] = best
            how[i] = "best BV fit (%s); formula suggested %d+" % (txt, alt)
            overruled.setdefault(e, set()).add((alt, best))
    for e, prs in sorted(overruled.items()):
        notes.append("Bond valences overrule the formula charge balance for %s: %s (BV fit decides; the "
                     "formula may be incomplete, e.g. missing H)."
                     % (e, ", ".join("%d+ instead of %d+" % (b, a) for a, b in sorted(prs))))
    return val, how, fits, formula_avg


def _uranyl(st, env, i, max_d=2.2, short=1.95, min_angle=160.0):
    """'O–U–O 174.5°, 1.79 / 2.15 Å' when U site i carries a uranyl ion: two O on opposite sides (angle
    >= min_angle) within max_d, at least one of them shorter than `short`; else None."""
    ox = sorted((nb for nb in env[i] if st.atoms[nb.j].el == "O" and 0.5 < nb.d <= max_d), key=lambda n: n.d)
    best = None
    for x, p in enumerate(ox):
        for q in ox[x + 1:]:
            if min(p.d, q.d) > short:
                continue
            v1, v2 = np.asarray(p.vec, float), np.asarray(q.vec, float)
            ang = float(np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))
            if ang >= min_angle and (best is None or p.d + q.d < best[0]):
                best = (p.d + q.d, "O–U–O %.1f°, %.2f / %.2f Å" % (ang, p.d, q.d))
    return best[1] if best else None


_ALT_LABEL = re.compile(r"^(.*\d)([A-Za-z]|'{1,2})$")


def alternative_sites(st, s):
    """Pairs of partly occupied non-H sites that are ALTERNATIVES of one site even when they are further
    apart than split_dist (up to group_dist):
      - their occupancies are tied to the same SHELXL free variable with opposite signs in the embedded
        .res (e.g. OW6A 81.0 / OW6B -81.0: p and 1 - p), or
      - their labels differ only in a final letter (OW6A / OW6B, O3' ...) and their occupancies add up
        to 1 (+- 0.05).
    Returns {(i, j): (nearest distance between the two, reason)}; each site excludes only its partner
    (the nearest copy, +0.05 A): other copies belong to neighbouring disordered sites and may be present
    together with it (e.g. around a 3-fold axis one B and two A positions coexist locally)."""
    if getattr(st, "_alt_sites", None) is not None and getattr(st, "_alt_key", None) == (s.group_dist,):
        return st._alt_sites
    n = len(st.atoms)
    cand = [i for i, a in enumerate(st.atoms) if a.el != "H" and a.occ < 0.999]
    pairs = {}
    # (1) SHELX free variables in the embedded .res
    fv = {}
    for line in (st.res_text or "").splitlines():
        a = _res_atom(line)
        if not a:
            continue
        try:
            occ = float(a[2])
        except ValueError:
            continue
        if abs(occ) >= 15.0:                       # 10 * k + p with k >= 2 (k = 1 fixes the occupancy)
            k = int(abs(occ) // 10)
            fv[a[0].upper()] = (k, 1 if occ > 0 else -1)
    by_label = {st.atoms[i].label.upper(): i for i in cand}
    for la, (ka, sa) in fv.items():
        for lb, (kb, sb) in fv.items():
            if la < lb and ka == kb and sa != sb and la in by_label and lb in by_label:
                pairs[(min(by_label[la], by_label[lb]), max(by_label[la], by_label[lb]))] = \
                    "SHELX free variable %d (%s / %s)" % (ka, "+" if sa > 0 else "−", "+" if sb > 0 else "−")
    # (2) A / B labels with complementary occupancies
    stems = {}
    for i in cand:
        m = _ALT_LABEL.match(st.atoms[i].label)
        if m:
            stems.setdefault((m.group(1).upper(), st.atoms[i].el), []).append(i)
    for (_stem, _el), lst in stems.items():
        if len(lst) < 2:
            continue
        tot = sum(st.atoms[i].occ for i in lst)
        if abs(tot - 1.0) <= 0.05:
            for i, j in combinations(sorted(lst), 2):
                pairs.setdefault((i, j), "labels %s / %s, occupancies sum to %.2f" % (
                    st.atoms[i].label, st.atoms[j].label, tot))
    # (3) nearby sites with COMPLEMENTARY occupancies (p and 1 - p, p not ~1/2): two configurations of one
    # disorder, e.g. Hg4 (0.036) and the C1-O3-O4 group (0.964). A 50/50 pair is ambiguous (two half-
    # occupied atoms may just as well belong together) - there bonds stronger than a valence are dropped.
    env = st.environment(max(s.group_dist, 0.1))
    for x, i in enumerate(cand):
        for j in cand[x + 1:]:
            oi, oj = st.atoms[i].occ, st.atoms[j].occ
            if abs(oi + oj - 1.0) <= 0.03 and abs(oi - oj) >= 0.1 and (i, j) not in pairs:
                if any(nb.j == j and 0.05 < nb.d <= s.group_dist for nb in env[i]):
                    pairs[(i, j)] = "complementary occupancies %.3f + %.3f" % (oi, oj)
    out = {}
    mixed = {tuple(sorted(p)) for p in mixed_sites(st)}  # partners at the same position: handled as mixed sites
    if pairs:
        for (i, j), why in pairs.items():
            if (min(i, j), max(i, j)) in mixed:
                continue
            ds = [nb.d for nb in env[i] if nb.j == j and nb.d > 0.05]
            if ds:
                out[(i, j)] = (min(ds), why)
    st._alt_sites, st._alt_key = out, (s.group_dist,)
    return out


def position_occupancy(st, j):
    """Occupancy of site j plus every other site at the same position (a mixed site)."""
    return st.atoms[j].occ + sum(st.atoms[b if a == j else a].occ for a, b in mixed_sites(st) if j in (a, b))


def mixed_sites(st):
    """Pairs (i, j) of partly occupied sites at the same position (a mixed site such as N / OW2)."""
    if getattr(st, "_mixed", None) is None:
        out = []
        for i, a in enumerate(st.atoms):
            for j in range(i + 1, len(st.atoms)):
                b = st.atoms[j]
                if a.occ < 0.999 and b.occ < 0.999 and a.el != "H" and b.el != "H":
                    df = np.array(a.f) - np.array(b.f)
                    df -= np.round(df)
                    if np.linalg.norm(st.M @ df) < 0.05:
                        out.append((i, j))
        st._mixed = out
    return st._mixed


class Exclusivity:
    """Decides whether two sites (at distance d) cannot be occupied at the same time:
    both partly occupied, closer than split_dist and occupancies summing to <= 1; or in
    different CIF disorder groups of the same assembly within group_dist (a negative group,
    SHELX PART -n, also excludes its own symmetry copies)."""

    def __init__(self, st, s):
        self.st, self.split, self.gdist = st, s.split_dist, s.group_dist
        self.nh4, self.roles, self.nh4_min = set(), None, s.nh4_cation_min
        self.alt = alternative_sites(st, s)      # {(i, j): (nearest distance, why)} - alternatives of one site

    def _alt(self, i, j, d):
        k = (min(i, j), max(i, j))
        hit = self.alt.get(k)
        return hit is not None and d <= hit[0] + 0.05      # only the partner itself (nearest copy)

    @property
    def reach(self):
        return max(self.split, self.gdist, self.nh4_min if self.nh4 else 0.0)

    def __call__(self, i, j, d):
        a, b = self.st.atoms[i], self.st.atoms[j]
        if d < 0.05:                     # mixed site (e.g. N / OW2 at one position): one or the other
            return i != j and a.occ < 0.999 and b.occ < 0.999 and a.occ + b.occ <= 1.05
        if a.occ < 0.999 and b.occ < 0.999 and d < self.split and a.occ + b.occ <= 1.05:
            return True                  # split positions (also a partly occupied cation too close to a partly
                                         # occupied anion, e.g. Ca 0.16 - OW 0.50 at 1.97 A: one or the other)
        if self.alt and i != j and self._alt(i, j, d):
            return True                  # alternatives of one site (SHELX free variable / A-B labels), any distance
        if self.nh4 and d < self.nh4_min and a.occ < 0.999 and b.occ < 0.999 and a.occ + b.occ <= 1.05 and \
                ((i in self.nh4 and self.roles[j] == "cation") or (j in self.nh4 and self.roles[i] == "cation")):
            return True                  # ammonium and a partly occupied cation too close to coexist
        if d < self.gdist and a.grp not in (".", "0") and b.grp not in (".", "0") and a.asm == b.asm:
            if a.grp != b.grp and a.occ < 0.999 and b.occ < 0.999:
                return True              # (a fully occupied atom in a PART group is present in every configuration)
            if i == j and a.grp.startswith("-"):
                return True
        return False

    def any_pairs(self):
        """List of (label, label, d) exclusive pairs among close neighbours (for the notes)."""
        out, seen = [], set()
        env = self.st.environment(self.reach)
        for i, nbs in enumerate(env):
            for nb in nbs:
                if self(i, nb.j, nb.d):
                    k = (min(i, nb.j), max(i, nb.j), round(nb.d, 2))
                    if k not in seen:
                        seen.add(k)
                        out.append((self.st.atoms[i].label, self.st.atoms[nb.j].label, nb.d))
        return out


HROLE_NAMES = {"acceptor": "acceptor only", "exclude": "excluded"}


def disorder_groups(st, bv, s):
    """Clusters of mutually exclusive positions (symmetry copies included): a group is a
    connected set of positions in which every linked pair cannot be occupied together, i.e.
    occupancy of one means the others are vacant. For each group: members (site, neighbour or
    None for the starting site), the sum of their occupancies (1 = exactly one position occupied,
    < 1 = sometimes all vacant, > 1 = inconsistent) and the set of site indices."""
    excl = st.excl
    env = st.environment(s.env_radius)
    # ammonium-cation exclusions are pairwise (the cations around one NH4+ may coexist), so they are
    # left out of the connected groups below and listed as groups of their own at the end
    nh4 = set(getattr(excl, "nh4", set()))
    excl.nh4 = set()
    try:
        groups = _exclusion_groups(st, env, excl, max(s.split_dist, s.group_dist))
    finally:
        excl.nh4 = nh4
    for i in sorted(nh4):
        near = [nb for nb in env[i] if nb.d < s.nh4_cation_min and bv.roles[nb.j] == "cation" and excl(i, nb.j, nb.d)]
        if not near:
            continue
        members = [(i, (0.0, 0.0, 0.0), None)] + [(nb.j, nb.vec, nb) for nb in near]
        pos = {}
        for nb in near:                         # occupancy per cation POSITION (mixed sites add up)
            k = tuple(round(x, 2) for x in nb.vec)
            pos[k] = pos.get(k, 0.0) + st.atoms[nb.j].occ
        groups.append({"members": members, "occ_sum": st.atoms[i].occ + max(pos.values()),
                       "sites": {m[0] for m in members}, "labels": tuple(sorted(st.atoms[m[0]].label for m in members)),
                       "kind": "nh4"})
    return groups


def _exclusion_groups(st, env, excl, reach):
    groups, seen = [], set()
    for i, a in enumerate(st.atoms):
        if a.occ >= 0.999 and a.grp in (".", "0"):
            continue
        nodes = [(i, (0.0, 0.0, 0.0), None)] + [(nb.j, nb.vec, nb) for nb in env[i] if nb.d <= 2 * reach]
        adj = [[] for _ in nodes]
        for p in range(len(nodes)):
            for q in range(p + 1, len(nodes)):
                v1, v2 = nodes[p][1], nodes[q][1]
                d = math.sqrt(sum((v1[k] - v2[k]) ** 2 for k in range(3)))
                if excl(nodes[p][0], nodes[q][0], d):
                    adj[p].append(q)
                    adj[q].append(p)
        # a group = a largest set of positions that ALL exclude each other (a clique containing this
        # position), so that "at most one is occupied" holds; chains A-B, B-C with A, C compatible
        # give two groups instead of one inconsistent one
        nb_sets = [set(a_) for a_ in adj]
        cliques = []

        def bron(r, p, x):
            if not p and not x:
                cliques.append(r)
                return
            piv = max(p | x, key=lambda u: len(nb_sets[u] & p))
            for v in list(p - nb_sets[piv]):
                bron(r | {v}, p & nb_sets[v], x & nb_sets[v])
                p = p - {v}
                x = x | {v}
                if len(cliques) > 200:
                    return
        bron({0}, set(nb_sets[0]), set())
        for comp in cliques:
            if len(comp) < 2:
                continue
            members = [nodes[k] for k in sorted(comp)]
            key = tuple(sorted(st.atoms[m[0]].label for m in members))
            if key in seen:
                continue
            seen.add(key)
            occ_sum = sum(st.atoms[m[0]].occ for m in members)
            groups.append({"members": members, "occ_sum": occ_sum, "sites": {m[0] for m in members},
                           "labels": key})
    return groups


def group_text(st, g):
    """One-line description of an exclusion group."""
    counts = {}
    for j, _, _ in g["members"]:
        counts[st.atoms[j].label] = counts.get(st.atoms[j].label, 0) + 1
    names = ", ".join("%s%s" % (k, " ×%d" % n if n > 1 else "") for k, n in sorted(counts.items()))
    o = g["occ_sum"]
    if g.get("kind") == "nh4":
        n0 = st.atoms[g["members"][0][0]].label
        cats = ", ".join("%s%s" % (k, " ×%d" % n if n > 1 else "") for k, n in sorted(counts.items()) if k != n0)
        return "%s (NH4+) excludes the nearby cations %s: NH4+ + the fullest cation position = %.2f%s" % (
            n0, cats, o, "" if o <= 1.05 else " ⚠ exceeds 1 - check the model")
    if abs(o - 1) <= 0.05:
        meaning = "exactly one position occupied"
    elif o < 1:
        meaning = "at most one occupied (all vacant %.0f%% of the time)" % (100 * (1 - o))
    else:
        meaning = "⚠ occupancies of mutually exclusive positions exceed 1 - check the model"
    return "%s: occupancies sum to %.2f → %s" % (names, o, meaning)


def apply_site_roles(st, bv, s, role_overrides=None):
    """Decide the H-bond role of every anion: 'normal', 'acceptor' (does not donate, its own sum is
    not scored) or 'exclude' (neither donor nor acceptor, not scored). Disordered water/hydroxyl
    follow s.disorder_mode; role_overrides {site: role} (from the user or a hard-fail
    re-evaluation) win. Stores bv.groups, bv.disordered, bv.hrole, bv.h_excluded."""
    bv.groups = disorder_groups(st, bv, s)
    dis = set()
    bv.mates = {}
    for g in bv.groups:
        dis |= g["sites"]
        for j in g["sites"]:
            bv.mates.setdefault(j, {j}).update(g["sites"])
    dis |= {i for i, a in enumerate(st.atoms) if bv.roles[i] == "anion" and a.occ < 0.999}
    for i, j in mixed_sites(st):              # e.g. N / OW2 at one position: alternatives of each other
        if bv.roles[i] == "anion" and bv.roles[j] == "anion":
            bv.mates.setdefault(i, {i}).add(j)
            bv.mates.setdefault(j, {j}).add(i)
    bv.disordered = dis
    st._disordered = dis                 # used to classify H...H contacts (geometry vs donor conflict)
    hrole = {}
    if s.disorder_mode in ("acceptor", "exclude"):
        for i in dis:
            if bv.roles[i] == "anion" and bv.cls[i] in DONOR_CLASSES:
                hrole[i] = s.disorder_mode
    for i, r in (role_overrides or {}).items():
        if r == "normal":
            hrole.pop(i, None)
        elif r in ("acceptor", "exclude") and 0 <= i < len(st.atoms) and bv.roles[i] == "anion":
            hrole[i] = r
    bv.hrole = hrole
    mult = site_multiplicities(st)
    bv.h_excluded = sum(class_h(bv, i) * st.atoms[i].occ * mult[i] for i in hrole)
    return bv


def _ammonium_bonds(st, env, roles, params, settings, overrides, res, nh4, notes):
    """NH4+ as a whole (García-Rodríguez et al. 2000, bvparm2020 symbol NH, code s): the N of every site
    classed as ammonium is bonded to its O / F / Cl neighbours with s = exp((Ro - d)/b) from the N...X
    distance - its four H are included in these parameters. These bonds count for both atoms like any
    cation-anion bond (also in the anions' sums without H, so their classes follow), the N's own N-H bonds
    are not summed, and the ammonium H only set the geometry (they donate no further bond valence). Sets
    res.nh4_gr (the sites) and st._nh4_gr."""
    res.nh4_gr = set()
    st._nh4_gr = set()
    if not nh4 or not getattr(settings, "nh4_gr", True):
        return
    touched, sums = set(), {}
    excl = getattr(st, "excl", None)
    for i in sorted(nh4):
        a = st.atoms[i]
        for bd in res.bonds[i]:
            if bd.to_h:
                bd.blocked, bd.why = True, "N-H of ammonium: included in the NH4+ group bonds"
        for nb in env[i]:
            j = nb.j
            b = st.atoms[j]
            if roles[j] != "anion" or b.el == "H" or j in nh4 or nb.d < max(0.05, settings.min_bond_length):
                continue
            p = params.get("NH", 1, b.el)
            if p is None:
                continue
            s = p.valence(nb.d)
            if s < settings.list_cutoff:
                continue
            bd = Bond(nb, s, b.occ, False, p)
            if excl is not None and excl(i, j, nb.d):
                bd.blocked, bd.why = True, "not together (other disorder configuration)"
            res.bonds[i].append(bd)
            rev = next((n for n in env[j] if n.j == i and math.dist(n.vec, (-nb.vec[0], -nb.vec[1], -nb.vec[2])) < 0.01),
                       None)
            if rev is not None:
                bd2 = Bond(rev, s, a.occ, False, p)
                bd2.blocked, bd2.why = bd.blocked, bd.why
                res.bonds[j].append(bd2)
                touched.add(j)
    for k in set(nh4) | touched:
        res.bonds[k].sort(key=lambda bd: bd.nb.d)
        tot, weak, n_, no_h = _sum(k, res.bonds[k], settings, overrides)
        res.bvs[k], res.weak[k], res.nsum[k], res.bvs_noH[k] = tot, weak, n_, no_h
    forced = getattr(st, "forced_classes", None) or {}
    changed = []
    for j in sorted(touched):                     # the anions' classes follow their new sums (not hand-set ones)
        if j in res.cls_manual or j in forced:
            continue
        el = st.atoms[j].el
        new = classify_oxygen(res.bvs_noH[j], settings) if el == "O" else (
            classify_nitrogen(res.bvs_noH[j], settings) if el == "N" else res.cls[j])
        if new != res.cls[j]:
            changed.append("%s %s → %s" % (st.atoms[j].label, res.cls[j], new))
            res.cls[j] = res.cls_auto[j] = new
    for i in nh4:
        res.ideal[i] = 1.0                        # NH4+ as a whole: BV sum 1
    res.nh4_gr = set(nh4)
    st._nh4_gr = set(nh4)
    notes.append("Ammonium (NH4+) bonded as a whole to its anions with García-Rodríguez et al. (2000) parameters "
                 "(NH4+–O Ro 2.226, –F 2.129, –Cl 2.619 Å, b 0.37; bvparm2020 code s): %s. These bonds count for the "
                 "anions like cation bonds; the ammonium H only set the geometry (no extra H-bond valence)."
                 % ", ".join("%s BVS %.2f" % (st.atoms[i].label, res.bvs[i]) for i in sorted(nh4))
                 + (" Classes changed by these bonds: %s." % "; ".join(changed) if changed else ""))


def analyse_bv(st, params, settings, overrides=None, class_overrides=None, valence_overrides=None,
               poh_fixed=None, mix_forced=None):
    """First-pass bond-valence analysis of the structure as given.
    class_overrides: {site index: class} set by the user (replaces the BV-based class).
    valence_overrides: {site index: {'v': n} | {'avg': q} | {'mix': None or [v1, v2]}} (cations).
    With settings.mix_balance the 'mix' sites are fitted together under the restriction that the whole
    model is neutral (cations + H from the classes = anions): the average charges stay as close to each
    site's own best BV fit as the balance allows (least squares on the BV misfit).
    poh_fixed {site: x}: H fractions of partial hydroxyls (instead of the start value from the BV sum);
    mix_forced {site: q}: average charges of the 'mix' sites, already balanced (see analyse_and_search)."""
    info = {}
    res = _analyse_bv(st, params, settings, overrides, class_overrides, valence_overrides, mix_forced, info,
                      poh_fixed)
    res.mix_info = info
    if mix_forced or not (getattr(settings, "mix_balance", False) and info):
        return res
    mult = site_multiplicities(st)
    forced = {}
    for _it in range(4):                         # the classes (H count) may shift with the valences
        q = formula_charge(st, res)
        if abs(q["total"]) < 0.005:
            break
        z = q["z"]
        items = [(i, info[i][0], info[i][1], info[i][2], info[i][3], st.atoms[i].occ * mult[i] / z)
                 for i in sorted(info)]
        target = sum(it[5] * res.valence[it[0]] for it in items) - q["total"]
        forced, _ok = _balance_lsq(items, target)     # not reachable: as close as the states allow
        res = _analyse_bv(st, params, settings, overrides, class_overrides, valence_overrides, forced, {},
                          poh_fixed)
    res.mix_info = info
    q = formula_charge(st, res)
    res.notes.append(("Charge-balanced mixed-valence fit (%s): formula charge %+.3f." if abs(q["total"]) < 0.01 else
                      "⚠ Charge-balanced mixed-valence fit (%s) cannot reach neutrality within the allowed states: "
                      "formula charge %+.3f.") % (", ".join("%s %+.3f" % (st.atoms[i].label, v)
                                                            for i, v in sorted(forced.items())) or "no change",
                                                 q["total"]))
    return res


def _analyse_bv(st, params, settings, overrides=None, class_overrides=None, valence_overrides=None,
                forced=None, mixinfo=None, poh_fixed=None):
    overrides = overrides or {}
    st.excl = Exclusivity(st, settings)
    env = st.environment(settings.env_radius)
    roles = assign_roles(st, env)
    st.excl.roles = roles                        # needed while the bonds are built (cation-anion pairs)
    notes = []
    if st.sym_warning:
        notes.append(st.sym_warning)
    if getattr(st, "occ_notes", None):
        notes.append("Occupancy above 1 treated as a fully occupied site (the CIF value is kept in its own "
                     "output): " + "; ".join(st.occ_notes) + ".")
    val, how, fits, formula_avg = _assign_valences(st, env, roles, params, settings, overrides, notes)
    for i, v in (getattr(st, "forced_valences", None) or {}).items():   # subscheme: as in the parent
        if roles[i] == "cation" and i in val:
            val[i], how[i] = v, "as in the disordered parent structure"
    if valence_overrides:
        _apply_valence_overrides(st, env, roles, params, settings, overrides, val, how, valence_overrides, notes,
                                 forced, mixinfo)

    def valence_of(k):
        return val.get(k) if roles[k] == "cation" else None

    missing = set()
    res = BVResult()
    res.roles, res.valence, res.how, res.fits = roles, {}, how, fits
    res.valence_spec = dict(val)                 # int, or ('mix', v1, v2, fraction of v1)
    res.val_manual = set(valence_overrides or {})
    res.bonds, res.bvs, res.weak, res.nsum, res.cls, res.ideal = [], [], [], [], [], []
    res.bvs_noH = []
    # parent atom of every H in the CIF (nearest non-H atom within loc_max_dh)
    res.h_parent, res.n_loc = {}, [0] * len(st.atoms)
    for k, a in enumerate(st.atoms):
        if a.el != "H":
            continue
        heavy = [nb for nb in env[k] if st.atoms[nb.j].el != "H"]
        if heavy and heavy[0].d <= settings.loc_max_dh:
            res.h_parent[k] = heavy[0].j
            res.n_loc[heavy[0].j] += 1
    for i, a in enumerate(st.atoms):
        bonds = _bond_list(st, env, roles, i, valence_of, params, settings, missing)
        tot, weak, n, no_h = _sum(i, bonds, settings, overrides)
        res.bvs_noH.append(no_h)
        if roles[i] == "cation":
            v = val.get(i)
            res.valence[i] = valence_value(v) if v is not None else None   # mixed: average charge
            ideal = res.valence[i]
        else:
            v = ANION_CHARGE.get(a.el, -2)
            res.valence[i] = v
            ideal = abs(v)
        res.bonds.append(bonds)
        res.bvs.append(tot)
        res.weak.append(weak)
        res.nsum.append(n)
        res.ideal.append(ideal)
        # O classes come from the sum WITHOUT bonds to H atoms
        if roles[i] != "anion":
            res.cls.append("")
        elif a.el == "O":
            res.cls.append(classify_oxygen(no_h, settings))
        elif a.el == "N":
            res.cls.append(classify_nitrogen(no_h, settings))
        else:
            res.cls.append("")
    if missing:
        notes.append("No BV parameters for: %s (bonds skipped)." % ", ".join(sorted(missing)))

    # user-set classes
    for i, c in (getattr(st, "forced_classes", None) or {}).items():   # subscheme: classes of the parent
        if 0 <= i < len(st.atoms) and roles[i] == "anion":
            res.cls[i] = c
    res.cls_auto = list(res.cls)
    res.cls_manual = set()
    for i, c in (class_overrides or {}).items():
        if 0 <= i < len(st.atoms) and roles[i] == "anion" and c in ALL_CLASSES:
            res.cls[i] = c
            res.cls_manual.add(i)
    # ammonium N carries its charge through its four H: it is not bonded to cations. Its cation contacts
    # are dropped from both sums, and partly occupied cations closer than nh4_cation_min become
    # mutually exclusive with it (see Exclusivity)
    nh4 = {i for i in range(len(st.atoms)) if res.cls[i] == CLASS_NH4}
    st.excl.nh4, st.excl.roles = nh4, roles
    if nh4:
        affected, dropped = set(), []
        for k in range(len(st.atoms)):
            for bd in res.bonds[k]:
                j = bd.nb.j
                if (k in nh4 and roles[j] == "cation") or (roles[k] == "cation" and j in nh4):
                    bd.blocked = True
                    affected.add(k)
                    if k in nh4:
                        dropped.append("%s–%s %.2f Å" % (st.atoms[k].label, st.atoms[j].label, bd.nb.d))
        for k in affected:
            tot, weak, n_, no_h = _sum(k, res.bonds[k], settings, overrides)
            res.bvs[k], res.weak[k], res.nsum[k], res.bvs_noH[k] = tot, weak, n_, no_h
        if dropped:
            notes.append("Ammonium (NH4+) is not bonded to cations - these contacts are not summed (for either "
                         "atom): " + "; ".join(sorted(set(dropped))[:20]))
        clash = []
        env_c = st.environment(max(settings.nh4_cation_min, 0.1))
        for i in nh4:
            for nb in env_c[i]:
                if roles[nb.j] == "cation" and nb.d < settings.nh4_cation_min:
                    a, b = st.atoms[i], st.atoms[nb.j]
                    ok = a.occ < 0.999 and b.occ < 0.999 and a.occ + b.occ <= 1.05
                    clash.append("%s–%s %.2f Å (%s)" % (a.label, b.label, nb.d,
                                                       "cannot be occupied together" if ok else
                                                       "⚠ occupancies too high to exclude each other"))
        if clash:
            notes.append("Ammonium N closer than %.2f Å to a cation: %s" % (
                settings.nh4_cation_min, "; ".join(sorted(set(clash))[:20])))
    _ammonium_bonds(st, env, roles, params, settings, overrides, res, nh4, notes)

    # partial hydroxyl: starting H fraction from the BV sum without H (base + x * 1 = |V|); refined
    # after the scheme search from the F&I sum with the accepted H-bonds (see refine_partial_oh)
    res.poh_x = {i: min(1.0, max(0.0, res.ideal[i] - res.bvs_noH[i]))
                 for i in range(len(st.atoms)) if res.cls[i] == CLASS_POH}
    for i, x in (poh_fixed or {}).items():      # refined earlier (charge-balanced round, see analyse_and_search)
        if i in res.poh_x:
            res.poh_x[i] = x
    if res.cls_manual:
        notes.append("Class set by hand: " + ", ".join(
            "%s %s (BV: %s)" % (st.atoms[i].label, res.cls[i], res.cls_auto[i] or "-") for i in sorted(res.cls_manual)))

    # H count: formula vs classes; occupancy of located H vs their parent
    res.h_formula = formula_h_per_cell(st)
    res.h_classes = class_h_per_cell(st, res)
    txt = "H per unit cell implied by the D/A classes: %.2f" % res.h_classes
    if res.h_formula is not None:
        txt += "; formula: %.2f (%s H x Z = %s)" % (res.h_formula, _g(st.formula.get("H", 0)), _g(st.Z))
        if abs(res.h_formula - res.h_classes) > 0.05:
            txt = "⚠ " + txt + " - MISMATCH; check the classes (right-click a class to change it)."
    notes.append(txt)
    for k, i in res.h_parent.items():
        hk = st.atoms[k]
        if abs(hk.occ - st.atoms[i].occ) > 0.02:
            notes.append("⚠ Located %s has occupancy %.3f but its parent %s has %.3f (daughter H should share the "
                         "parent's occupancy)." % (hk.label, hk.occ, st.atoms[i].label, st.atoms[i].occ))
    ex = st.excl.any_pairs()
    if ex:
        notes.append("Sites that cannot be occupied together (split / disorder groups; their H···H contacts are "
                     "ignored): " + "; ".join("%s–%s %.2f Å" % e for e in ex[:30]) + (" ..." if len(ex) > 30 else ""))
    alt = st.excl.alt
    if alt:
        notes.append("Alternative positions of one site (never present together, never donor / acceptor of each "
                     "other, at any distance): " + "; ".join(
                         "%s / %s %.2f Å (%s)" % (st.atoms[i].label, st.atoms[j].label, d, why)
                         for (i, j), (d, why) in sorted(alt.items())))
    # partly occupied pairs that look like alternatives but carry no evidence (no free variable, no A/B labels)
    sus, env_g = [], st.environment(max(settings.group_dist, 0.1))
    part = [i for i, a in enumerate(st.atoms) if a.el != "H" and a.occ < 0.999 and roles[i] == "anion"]
    for x, i in enumerate(part):
        for j in part[x + 1:]:
            if (min(i, j), max(i, j)) in alt or st.atoms[i].el != st.atoms[j].el:
                continue
            if abs(st.atoms[i].occ + st.atoms[j].occ - 1.0) > 0.05:
                continue
            ds = [nb.d for nb in env_g[i] if nb.j == j and nb.d > 0.05]
            if ds and min(ds) >= settings.split_dist:
                sus.append("%s (%.2f) / %s (%.2f) %.2f Å" % (st.atoms[i].label, st.atoms[i].occ, st.atoms[j].label,
                                                            st.atoms[j].occ, min(ds)))
    if sus:
        notes.append("⚠ Possible alternatives NOT treated as such (occupancies add up to 1 but no shared free variable, "
                     "no A/B labels, no CIF disorder group): " + "; ".join(sus[:10]) + ". If they are one disordered "
                     "site, give them the same _atom_site_disorder_assembly with different disorder groups (PART 1 / "
                     "PART 2), or labels such as X1A / X1B.")
    mx = [(i, j) for i, j in mixed_sites(st) if roles[i] == "anion" and roles[j] == "anion"]
    if mx:
        notes.append("Mixed anion sites (one position, either atom; each counts with its own occupancy, their H "
                     "never clash): " + "; ".join("%s (%.2f, %s) / %s (%.2f, %s)" % (
                         st.atoms[i].label, st.atoms[i].occ, res.cls[i] or "-", st.atoms[j].label,
                         st.atoms[j].occ, res.cls[j] or "-") for i, j in mx))
    sp = [(i, op, d) for i, op, d in symmetry_disorder_pairs(st, settings) if st.atoms[i].el == "H"]
    if sp:
        notes.append("⚠ H disordered across a symmetry element (half-occupied H next to its own copy): " +
                     "; ".join("%s ↔ %s (%s) %.2f Å" % (st.atoms[i].label, st.atoms[i].label,
                                                         st.symop_text(op, (0, 0, 0)), d) for i, op, d in sp[:8]) +
                     ". The H-bond scheme may be ordered in a lower symmetry (e.g. a ring running either way "
                     "round): Disorder tab > 'Lower symmetry...'.")

    # vanadium alternative (Cooper et al. 2019)
    res.vanadium = []
    for i, a in enumerate(st.atoms):
        if a.el != "V" or roles[i] != "cation":
            continue
        tmp = set()
        bonds = _bond_list(st, env, roles, i, lambda k: settings.v_vc_state if k == i else None,
                           params, settings, tmp)
        vc = _sum(i, bonds, settings, overrides)[0]
        vp = settings.v_slope * vc + settings.v_intercept
        res.vanadium.append({"i": i, "label": a.label, "vc": vc, "vp": vp,
                             "frac_v4": min(1.0, max(0.0, 5.0 - vp))})
    if res.vanadium:
        mean = sum(d["vp"] for d in res.vanadium) / len(res.vanadium)
        notes.append("Vanadium (Cooper et al. 2019, Vp = %.3f*Vc %+.3f, Vc from V%d+ parameters): mean Vp = %.3f"
                     % (settings.v_slope, settings.v_intercept, settings.v_vc_state, mean)
                     + ("; formula average %.3f" % formula_avg["V"] if formula_avg.get("V") else "") + ".")
    if any(a.el == "H" for a in st.atoms):
        notes.append("H atoms are present in the CIF: 'BVS' includes their O-H bonds, 'BVS w/o H' and the "
                     "O classes do not.")
    res.notes = notes
    res.formula_avg = formula_avg
    return res


# ============================================================================
# Inferred hydrogen bonds (H not located)
# ============================================================================

class Candidate:
    pass


class Option:
    """One way a donor can supply its missing H: 2 H-bonds for a water without located H,
    1 for a hydroxyl, 1 for a water whose other H is located."""
    pass


class HAtom:
    """H atom on donor site `donor`: hypothetical (located is None) or located in the CIF
    (located = its CIF label)."""
    __slots__ = ("donor", "cand", "vec", "frac", "d_dh", "d_ha", "angle", "s_grg", "sdh_grg",
                 "s_fi", "sdh_fi", "key", "located", "weight", "conditional", "flipped",
                 "p", "primary", "branches", "flex", "pflip", "fixlen", "grg_len", "occf", "alt",
                 "accf", "comp", "compw", "base")


_SHIFTS27 = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float)


def adc_penalty(theta, s):
    """0 at/above adc_soft_angle, linear down to adc_hard_angle, quadratic below that."""
    if theta is None or theta >= s.adc_soft_angle:
        return 0.0
    if theta >= s.adc_hard_angle:
        return -s.adc_soft_w * (s.adc_soft_angle - theta)
    return -(s.adc_soft_w * (s.adc_soft_angle - s.adc_hard_angle) + s.adc_hard_w * (s.adc_hard_angle - theta) ** 2)


def pair_angle_penalty(theta, s):
    """A1-D-A2 penalty: 0 at/above pair_min_angle, quadratic to pair_soft_penalty at pair_hard_angle,
    below that pair_angle_penalty x (1 + degrees below / 10). Mirrored at the wide end: 0 up to
    pair_max_angle, quadratic to pair_soft_penalty at pair_max_hard, then pair_angle_penalty x
    (1 + degrees above / 10) - the two H-bonds would point away from the side the water's H are on."""
    if theta is None:
        return 0.0
    if theta > s.pair_max_angle:
        if theta <= s.pair_max_hard:
            x = (theta - s.pair_max_angle) / max(1e-6, s.pair_max_hard - s.pair_max_angle)
            return s.pair_soft_penalty * x * x
        return s.pair_angle_penalty * (1.0 + (theta - s.pair_max_hard) / 10.0)
    if theta >= s.pair_min_angle:
        return 0.0
    if theta >= s.pair_hard_angle:
        x = (s.pair_min_angle - theta) / max(1e-6, s.pair_min_angle - s.pair_hard_angle)
        return s.pair_soft_penalty * x * x
    return s.pair_angle_penalty * (1.0 + (s.pair_hard_angle - theta) / 10.0)


def fi_valence(d_da, s):
    """Ferraris & Ivaldi (1988) H...A bond valence from the D...A distance."""
    return (d_da / s.fi_d0) ** (-s.fi_n) + s.fi_c


def site_weights(st, s):
    """Weight of each site's BV cost in the scheme score: its occupancy (a low-occupancy position is less
    certain and less of the structure), or 1 for every site when occ_weight is off."""
    if not getattr(s, "occ_weight", True):
        return np.ones(len(st.atoms))
    return np.array([max(0.0, min(1.0, a.occ)) for a in st.atoms])


def site_penalty(x, center, band, minor, s):
    """Cost of BV sums x against a target center +- band: 0 inside the band, a minor linear
    penalty (up to s.minor_penalty) over the next `minor` vu, and a steep
    ((outside / bv_scale) ** bv_power) penalty for everything outside the band."""
    d = np.abs(np.asarray(x, dtype=float) - center)
    over = np.maximum(0.0, d - band)
    z = over * (1.0 / s.bv_scale)
    if s.bv_power == 4.0:                      # fast path for the default power
        z2 = z * z
        pen = z2 * z2
    else:
        pen = z ** s.bv_power
    minor = np.asarray(minor, dtype=float)
    if not np.any(minor > 0):
        return pen
    safe = np.where(minor > 0, minor, 1.0)
    return pen + np.where(minor > 0, s.minor_penalty * np.minimum(over / safe, 1.0), 0.0)


def place_hydrogens(avecs, r, hdh_deg):
    """H positions (vectors from the donor) at distance r. One acceptor: H on the D->A line.
    Two acceptors: H1-D-H2 = hdh_deg, oriented to minimise |H1-A1|^2 + |H2-A2|^2
    (the optimum lies in the A1-D-A2 plane and has a closed-form solution)."""
    a1 = np.array(avecs[0], dtype=float)
    if len(avecs) == 1:
        return [tuple(float(x) for x in r * a1 / np.linalg.norm(a1))]
    if len(avecs) >= 3:                       # NH4 / NH3 / H3O: ideal tetrahedral directions
        return fit_tetrahedral(avecs, r)
    a2 = np.array(avecs[1], dtype=float)
    L1, L2 = np.linalg.norm(a1), np.linalg.norm(a2)
    e1 = a1 / L1
    p = a2 - a2.dot(e1) * e1
    if np.linalg.norm(p) < 1e-6:
        t = np.array([1.0, 0.0, 0.0]) if abs(e1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        p = t - t.dot(e1) * e1
    e2 = p / np.linalg.norm(p)
    th2 = math.atan2(a2.dot(e2), a2.dot(e1))
    best = None
    for sgn in (1.0, -1.0):
        delta = sgn * math.radians(hdh_deg)
        c = L1 + L2 * complex(math.cos(delta - th2), math.sin(delta - th2))
        if best is None or abs(c) > best[0]:
            best = (abs(c), -math.atan2(c.imag, c.real), delta)
    _, phi, delta = best
    u1 = math.cos(phi) * e1 + math.sin(phi) * e2
    u2 = math.cos(phi + delta) * e1 + math.sin(phi + delta) * e2
    return [tuple(float(x) for x in r * u1), tuple(float(x) for x in r * u2)]


_TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / math.sqrt(3.0)


def fit_tetrahedral(dirs, r, weights=None):
    """H positions (vectors, length r) for 3 or 4 H of an NH4+ / NH3 / H3O+ group: the ideal tetrahedron
    (H-X-H 109.5 deg) rotated to match the given directions (acceptor vectors, or located-H vectors
    with a larger weight) as closely as possible (weighted Kabsch fit over all assignments).
    Returns one vector per input direction, in the same order."""
    from itertools import permutations
    T = np.array([np.asarray(d, dtype=float) / np.linalg.norm(d) for d in dirs])
    w = np.ones(len(T)) if weights is None else np.asarray(weights, dtype=float)
    best = None
    for perm in permutations(range(4), len(T)):
        E = _TETRA[list(perm)]
        H = (E * w[:, None]).T @ T                     # 3x3 covariance
        U, S, Vt = np.linalg.svd(H)
        dmat = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ dmat @ U.T
        res = float(np.sum(w * np.sum((E @ R.T - T) ** 2, axis=1)))
        if best is None or res < best[0]:
            best = (res, E @ R.T)
    return [tuple(float(x) for x in r * v) for v in best[1]]


def place_second_hydrogen(u_loc, a, r, hdh_deg):
    """Second H of a water whose first H (direction u_loc) is located: hdh_deg from it,
    as close as possible to acceptor vector a."""
    u = np.array(u_loc, dtype=float)
    u /= np.linalg.norm(u)
    av = np.array(a, dtype=float)
    w = av - av.dot(u) * u
    if np.linalg.norm(w) < 1e-6:
        t = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        w = t - t.dot(u) * u
    w /= np.linalg.norm(w)
    th = math.radians(hdh_deg)
    return tuple(float(x) for x in r * (math.cos(th) * u + math.sin(th) * w))


def _minv(st):
    if getattr(st, "_Minv", None) is None:
        st._Minv = np.linalg.inv(st.M)
    return st._Minv


def _make_candidate(st, bv, s, params, nb, cations):
    c = Candidate()
    c.nb, c.acceptor, c.ordinal = nb, nb.j, 0
    c.cation_angles = [(cn, angle_between(nb.vec, cn.vec)) for cn in cations]
    c.min_adc = min((ang for _, ang in c.cation_angles), default=None)
    c.cone_ok = c.min_adc is None or c.min_adc >= s.cone_angle / 2.0
    c.adc_ok = c.min_adc is None or c.min_adc > s.adc_pass_angle
    c.adc_pen = adc_penalty(c.min_adc, s)
    c.s_fi = fi_valence(nb.d, s)
    p = params.get("H", 1, st.atoms[nb.j].el)
    c.s_grg = p.valence(nb.d - s.dh_length) if p else None       # H on the D-A line
    c.passes = c.cone_ok and c.adc_ok and p is not None
    reasons = []
    if not c.cone_ok:
        reasons.append("inside cone")
    if not c.adc_ok:
        reasons.append("A-D-C <= %.0f" % s.adc_pass_angle)
    if p is None:
        reasons.append("no H-%s parameter" % st.atoms[nb.j].el)
    c.reason = ", ".join(reasons)
    return c


def _make_h(st, i, c, vec, s, params, located=None, grg_len=None):
    """H atom at D + vec. Located H: grg_len = its real D-H. Hypothetical H are placed at the geometry
    length (geom_dh_length) but their GRG bond valences use grg_len (default dh_length) along the same
    direction (the H-bond valence s_grg is evaluated at that GRG position)."""
    h = HAtom()
    h.donor, h.cand, h.vec, h.located = i, c, tuple(float(x) for x in vec), located
    h.frac = tuple(float(x) for x in np.array(st.atoms[i].f) + _minv(st) @ np.array(vec))
    h.d_dh = math.sqrt(sum(x * x for x in vec))
    p_ho = params.get("H", 1, st.atoms[i].el)
    # GRG D-H: dh_length for O; for other donors (N) the R0 of their H-X parameter (s = 1 vu)
    ref_len = (s.dh_length if st.atoms[i].el == "O" or p_ho is None else p_ho.ro) if s is not None else h.d_dh
    if located:
        grg_len = ref_len if (s is not None and getattr(s, "grg_norm_located", False)) else h.d_dh
    elif grg_len is None:
        grg_len = ref_len
    h.grg_len = float(grg_len)
    gv = [x * h.grg_len / h.d_dh for x in vec] if h.d_dh > 1e-9 else list(vec)
    h.sdh_grg = p_ho.valence(h.grg_len) if p_ho else 1.0
    if c is not None:
        av = c.nb.vec
        dv = (av[0] - vec[0], av[1] - vec[1], av[2] - vec[2])
        h.d_ha = math.sqrt(sum(x * x for x in dv))
        h.angle = angle_between((-vec[0], -vec[1], -vec[2]), dv)
        p = params.get("H", 1, st.atoms[c.acceptor].el)
        d_ha_g = math.sqrt(sum((av[k] - gv[k]) ** 2 for k in range(3)))
        h.s_grg = p.valence(d_ha_g) if p else 0.0
        h.s_fi = fi_valence(c.nb.d, s)
        mult = site_multiplicities(st)
        h.weight = st.atoms[i].occ * mult[i] / mult[c.acceptor]
        if i in getattr(st, "_nh4_gr", ()):   # ammonium: its N...X bonds already carry the bond valence
            h.s_fi = h.s_grg = 0.0
    else:
        h.d_ha = h.angle = None
        h.s_grg = h.s_fi = h.weight = 0.0
    h.sdh_fi = 1.0 - h.s_fi
    h.key = ("loc", located) if located else (i, round(vec[0], 2), round(vec[1], 2), round(vec[2], 2))
    h.conditional = False
    h.flipped = False
    h.p = 1.0                 # probability that this H-bond exists (branch of a conditional H)
    h.primary = True          # the entry that represents the H atom itself (position, count, O-H bond)
    h.branches = None         # all branch entries of a conditional located H
    h.flex = frozenset()      # sites whose inferred H never clash with this H (its disordered acceptors)
    h.pflip = 0.0             # for a flipped branch: the probability that was lost
    h.fixlen = False          # True: D-H length was refined - GRG uses it as is (no +- dh_range re-fit)
    h.occf = 1.0              # fraction of the H atom at this position (split H to disordered acceptors)
    h.alt = frozenset()       # sites that are absent whenever this (split) H is present
    h.accf = 1.0              # fraction of the cells in which its acceptor receives this H-bond (combined schemes)
    h.comp = None             # combined (disordered) scheme: the components this H belongs to (None = all)
    h.compw = None            # ... and {component: weight}
    h.base = None             # ... and its own (occf, p, accf) within a component
    return h


def _clone_h(h):
    c = HAtom()
    for k in HAtom.__slots__:
        setattr(c, k, getattr(h, k, None))
    return c


def combine_hsets(comps):
    """A 'disordered' scheme from several schemes (H sets) with weights [(hs, w), ...]: each component is
    present in a fraction w of the cells (weights normalised to sum 1). H at the same position in several
    components are merged (their weights add up); an H in all components is whole. Every H keeps the
    components it belongs to, so H of different components never clash with each other. Its occupancy
    fraction, donated and accepted H-bond valences are scaled by its total weight."""
    tot = sum(max(0.0, w) for _, w in comps) or 1.0
    compw = {k: max(0.0, w) / tot for k, (_, w) in enumerate(comps)}
    acc, order = {}, []
    for k, (hs, _w) in enumerate(comps):
        if compw[k] <= 0:
            continue
        for h in hs:
            key = (tuple(h.key), bool(h.primary))
            if key not in acc:
                acc[key] = [h, set()]
                order.append(key)
            acc[key][1].add(k)
    used = {k for k, w in compw.items() if w > 0}
    out = []
    for key in order:
        h, cs = acc[key]
        ws = sum(compw[k] for k in cs)
        c = _clone_h(h)
        c.base = (getattr(h, "occf", 1.0), h.p, getattr(h, "accf", 1.0) or 1.0)
        c.comp = None if cs >= used else frozenset(cs)
        c.compw = compw
        c.occf, c.p, c.accf = c.base[0] * ws, c.base[1] * ws, c.base[2] * ws
        if ws < 0.999:
            c.conditional = True
        if not h.located:
            c.key = ("dis",) + tuple(h.key)
        out.append(c)
    return out


def component_hsets(hs):
    """{component: H list as in that component} for a combined scheme (None if hs is not combined)."""
    compw = next((h.compw for h in hs if getattr(h, "compw", None)), None)
    if not compw:
        return None
    out = {}
    for k, w in compw.items():
        if w <= 0:
            continue
        lst = []
        for h in hs:
            if h.comp is None or k in h.comp:
                c = _clone_h(h)
                c.occf, c.p, c.accf = h.base if h.base else (1.0, 1.0, 1.0)
                c.comp = c.compw = c.base = None
                lst.append(c)
        out[k] = lst
    return out


def _carry_split(src, dst):
    """Copy the split-H attributes of src onto a rebuilt H dst (refinement)."""
    if getattr(src, "occf", 1.0) < 0.999:
        dst.conditional, dst.p, dst.occf, dst.alt = True, src.p, src.occf, src.alt
    return dst


def flip_clone(h):
    """Copy of a located-H branch that has flipped away from its (disordered) acceptor: no H-bond in
    the configurations where that acceptor is present (the lost probability is kept in pflip)."""
    c = HAtom()
    for k in HAtom.__slots__:
        setattr(c, k, getattr(h, k))
    c.cand, c.d_ha, c.angle = None, None, None
    c.s_grg = c.s_fi = c.weight = 0.0
    c.sdh_fi = 1.0
    c.flipped = True
    c.pflip, c.p = h.p, 0.0
    c.key = ("flip",) + tuple(h.key)
    return c


def mates(bv, j):
    """Site j plus every site that shares an exclusion group with it."""
    return getattr(bv, "mates", {}).get(j, {j})


def branch_flips(st, hyp_hs, located):
    """Branches of conditional located H that must flip because an inferred H on the branch's own
    acceptor would clash with that located H (< hh_red). In the cells where that acceptor is present
    the located H turns away; in the other cells it keeps its H-bond to the alternative present."""
    prim = [h for h in located if h.primary and h.conditional and h.branches]
    if not prim or not hyp_hs:
        return []
    out = []
    s_red = getattr(st, "_hh_red", 2.0)
    for L, hh, d, _, _ in h_contacts(st, prim, list(hyp_hs), cutoff=s_red, ignore_flex=True):
        for b in L.branches:
            if b.cand is not None and b.cand.acceptor == hh.donor and b not in out:
                out.append(b)
    return out


def resolve_flips(st, bv, hs, s):
    """Replace located-H branches that must flip (see branch_flips) by flipped copies."""
    st._hh_red = s.hh_red
    loc = [h for h in hs if h.located]
    hyp = [h for h in hs if not h.located]
    flip = {id(b) for b in branch_flips(st, hyp, loc)}
    return [flip_clone(h) if id(h) in flip else h for h in hs]


def located_hatoms(st, bv, params, s):
    """H atoms present in the CIF, attached to their parent anion, each with its main
    H-bond: the shortest H...A meeting the located-H criteria (or none)."""
    env = st.environment(s.env_radius)
    out = []
    for k, a in enumerate(st.atoms):
        if a.el != "H":
            continue
        i = bv.h_parent.get(k)
        if i is None or bv.roles[i] != "anion" or getattr(bv, "hrole", {}).get(i):
            continue
        nbH = min((n for n in env[i] if n.j == k), key=lambda n: n.d, default=None)
        if nbH is None or nbH.d > s.loc_max_dh:
            continue
        qual = []                         # every acceptor this H could bond to: (H...A, neighbour)
        for nb in env[i]:
            if nb.d > s.loc_max_da:
                break
            if bv.roles[nb.j] != "anion":
                continue
            ha = (nb.vec[0] - nbH.vec[0], nb.vec[1] - nbH.vec[1], nb.vec[2] - nbH.vec[2])
            d_ha = math.sqrt(sum(x * x for x in ha))
            if d_ha > s.loc_max_ha:
                continue
            if angle_between((-nbH.vec[0], -nbH.vec[1], -nbH.vec[2]), ha) < s.loc_min_angle:
                continue
            qual.append((d_ha, nb))
        best = min(qual, key=lambda q: q[0]) if qual else None
        cations = [bd.nb for bd in bv.bonds[i]
                   if bd.included and bv.roles[bd.nb.j] == "cation" and st.atoms[bd.nb.j].el != "H"]
        c = _make_candidate(st, bv, s, params, best[1], cations) if best else None
        h = _make_h(st, i, c, nbH.vec, s, params, located=a.label)
        out.append(h)
        if c is None or c.acceptor not in getattr(bv, "disordered", set()):
            continue
        # H-bond to a disordered site: the H bonds to whichever alternative position is present.
        # One branch per alternative it can reach, with that alternative's occupancy as probability.
        members = mates(bv, c.acceptor)
        h.conditional, h.p, h.flex = True, st.atoms[c.acceptor].occ, frozenset(members)
        branches = [h]
        for m in sorted(members - {c.acceptor}):
            reach = [q for q in qual if q[1].j == m]
            if not reach:
                continue
            nb = min(reach, key=lambda q: q[0])[1]
            b = _make_h(st, i, _make_candidate(st, bv, s, params, nb, cations), nbH.vec, s, params,
                        located=a.label)
            b.conditional, b.p, b.flex, b.primary = True, st.atoms[m].occ, h.flex, False
            b.key = ("loc", a.label, m)
            branches.append(b)
            out.append(b)
        for b in branches:
            b.branches = branches
    return out


def infer_hbonds(st, bv, params, s, located=()):
    """Donors (O classed water / hydroxyl, or carrying located H), their candidate acceptors
    and options for every H that is not located."""
    env = st.environment(s.env_radius)
    loc_by = {}
    for h in located:
        loc_by.setdefault(h.donor, []).append(h)
    donors = []
    hrole = getattr(bv, "hrole", {})
    for i, a in enumerate(st.atoms):
        cls = bv.cls[i]
        locs = loc_by.get(i, [])
        if cls not in DONOR_CLASSES and not locs:
            continue
        if hrole.get(i):                       # acceptor-only or excluded: not a donor
            continue
        excl = st.excl
        # cations that block H directions (A-D-C test); a cation site that is mostly vacant (total
        # occupancy of the position < adc_min_occ, e.g. Na 0.24 + Ca 0.06) does not block
        cations = [bd.nb for bd in bv.bonds[i] if bd.included and bv.roles[bd.nb.j] == "cation"
                   and st.atoms[bd.nb.j].el != "H" and not excl(i, bd.nb.j, bd.nb.d)
                   and position_occupancy(st, bd.nb.j) >= s.adc_min_occ]
        cands = []
        for nb in env[i]:
            if nb.d > s.da_max:
                break
            if nb.d < s.da_min or bv.roles[nb.j] != "anion" or excl(i, nb.j, nb.d):
                continue
            if hrole.get(nb.j) == "exclude" or bv.cls[nb.j] in NON_ACCEPTOR_CLASSES:
                continue
            if nb.d < 0.05:                    # mixed site partner at the same position
                continue
            cands.append(_make_candidate(st, bv, s, params, nb, cations))
        for k, c in enumerate(cands, 1):
            c.ordinal = k
        bykey = {c.nb.key: c for c in cands}
        for h in locs:
            if h.cand is not None and h.cand.nb.key in bykey:
                h.cand = bykey[h.cand.nb.key]
        if cls in DONOR_CLASSES:
            n_h = H_PER_CLASS[cls]             # partial hydroxyl: 1 H with fraction x (see _build_options)
        else:
            n_h = len(locs)
        d = {"i": i, "label": a.label, "cls": cls or "has located H", "bvs": bv.bvs_noH[i], "n_h": n_h,
             "cations": cations, "cands": cands, "located": locs, "n_missing": max(0, n_h - len(locs))}
        d["options"] = _build_options(st, bv, d, params, s) if d["n_missing"] else []
        d["n_impossible"] = 0
        if d["options"]:
            for o in d["options"]:
                o.impossible = _impossible_reasons(st, bv, s, o, located)
            d["n_impossible"] = sum(1 for o in d["options"] if o.impossible)
            if not s.show_severe:
                d["options"] = [o for o in d["options"] if not o.impossible]
                for n, o in enumerate(d["options"], 1):
                    o.number = n
        donors.append(d)
    return donors


def _impossible_reasons(st, bv, s, o, located):
    """Why an option is physically impossible ('' if it is not): an H...H contact below
    hh_severe within the option or with located H, or an H too close to a cation or to an anion
    other than its parent (sites that cannot coexist with the donor are ignored)."""
    reasons = []
    st._hh_red = s.hh_red
    cs = h_contacts(st, o.hs, cutoff=s.hh_severe)
    o.flips = branch_flips(st, o.hs, located) if located else []   # resolvable by flipping
    if located:
        cs = cs + h_contacts(st, o.hs, list(located), cutoff=s.hh_severe)
    cs = [c for c in cs if contact_is_geometry(st, c) and c[2] < s.hh_severe]   # not disorder-only
    if cs:
        reasons.append("H···H %.2f Å" % min(c[2] for c in cs))
    env = st.environment(s.env_radius)
    lim = max(s.h_min_cation, s.h_min_anion) + max(s.dh_length, s.geom_dh_length) + 0.1
    for h in o.hs:
        i = h.donor
        for nb in env[i]:
            if nb.d > lim:
                break
            if st.atoms[nb.j].el == "H" or st.excl(i, nb.j, nb.d):
                continue
            d = math.sqrt(sum((nb.vec[k] - h.vec[k]) ** 2 for k in range(3)))
            if bv.roles[nb.j] == "cation" and d < s.h_min_cation:
                reasons.append("H···%s %.2f Å" % (st.atoms[nb.j].label, d))
            elif bv.roles[nb.j] == "anion" and d < s.h_min_anion:
                reasons.append("H···%s %.2f Å" % (st.atoms[nb.j].label, d))
    return ", ".join(reasons)


def hs_for_choice(st, bv, d, cands, params, s):
    """H atoms of donor d for a hand-picked list of its candidate acceptors (scheme builder), placed
    like the search places them: one acceptor on the D->A line, two at the H-D-H angle, 3-4 on an ideal
    tetrahedron; a donor with located H keeps them and fits the missing H around them."""
    cands = [c for c in cands if c is not None]
    if not cands:
        return []
    i, locs = d["i"], [h for h in d["located"] if h.primary]
    r_g, hdh = s.geom_dh_length, s.hdh_angle
    if locs and d["n_h"] == 2 and len(locs) == 1:
        vecs = [place_second_hydrogen(locs[0].vec, cands[0].nb.vec, r_g, hdh)]
        cands = cands[:1]
    elif locs and d["n_h"] >= 3:
        lv = [h.vec for h in locs]
        vecs = fit_tetrahedral(lv + [c.nb.vec for c in cands], r_g, [10.0] * len(lv) + [1.0] * len(cands))[len(lv):]
    else:
        vecs = place_hydrogens([c.nb.vec for c in cands], r_g, hdh)
    hs = [_make_h(st, i, c, v, s, params) for c, v in zip(cands, vecs)]
    x = getattr(bv, "poh_x", {}).get(i) if bv.cls[i] == CLASS_POH else None
    if x is not None:
        for h in hs:
            h.occf, h.p, h.conditional = x, x, True
    return hs


def choice_problem(st, bv, d, cands, params, s, located=()):
    """Why a hand-picked set of acceptors for donor d is impossible ('' if it is not): a candidate that
    fails the angle tests, or an H...H / H...cation clash of the resulting H (as in the search)."""
    bad = [c for c in cands if c is not None and not c.passes]
    if bad:
        return "; ".join("%s fails: %s" % (st.atoms[c.acceptor].label, c.reason or "angle test") for c in bad)
    o = Option()
    o.hs = hs_for_choice(st, bv, d, cands, params, s)
    return _impossible_reasons(st, bv, s, o, list(located)) if o.hs else ""


def _coincident(cands):
    """Two acceptor positions at the same place (a mixed site, e.g. O/F): never both."""
    return any(math.dist(a.nb.vec, b.nb.vec) < 0.05 for a, b in combinations(cands, 2))


def _build_options(st, bv, d, params, s):
    i = d["i"]
    locs = d["located"]
    used = {h.cand.nb.key for h in locs if h.cand is not None}
    passing = [c for c in d["cands"] if c.passes and c.nb.key not in used]
    r_g, hdh = s.geom_dh_length, s.hdh_angle
    # split units: two passing acceptor positions that are alternatives of one disorder (cannot be
    # occupied together) - the H is then split too: one position towards each, weighted by occupancy
    pairs = []
    if getattr(s, "split_acceptors", True) and d["n_h"] <= 2:
        for c1, c2 in combinations(passing, 2):
            d12 = math.dist(c1.nb.vec, c2.nb.vec)
            if d12 > 0.05 and st.excl(c1.acceptor, c2.acceptor, d12):
                pairs.append((c1, c2))

    def split_hs(pair, vecs):
        occ = [max(1e-3, st.atoms[c.acceptor].occ) for c in pair]
        hs_ = []
        for c, v, oc, other in zip(pair, vecs, occ, pair[::-1]):
            h = _make_h(st, i, c, v, s, params)
            h.conditional, h.p = True, oc / sum(occ)
            h.occf = h.p
            h.alt = frozenset((mates(bv, other.acceptor) | {other.acceptor}) - {c.acceptor})
            h.key = tuple(h.key) + ("split",)
            hs_.append(h)
        return hs_

    opts = []
    if locs and d["n_h"] == 2 and d["n_missing"] == 1:
        u = locs[0].vec
        la = locs[0].cand.nb.vec if locs[0].cand is not None else None
        for c in passing:
            o = Option()
            o.donor, o.cands = d, (c,)
            o.hs = [_make_h(st, i, c, place_second_hydrogen(u, c.nb.vec, r_g, hdh), s, params)]
            o.angle = angle_between(la, c.nb.vec) if la is not None else None
            o.hdh = angle_between(u, o.hs[0].vec)
            opts.append(o)
        for pair in pairs:
            o = Option()
            o.donor, o.cands = d, pair
            o.hs = split_hs(pair, [place_second_hydrogen(u, c.nb.vec, r_g, hdh) for c in pair])
            o.alt_angles = [(h.occf, angle_between(la, h.cand.nb.vec) if la is not None else None) for h in o.hs]
            o.angle = o.alt_angles[0][1]
            o.hdh = angle_between(u, o.hs[0].vec)
            opts.append(o)
    elif locs and d["n_h"] >= 3 and d["n_missing"] >= 1:
        # NH4+ / NH3 / H3O+ with some H located: fit the tetrahedron to the located H (high weight)
        # and the chosen acceptors
        lv = [h.vec for h in locs if h.primary]
        lacc = [h.cand.nb.vec for h in locs if h.primary and h.cand is not None]
        k = min(d["n_missing"], len(passing))
        for combo in combinations(passing, k):
            if _coincident(combo):
                continue
            vecs = fit_tetrahedral(lv + [c.nb.vec for c in combo], r_g, [10.0] * len(lv) + [1.0] * k)[len(lv):]
            o = Option()
            o.donor, o.cands = d, combo
            o.hs = [_make_h(st, i, c, v, s, params) for c, v in zip(combo, vecs)]
            o.alt_angles = [(1.0, angle_between(a.nb.vec, b.nb.vec)) for a, b in combinations(combo, 2)] + \
                           [(1.0, angle_between(a, c.nb.vec)) for a in lacc for c in combo]
            o.angle = min((a for _w, a in o.alt_angles), default=None)
            o.hdh = None
            opts.append(o)
    else:
        k = min(d["n_missing"], len(passing))
        if k == 0:
            return []
        units = [(c,) for c in passing] + pairs
        for combo in combinations(units, k):
            n_split = sum(1 for un in combo if len(un) == 2)
            flat = [c for un in combo for c in un]
            if n_split > 1 or len({c.nb.key for c in flat}) < len(flat) or (n_split == 0 and _coincident(flat)):
                continue
            o = Option()
            o.donor, o.cands = d, tuple(flat)
            if n_split == 0:
                vecs = place_hydrogens([c.nb.vec for c in flat], r_g, hdh)
                o.hs = [_make_h(st, i, c, v, s, params) for c, v in zip(flat, vecs)]
                o.angle = angle_between(flat[0].nb.vec, flat[1].nb.vec) if k == 2 else None
                o.hdh = angle_between(o.hs[0].vec, o.hs[1].vec) if k == 2 else None
                if k >= 3:                          # every acceptor pair of a tetrahedral group
                    o.alt_angles = [(1.0, angle_between(a.nb.vec, b.nb.vec)) for a, b in combinations(flat, 2)]
                    o.angle = min(a for _w, a in o.alt_angles)
                    o.hdh = None
            elif k == 1:                            # hydroxyl: its H split over the two alternatives
                pair = combo[0]
                o.hs = split_hs(pair, [place_hydrogens([c.nb.vec], r_g, hdh)[0] for c in pair])
                o.angle, o.hdh = None, None
            else:                                   # water: one whole H + one H split over two alternatives
                one = next(un for un in combo if len(un) == 1)[0]
                pair = next(un for un in combo if len(un) == 2)
                v1 = [place_hydrogens([one.nb.vec, c.nb.vec], r_g, hdh)[0] for c in pair]
                u1 = np.array(v1[0]) + np.array(v1[1])
                u1 = u1 / np.linalg.norm(u1) * r_g     # the whole H: between its two placements
                h1 = _make_h(st, i, one, u1, s, params)
                o.hs = [h1] + split_hs(pair, [place_second_hydrogen(tuple(u1), c.nb.vec, r_g, hdh)
                                              for c in pair])
                o.cands = (one,) + tuple(pair)
                o.alt_angles = [(h.occf, angle_between(one.nb.vec, h.cand.nb.vec)) for h in o.hs[1:]]
                o.angle = o.alt_angles[0][1]
                o.hdh = angle_between(o.hs[0].vec, o.hs[1].vec)
            opts.append(o)
    x_poh = getattr(bv, "poh_x", {}).get(i) if bv.cls[i] == CLASS_POH else None
    if x_poh is not None:                       # partial hydroxyl: only a fraction x of the H is there
        for o in opts:
            for h in o.hs:
                h.occf *= x_poh
                h.p *= x_poh
                h.conditional = True
    for o in opts:
        alt = getattr(o, "alt_angles", None)
        if alt:
            o.angle_pen = sum(w * pair_angle_penalty(a, s) for w, a in alt)
            ang_ok = all(a is None or s.pair_hard_angle <= a <= s.pair_max_hard for _w, a in alt)
        else:
            o.angle_pen = pair_angle_penalty(o.angle, s)
            ang_ok = o.angle is None or s.pair_hard_angle <= o.angle <= s.pair_max_hard
        o.adc_pen = sum(h.cand.adc_pen * getattr(h, "occf", 1.0) for h in o.hs)
        o.geom = o.angle_pen + o.adc_pen
        o.incomplete = sum(getattr(h, "occf", 1.0) for h in o.hs) < d["n_missing"] * (x_poh or 1.0) - 0.01
        o.donorF = d["bvs"] - sum(h.s_fi * h.p for h in list(locs) + o.hs)  # before accepted H-bonds
        o.local = o.geom - 0.01 * sum(c.nb.d * w for c, w in
                                      ((h.cand, getattr(h, "occf", 1.0)) for h in o.hs))  # shorter H-bonds
        o.passes = ang_ok and not o.incomplete
    opts.sort(key=lambda o: -o.local)
    for n, o in enumerate(opts, 1):
        o.number = n
    return opts


# ============================================================================
# Bond valence including H, H...H contacts, scheme search
# ============================================================================

def site_multiplicities(st):
    """Number of symmetry-equivalent positions of each site in the unit cell."""
    if getattr(st, "_mult", None) is not None:
        return st._mult
    R, T = _sym_arrays(st)
    out = []
    for a in st.atoms:
        g = np.round((R @ np.array(a.f) + T) % 1.0, 3) % 1.0
        out.append(len({tuple(r) for r in g.tolist()}))
    st._mult = out
    return out


def position_multiplicity(st, frac):
    R, T = _sym_arrays(st)
    g = np.round((R @ np.array(frac) + T) % 1.0, 3) % 1.0
    return len({tuple(r) for r in g.tolist()})


def h_symmetry_copies(st, h):
    """How many symmetry copies of H `h` sit on the same donor atom (1 = no extra copies)."""
    mult = site_multiplicities(st)
    return max(1, round(position_multiplicity(st, h.frac) / mult[h.donor]))


def site_stabilizer(st, i, tol=2e-3):
    """[(R, t)] fractional operations (incl. lattice shift in t) that map site i onto itself, identity excluded."""
    f = np.array(st.atoms[i].f, dtype=float)
    out = []
    for R, t in st.symops:
        R, t = np.array(R, dtype=float), np.array(t, dtype=float)
        g = R @ f + t
        sh = np.round(f - g)
        if np.all(np.abs(g + sh - f) < tol) and not (np.allclose(R, np.eye(3)) and np.allclose(t + sh, 0)):
            out.append((R, t + sh))
    return out


def h_site_images(st, hs, tol=0.30):
    """Inferred H that are symmetry images of another H of the same scheme on the same donor, through an
    operation of the donor's own site symmetry (e.g. the two H of a water on a 2-fold axis, H-bonded to
    two symmetry-equivalent acceptors): the H set is then ORDERED on that site - one H per orbit is the
    independent atom (full occupancy), the others are generated by the symmetry.
    Returns {id(image H): (representative H, R_cart)} - R_cart maps the representative's D-H vector onto
    the image's. Images are matched by their H-bond (the acceptor's position must map exactly, within
    0.05 A) and their H positions must agree within `tol` A (refinement may have moved them a little)."""
    by = {}
    for h in hs:
        if not h.located and h.primary and h.cand is not None:
            by.setdefault(h.donor, []).append(h)
    out = {}
    for i, lst in by.items():
        if len(lst) < 2:
            continue
        stab = site_stabilizer(st, i)
        if not stab:
            continue
        M, Mi = st.M, _minv(st)
        done = set()
        for r in lst:
            if id(r) in out or id(r) in done:
                continue
            for R, _t in stab:
                Rc = M @ R @ Mi
                av = Rc @ np.array(r.cand.nb.vec, dtype=float)
                hv = Rc @ np.array(r.vec, dtype=float)
                for h in lst:
                    if h is r or id(h) in out or id(h) in done:
                        continue
                    if h.cand.acceptor != r.cand.acceptor or abs(getattr(h, "occf", 1.0) - getattr(r, "occf", 1.0)) > 1e-3 \
                            or getattr(h, "comp", None) != getattr(r, "comp", None):
                        continue
                    if np.linalg.norm(av - np.array(h.cand.nb.vec)) < 0.05 and \
                            np.linalg.norm(hv - np.array(h.vec)) < tol:
                        out[id(h)] = (r, Rc)
            done.add(id(r))
    # keep only complete orbits: every symmetry copy of the representative must be in the scheme
    reps = {}
    for k, (r, _) in out.items():
        reps.setdefault(id(r), [r, 0])[1] += 1
    bad = {rid for rid, (r, n) in reps.items() if n + 1 != h_symmetry_copies(st, r)}
    return {k: v for k, v in out.items() if id(v[0]) not in bad}


def _sym_arrays(st):
    return (np.array([op[0] for op in st.symops], dtype=float),
            np.array([op[1] for op in st.symops], dtype=float))


def site_targets(st, bv, s):
    """Per-site targets. F&I sums exclude each O's own O-H bonds (Σ = cations + accepted -
    donated), so water aims at 0 and hydroxyl at 1 (+- donor_band, minor penalty over the
    next donor_minor vu); other anions aim at |valence|. GRG sums include the O-H bonds,
    so every anion aims at |valence|."""
    n = len(st.atoms)
    hrole = getattr(bv, "hrole", {})
    gr = getattr(bv, "nh4_gr", set())           # NH4+ bonded as a whole: a cation for the scoring
    anion = np.array([r == "anion" and i not in hrole and i not in gr for i, r in enumerate(bv.roles)])
    cF, bF, mF, cG = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        ideal = bv.ideal[i] or 0.0
        cG[i] = cF[i] = ideal
        cls = bv.cls[i]
        if cls in DONOR_CLASSES:              # own X-H bonds not counted: |V| - number of H
            cF[i] = ideal - class_h(bv, i)    # water 0, OH 1, NH4 -1, NH3 0, NH2 1, H3O -1, partial OH 2 - x
        elif not s.ignore_located and bv.n_loc[i] > 0:
            cF[i] = max(0.0, ideal - bv.n_loc[i])
        else:
            continue
        bF[i], mF[i] = s.donor_band, s.donor_minor
        if cls in NON_ACCEPTOR_CLASSES:       # NH4+ / NH3 / H3O+: F&I gives each X-H ~0.85 vu, so the centre
            bF[i] = s.group_band              # always looks overbonded - judge it with a wider band
    return anion, cF, bF, mF, cG


def _grg_optimize(st, hs, base, s, params, target, cache=None):
    """GRG: each D-H may change by up to +- dh_range (step dh_step). For every
    donor, pick the D-H lengths that bring its BV sum (incl. accepted H-bonds) closest to the
    target; iterate because accepted H-bonds depend on the other donors' choices."""
    n = len(st.atoms)
    if not hs:
        return [], [], []
    r_, st_ = max(0.0, s.dh_range), max(1e-3, s.dh_step)
    steps = np.arange(-r_, r_ + 1e-9, st_)
    steps = steps[np.argsort(np.abs(steps), kind="stable")]     # smallest change first: wins ties
    zero = np.zeros(1)
    grids = []
    for h in hs:
        stp = zero if getattr(h, "fixlen", False) else steps    # refined D-H lengths stay as they are
        key = (h.donor, h.vec, h.cand.nb.key if h.cand is not None else None, len(stp),
               round(getattr(h, "grg_len", 0.0), 5))
        if cache is not None and key in cache:
            grids.append(cache[key])
            continue
        u = np.array(h.vec) / h.d_dh
        L = getattr(h, "grg_len", h.d_dh) + stp          # GRG works at its own D-H (0.918 for hypothetical H)
        p_ho = params.get("H", 1, st.atoms[h.donor].el)
        sdh = np.exp((p_ho.ro - L) / p_ho.b) if p_ho else np.ones_like(L)
        if h.cand is not None:
            a = np.array(h.cand.nb.vec)
            dha = np.linalg.norm(a[None, :] - L[:, None] * u[None, :], axis=1)
            pA = params.get("H", 1, st.atoms[h.cand.acceptor].el)
            sha = np.exp((pA.ro - dha) / pA.b) if pA else np.zeros_like(L)
        else:
            sha = np.zeros_like(L)
        if h.donor in getattr(st, "_nh4_gr", ()):       # ammonium: N...X bonds carry it (García-Rodríguez)
            sdh, sha = np.zeros_like(L), np.zeros_like(L)
        g = (L, sdh, sha)
        if cache is not None:
            cache[key] = g
        grids.append(g)
    idx = [0] * len(hs)
    # extra H-bond branches of a located H share its position: they follow the primary's length
    prim_of = {h.located: k for k, h in enumerate(hs) if h.located and h.primary}
    lead = [k if (h.primary or h.located not in prim_of) else prim_of[h.located] for k, h in enumerate(hs)]
    by = {}
    for k, h in enumerate(hs):
        if lead[k] == k:
            by.setdefault(h.donor, []).append(k)
    for _ in range(8):
        acc = np.zeros(n)
        for k, h in enumerate(hs):
            if h.cand is not None:
                acc[h.cand.acceptor] += grids[k][2][idx[lead[k]]] * h.weight * (getattr(h, "accf", 1.0) or 1.0)
        changed = False
        for i, ks in by.items():
            fixed = base[i] + acc[i]
            w = [getattr(hs[k], "occf", 1.0) for k in ks]       # split H count with their fraction
            if len(ks) == 1:
                new = [int(np.argmin(np.abs(fixed + w[0] * grids[ks[0]][1] - target[i])))]
            elif len(ks) == 2:
                f = np.abs(fixed + w[0] * grids[ks[0]][1][:, None] + w[1] * grids[ks[1]][1][None, :] - target[i])
                a0, a1 = np.unravel_index(np.argmin(f), f.shape)
                new = [int(a0), int(a1)]
            else:
                new = [idx[k] for k in ks]
                for m, k in enumerate(ks):
                    other = sum(w[z] * grids[q][1][new[z]] for z, q in enumerate(ks) if z != m)
                    new[m] = int(np.argmin(np.abs(fixed + other + w[m] * grids[k][1] - target[i])))
            for k, v in zip(ks, new):
                if idx[k] != v:
                    idx[k] = v
                    changed = True
        if not changed:
            break
    idx = [idx[lead[k]] for k in range(len(hs))]
    return ([float(grids[k][0][idx[k]]) for k in range(len(hs))],
            [float(grids[k][1][idx[k]]) for k in range(len(hs))],
            [float(grids[k][2][idx[k]]) for k in range(len(hs))])


def bv_with_h(st, bv, hs, s, params, cache=None):
    """BV sums with H atoms `hs` (located and/or hypothetical), two ways:
    F&I  - Ferraris & Ivaldi (1988) s(H...A) from D...A; Σ = cations + accepted - donated
           (an O's own O-H bonds are not counted);
    GRG  - Gagne & Hawthorne (2015) H-O parameters for O-H and H...A, D-H lengths optimised
           within dh_range; Σ = cations + O-H + accepted.
    Contributions to an acceptor are weighted by donor occupancy x mult(donor)/mult(acceptor)."""
    n = len(st.atoms)
    base = np.array(bv.bvs_noH, dtype=float)
    anion, cF, bF, mF, cG = site_targets(st, bv, s)
    r = {"base": base, "anion": anion, "cF": cF, "bF": bF, "mF": mF, "cG": cG}
    F_don, F_acc = np.zeros(n), np.zeros(n)
    for h in hs:
        if h.cand is None:
            continue
        F_don[h.donor] += h.s_fi * h.p          # a conditional H donates only when its acceptor is there
        F_acc[h.cand.acceptor] += h.s_fi * h.weight * (getattr(h, "accf", 1.0) or 1.0)
    r["F_don"], r["F_acc"] = F_don, F_acc
    r["totF"] = base - F_don + F_acc
    L, sdh, sha = _grg_optimize(st, hs, base, s, params, cG, cache)
    G_dh, G_acc = np.zeros(n), np.zeros(n)
    for k, h in enumerate(hs):
        if h.primary:
            G_dh[h.donor] += sdh[k] * getattr(h, "occf", 1.0)     # one O-H bond per H atom
        if h.cand is not None:
            G_acc[h.cand.acceptor] += sha[k] * h.weight * (getattr(h, "accf", 1.0) or 1.0)
    r["G_dh"], r["G_acc"] = G_dh, G_acc
    r["totG"] = base + G_dh + G_acc
    r["L"], r["sdh"], r["sha"] = L, sdh, sha
    r["hpos"] = {id(h): k for k, h in enumerate(hs)}
    r["devF"] = r["totF"] - cF
    r["devG"] = r["totG"] - cG
    r["outF"] = np.maximum(0.0, np.abs(r["devF"]) - bF)
    w = site_weights(st, s)
    r["penF"] = float((site_penalty(r["totF"][anion], cF[anion], bF[anion], mF[anion], s) * w[anion]).sum())
    r["penG"] = float((site_penalty(r["totG"][anion], cG[anion], 0.0, 0.0, s) * w[anion]).sum())
    return r


def h_contacts(st, hs1, hs2=None, cutoff=2.1, ignore_flex=False):
    """H...H contacts shorter than cutoff between H atoms on *different* donor atoms
    (symmetry images included). hs2=None: within hs1 (each contact once); otherwise hs1 x hs2.
    Only the entry that represents each H atom is used (not the extra H-bond branches of a
    conditional located H). A conditional located H never clashes with inferred H on the disordered
    sites it bonds to (in each cell only one arrangement exists) unless ignore_flex.
    Returns (h, h_other, distance, op, shift)."""
    hs1 = [h for h in hs1 if getattr(h, "primary", True)]
    if hs2 is not None:
        hs2 = [h for h in hs2 if getattr(h, "primary", True)]
    within = hs2 is None
    hsB = hs1 if within else hs2
    if not hs1 or not hsB:
        return []
    excl = getattr(st, "excl", None)
    R, T = _sym_arrays(st)
    F = np.array([a.f for a in st.atoms], dtype=float)
    fb = np.array([h.frac for h in hsB])
    fdb = F[[h.donor for h in hsB]]
    G = np.einsum("oij,bj->obi", R, fb) + T[:, None, :]
    GD = np.einsum("oij,bj->obi", R, fdb) + T[:, None, :]
    MT = st.M.T
    out, seen = [], set()
    for ka, ha in enumerate(hs1):
        d0 = G - np.array(ha.frac)
        base = -np.round(d0)
        C = ((d0 + base)[:, :, None, :] + _SHIFTS27[None, None, :, :]) @ MT
        dist = np.sqrt((C * C).sum(-1))
        for o, b, sh in zip(*np.nonzero(dist < cutoff)):
            if within and b < ka:
                continue
            bse, dval = base[o, b], float(dist[o, b, sh])
            dpos = GD[o, b] + bse + _SHIFTS27[sh] - F[ha.donor]
            dd = float(np.linalg.norm(st.M @ dpos))
            if dd < 0.1:                                # both H on the same donor atom
                continue
            if excl is not None and excl(ha.donor, hsB[b].donor, dd):
                continue                                # donors cannot be present together
            if excl is not None and _h_occ_exclusive(st, ha, hsB[b]):
                continue
            hb_ = hsB[b]
            ca, cb = getattr(ha, "comp", None), getattr(hb_, "comp", None)
            if ca is not None and cb is not None and not (ca & cb):
                continue                                # different components of a combined scheme
            alt_a, alt_b = getattr(ha, "alt", ()), getattr(hb_, "alt", ())
            if alt_a or alt_b:                          # split H: present only with its own acceptor
                if hb_.donor in alt_a or ha.donor in alt_b or \
                        (ha.cand is not None and ha.cand.acceptor in alt_b) or \
                        (hb_.cand is not None and hb_.cand.acceptor in alt_a):
                    continue
            if not ignore_flex and ((ha.located and not hb_.located and hb_.donor in ha.flex) or
                                    (hb_.located and not ha.located and ha.donor in hb_.flex)):
                continue                                # config-dependent located H: no clash
            d = dval
            if within and b == ka:
                key = (ka, round(d, 3))
                if key in seen:
                    continue
                seen.add(key)
            shift = tuple(int(round(x)) for x in bse + _SHIFTS27[sh])
            out.append((ha, hsB[b], d, int(o), shift))
    return out


def _h_occ_exclusive(st, ha, hb):
    """Two located H that are themselves split partners (both partly occupied, occupancies summing
    to <= 1) cannot both be present."""
    if not (ha.located and hb.located):
        return False
    a = next((x for x in st.atoms if x.label == ha.located), None)
    b = next((x for x in st.atoms if x.label == hb.located), None)
    return a is not None and b is not None and a.occ < 0.999 and b.occ < 0.999 and a.occ + b.occ <= 1.05


def _donor_disordered(st, i):
    return i in getattr(st, "_disordered", set()) or st.atoms[i].occ < 0.999


def contact_is_geometry(st, c):
    """An H...H contact is a geometry violation unless both donors are disordered and their
    occupancies sum to <= 1 (they may never be present together) - then it is only a donor conflict."""
    a, b = c[0].donor, c[1].donor
    if _donor_disordered(st, a) and _donor_disordered(st, b):
        return st.atoms[a].occ + st.atoms[b].occ > 1.0 + 1e-6
    return True


def hh_penalty(d, s):
    """Sliding scale: 0 at/above hh_mod, quadratic to hh_mod_penalty at hh_red, then hh_red_penalty
    growing by another hh_red_penalty for every 0.1 A further below hh_red."""
    if d >= s.hh_mod:
        return 0.0
    if d >= s.hh_red:
        x = (s.hh_mod - d) / max(1e-6, s.hh_mod - s.hh_red)
        return s.hh_mod_penalty * x * x
    return s.hh_red_penalty * (1.0 + (s.hh_red - d) / 0.1)


def eff_dist(c, s):
    """Distance used for the H...H PENALTY only: the real distance plus the optional h_flex for every
    hypothetical (not located) H in it (default 0). A contact that is red at its real distance is never
    penalised less than one just inside the red limit."""
    n_hyp = (0 if c[0].located else 1) + (0 if c[1].located else 1)
    d = c[2] + s.h_flex * n_hyp
    if c[2] < s.hh_red:
        d = min(d, s.hh_red - 1e-6)
    return d


def count_contacts(contacts, s, st=None):
    """(red, moderate, severe) counts of GEOMETRY contacts at their REAL distance: red < hh_red,
    moderate hh_red..hh_mod, severe < hh_severe (severe contacts are also red)."""
    if st is not None:
        contacts = [c for c in contacts if contact_is_geometry(st, c)]
    ds = [c[2] for c in contacts]
    red = sum(1 for d in ds if d < s.hh_red)
    mod = sum(1 for d in ds if s.hh_red <= d < s.hh_mod)
    sev = sum(1 for d in ds if d < s.hh_severe)
    return red, mod, sev


def contacts_score(st, contacts, s):
    """(red, mod, sev, n_disorder_conflicts, penalty) for a list of H...H contacts."""
    geo = [c for c in contacts if contact_is_geometry(st, c)]
    dis = sum(1 for c in contacts if not contact_is_geometry(st, c) and c[2] < s.hh_mod)
    red, mod, sev = count_contacts(geo, s)
    pen = sum(hh_penalty(eff_dist(c, s), s) for c in geo) + dis * s.dis_conflict_penalty
    return red, mod, sev, dis, pen


# --- water acceptor cone ----------------------------------------------------------
def _water_cone(st, w_hs, s):
    """Forbidden cone of a water from its two H (vectors from O): axis = bisector of the O-H
    directions, apex at O, base diameter = acc_cone_factor x H...H at the height of the H.
    Returns (axis unit vector, half-angle in degrees) or None."""
    if len(w_hs) < 2:
        return None
    v1, v2 = np.array(w_hs[0].vec), np.array(w_hs[1].vec)
    u1, u2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    ax = u1 + u2
    if np.linalg.norm(ax) < 1e-6:
        return None
    ax /= np.linalg.norm(ax)
    height = 0.5 * (v1.dot(ax) + v2.dot(ax))
    if height <= 1e-6:
        return None
    radius = 0.5 * s.acc_cone_factor * np.linalg.norm(v1 - v2)
    return ax, math.degrees(math.atan2(radius, height))


def cone_check(st, h_in, cone):
    """Angle (deg) between the direction from the acceptor water to the incoming H and the water's
    cone axis, and whether it lies inside the cone (H-bond arriving between the water's own H)."""
    ax, half = cone
    nb = h_in.cand.nb
    R = np.array(st.symops[nb.op][0], dtype=float)
    Rc = st.M @ R @ _minv(st)                  # rotation of that acceptor image in Cartesian space
    axi = Rc @ ax
    v = np.array(h_in.vec) - np.array(nb.vec)  # acceptor image -> incoming H
    ang = angle_between(v, axi)
    return ang, ang < half


def _water_like(lst):
    """Two H on one donor (a water, NH2 ...; one of them may be split): the acceptor cone applies."""
    whole = sum(1 for h in lst if getattr(h, "occf", 1.0) >= 0.999)
    split = len(lst) - whole
    return whole == 2 or (whole == 1 and split >= 1)


def cone_violations(st, hs, s):
    """[(incoming H, acceptor water site, angle, half-angle)] for H-bonds that arrive inside the
    forbidden cone of a water that carries two H in this set."""
    waters = {}
    for h in hs:
        if h.primary:
            waters.setdefault(h.donor, []).append(h)
    # a water with one H split over disordered acceptors: use its whole H + the first split position
    cones = {i: _water_cone(st, sorted(lst, key=lambda h: getattr(h, "occf", 1.0) < 0.999)[:2], s)
             for i, lst in waters.items() if _water_like(lst)}
    out = []
    for h in hs:
        if h.cand is None:
            continue
        cone = cones.get(h.cand.acceptor)
        if cone is None:
            continue
        ang, inside = cone_check(st, h, cone)
        if inside:
            out.append((h, h.cand.acceptor, ang, cone[1]))
    return out


class SchemeResult:
    pass


def geom_penalty(hs, s):
    """Angle penalties of a set of H: A-D-C penalty of every H-bond, plus the A1-D-A2
    penalty for a donor whose first two H-bonds subtend less than pair_min_angle."""
    by = {}
    for h in hs:
        if h.cand is not None and h.primary:
            by.setdefault(h.donor, []).append(h)
    pen = sum(s.flip_penalty * h.pflip for h in hs if getattr(h, "flipped", False))
    for lst in by.values():
        pen += sum(h.cand.adc_pen * getattr(h, "occf", 1.0) for h in lst)
        whole = [h for h in lst if getattr(h, "occf", 1.0) >= 0.999]
        split = [h for h in lst if getattr(h, "occf", 1.0) < 0.999]
        if split and whole:                     # water with one H split over disordered acceptors
            w0 = sorted(whole, key=lambda h: (h.located is None, h.cand.ordinal))[0]
            pen += sum(h.occf * pair_angle_penalty(angle_between(w0.cand.nb.vec, h.cand.nb.vec), s)
                       for h in split)
        elif len(lst) == 2 and not split:
            pen += pair_angle_penalty(angle_between(lst[0].cand.nb.vec, lst[1].cand.nb.vec), s)
        elif len(lst) >= 3 and not split:       # NH4+ / NH3 / H3O+: every pair of its H-bonds
            pen += sum(pair_angle_penalty(angle_between(a.cand.nb.vec, b.cand.nb.vec), s)
                       for a, b in combinations(lst, 2))
    return pen


def evaluate_hset(st, bv, hs, s, params, cache=None):
    """Score any set of H atoms (a scheme, a refined scheme, the H in the CIF) the same way the
    scheme search does: total = BV score (rank_by) + angle penalties + H...H penalties.
    A combined (disordered) scheme: BV sums from all its H with their weights; angle, H...H and cone
    penalties = weight-averaged over its components (red / moderate / ... counts: the largest)."""
    comps = component_hsets(hs) if hs else None
    if comps:
        compw = next(h.compw for h in hs if getattr(h, "compw", None))
        evs = {k: evaluate_hset(st, bv, lst, s, params, cache) for k, lst in comps.items()}
        r = bv_with_h(st, bv, hs, s, params, cache)
        an = r["anion"]
        out = {"bvF": -r["penF"], "bvG": -r["penG"], "r": r, "cones": [],
               "geom": sum(compw[k] * e["geom"] for k, e in evs.items()),
               "hh": sum(compw[k] * e["hh"] for k, e in evs.items()),
               "contacts": h_contacts(st, hs, cutoff=s.hh_mod),
               "maxF": float(r["outF"][an].max()) if an.any() else 0.0,
               "maxG": float(np.abs(r["devG"][an]).max()) if an.any() else 0.0,
               "components": {k: (compw[k], e) for k, e in evs.items()}}
        for key in ("red", "mod", "sev", "dis", "cone"):
            out[key] = max(int(e[key]) for e in evs.values())
        bvs = {"fi": out["bvF"], "grg": out["bvG"]}.get(s.rank_by, 0.5 * (out["bvF"] + out["bvG"]))
        out["total"] = s.score_base + s.w_bv * bvs + out["geom"] + out["hh"]
        out["h_cell"] = h_per_cell(st, hs)
        return out
    r = bv_with_h(st, bv, hs, s, params, cache)
    an = r["anion"]
    cs = h_contacts(st, hs, cutoff=s.hh_mod) if hs else []
    red, mod, sev, dis, hpen = contacts_score(st, cs, s)
    cones = cone_violations(st, hs, s)
    out = {"bvF": -r["penF"], "bvG": -r["penG"], "geom": geom_penalty(hs, s),
           "red": red, "mod": mod, "sev": sev, "dis": dis, "cone": len(cones), "cones": cones,
           "contacts": cs, "r": r,
           "hh": hpen + len(cones) * s.hh_red_penalty,
           "maxF": float(r["outF"][an].max()) if an.any() else 0.0,
           "maxG": float(np.abs(r["devG"][an]).max()) if an.any() else 0.0}
    bvs = {"fi": out["bvF"], "grg": out["bvG"]}.get(s.rank_by, 0.5 * (out["bvF"] + out["bvG"]))
    out["total"] = s.score_base + s.w_bv * bvs + out["geom"] + out["hh"]
    out["h_cell"] = h_per_cell(st, hs)
    return out


def h_count_text(st, bv, n):
    """'H/cell 36.00 (formula 36.00)' with a warning mark when they differ."""
    f = getattr(bv, "h_formula", None)
    if f is None:
        return "H/cell %.2f (formula: n/a)" % n
    ex = getattr(bv, "h_excluded", 0.0)
    if ex > 0.005:
        txt = "H/cell %.2f + %.2f on acceptor-only/excluded sites (formula %.2f)" % (n, ex, f)
        return txt if abs(n + ex - f) <= 0.05 else "⚠ " + txt
    return ("H/cell %.2f (formula %.2f)" % (n, f)) if abs(n - f) <= 0.05 else \
        ("⚠ H/cell %.2f ≠ formula %.2f" % (n, f))


def _balance_poh_and_mix(st, bv, x_free, x_slope):
    """Partial-hydroxyl H fractions x (and the average charges of 'mixed' valence sites, when there are any)
    under the restriction that the model is neutral: least squares on the BV misfit - each x as close to its
    own F&I value, each mixed site as close to its own best BV fit, as the balance allows. Returns the new
    {site: x}; stores the mixed-site charges wanted in bv.mix_target (applied by analyse_and_search, which
    re-runs the bond valences with them) and whether neutrality is reachable in bv.balance_reached."""
    q = formula_charge(st, bv)
    z, mult = q["z"], site_multiplicities(st)
    items, cur = [], 0.0
    for i, (qs_, a, lo, hi) in (getattr(bv, "mix_info", None) or {}).items():
        m = st.atoms[i].occ * mult[i] / z
        items.append((("q", i), qs_, a, lo, hi, m))
        cur += m * bv.valence[i]
    for i, x in x_free.items():
        m = st.atoms[i].occ * mult[i] / z
        items.append((("x", i), x, x_slope[i], 0.0, 1.0, m))
        cur += m * bv.poh_x.get(i, 0.5)
    vals, ok = _balance_lsq(items, cur - q["total"])
    bv.mix_target = {k[1]: v for k, v in vals.items() if k[0] == "q"}
    bv.balance_reached = ok
    return {k[1]: v for k, v in vals.items() if k[0] == "x"}


def _balance_lsq(items, target):
    """items [(key, centre, slope, lo, hi, m)]: minimise sum slope^2 (v - centre)^2 subject to sum m v = target
    and lo <= v <= hi. Returns ({key: v}, reachable)."""
    def vals(lam):
        return {k: min(hi, max(lo, c + lam * m / (a * a))) for k, c, a, lo, hi, m in items}

    def tot(lam):
        v = vals(lam)
        return sum(m * v[k] for k, _c, _a, _lo, _hi, m in items)
    lo_, hi_ = -1e4, 1e4
    if tot(hi_) < target - 1e-6:
        return vals(hi_), False
    if tot(lo_) > target + 1e-6:
        return vals(lo_), False
    for _ in range(100):
        mid = 0.5 * (lo_ + hi_)
        if tot(mid) < target:
            lo_ = mid
        else:
            hi_ = mid
    return vals(0.5 * (lo_ + hi_)), True


def analyse_and_search(st, params, s, overrides=None, class_overrides=None, valence_overrides=None,
                       role_overrides=None, progress=None):
    """Bond valences, roles, located H, candidates and the scheme search. With s.mix_balance and partial
    hydroxyls, their H fractions and the mixed-valence sites are fitted together under overall charge
    balance: the search's charges for the mixed sites are applied to the bond valences and everything is
    repeated until they agree. Returns (bv, located H used, donors, schemes, all located H)."""
    poh_fixed = mix_forced = None
    for _round in range(4):
        bv = analyse_bv(st, params, s, overrides, class_overrides, valence_overrides, poh_fixed, mix_forced)
        apply_site_roles(st, bv, s, role_overrides)
        loc_all = located_hatoms(st, bv, params, s)
        loc = [] if s.ignore_located else loc_all
        donors = infer_hbonds(st, bv, params, s, loc)
        res = search_schemes(st, bv, donors, loc, s, params, progress)
        tgt = getattr(bv, "mix_target", None)
        if not getattr(s, "mix_balance", False) or not tgt or \
                all(abs(tgt[i] - bv.valence[i]) < 0.002 for i in tgt):
            break
        mix_forced, poh_fixed = dict(tgt), dict(bv.poh_x)
    if getattr(s, "mix_balance", False) and getattr(bv, "poh_x", None) and \
            any(bv.cls[i] == CLASS_POH for i in bv.poh_x):
        q = formula_charge(st, bv)
        bv.notes.append(("Charge balance with the partial hydroxyl(s)%s: formula charge %+.3f." if
                         getattr(bv, "balance_reached", True) else
                         "⚠ Charge balance with the partial hydroxyl(s)%s cannot be reached within x = 0-1 (and "
                         "the allowed states): formula charge %+.3f.") % (
            " and the mixed-valence sites" if getattr(bv, "mix_target", None) else "", q["total"]))
    return bv, loc, donors, res, loc_all


def search_schemes(st, bv, donors, located, s, params, progress=None):
    """Scheme search (see _search_schemes). Partial hydroxyls: their H fraction x is refined from
    the best scheme - F&I sum of the site = cations + accepted - x * donated must equal |V| - x, so
    x = (|V| - cations - accepted) / (1 - donated) (clamped to 0..1) - and the search is repeated
    with the new x until it changes by < 0.02 (at most 4 rounds)."""
    res = _search_schemes(st, bv, donors, located, s, params, progress)
    poh = [d for d in donors if bv.cls[d["i"]] == CLASS_POH and d.get("options")]
    if not poh or not res.schemes:
        return res
    hist = {d["i"]: [bv.poh_x.get(d["i"], 0.5)] for d in poh}
    balance = bool(getattr(s, "mix_balance", False))
    for _ in range(6 if balance else 4):
        hs = scheme_hatoms(res, res.schemes[0])
        r = bv_with_h(st, bv, hs, s, params)
        changed = False
        free, slope = {}, {}
        for d in poh:
            i = d["i"]
            x_old = bv.poh_x.get(i, 0.5)
            own = [h for h in hs if h.donor == i and not h.located and h.cand is not None]
            don = sum(h.s_fi * h.p for h in own) / max(1e-6, x_old) if own else 0.0   # per whole H
            x = (bv.ideal[i] - bv.bvs_noH[i] - r["F_acc"][i]) / max(1e-6, 1.0 - don)
            free[i] = min(1.0, max(0.0, float(x)))
            slope[i] = max(0.05, 1.0 - don)       # F&I misfit change per unit of x
        new = dict(free)
        if balance:
            new = _balance_poh_and_mix(st, bv, free, slope)
        for d in poh:
            i = d["i"]
            x_old, x = bv.poh_x.get(i, 0.5), new[i]
            hist[i].append(x)
            if abs(x - x_old) > (1e-4 if balance else 0.02 - 1e-12):
                changed = changed or abs(x - x_old) >= 0.02
                f = max(1e-6, x) / max(1e-6, x_old)
                for o in d["options"]:
                    for h in o.hs:
                        h.occf *= f
                        h.p *= f
                    o.donorF = d["bvs"] - sum(h.s_fi * h.p for h in list(d["located"]) + o.hs)
                bv.poh_x[i] = max(1e-6, x)
        if not changed:
            break
        res = _search_schemes(st, bv, donors, located, s, params, progress)
    bv.h_classes = class_h_per_cell(st, bv)
    txt = "Partial hydroxyl H fraction x (refined from the F&I sum of the best scheme%s): " % (
        ", under overall charge balance" if balance else "") + "; ".join(
        "%s x = %.2f (%s)" % (st.atoms[i].label, bv.poh_x[i], " → ".join("%.2f" % v for v in h))
        for i, h in hist.items())
    bv.notes = [n for n in bv.notes if not n.startswith("Partial hydroxyl H fraction")] + [txt]
    return res


def _search_schemes(st, bv, donors, located, s, params, progress=None):
    """Rank whole-structure H-bond schemes (one option per donor with missing H; located H
    always included) by total = BV score + angle penalties + H...H penalties.
    The search ranks by F&I (fast, exact); the kept schemes are then re-evaluated with
    GRG (D-H optimisation) and re-ranked if rank_by is 'grg' or 'both'. Unless
    s.show_severe, schemes with an H...H contact < hh_severe involving an inferred H are dropped."""
    res = SchemeResult()
    res.st, res.bv, res.s = st, bv, s          # used by scheme_hatoms to resolve flips
    res.located = list(located)
    res.donors = [d for d in donors if d["options"]]
    res.fixed = [d for d in donors if d["n_missing"] == 0]
    res.skipped = [d for d in donors if d["n_missing"] > 0 and not d["options"]]
    # hard fails: donors with missing H but no viable option at all
    res.hard_fail = {d["i"]: "no viable option (%d candidate(s) pass, %d option(s) impossible)" % (
        sum(1 for c in d["cands"] if c.passes), d.get("n_impossible", 0)) for d in res.skipped}
    res.opts = [d["options"][:max(1, int(s.max_options))] for d in res.donors]
    res.schemes, res.n_total, res.method, res.n_hidden = [], 0, "none", 0
    res.fixed_contacts = h_contacts(st, res.located, cutoff=s.hh_mod) if res.located else []
    nd = len(res.donors)
    n = len(st.atoms)
    anion, cF, bF, mF, cG = site_targets(st, bv, s)
    baseF = np.array(bv.bvs_noH, dtype=float)
    for h in res.located:
        if h.cand is not None:
            baseF[h.donor] -= h.s_fi * h.p
            baseF[h.cand.acceptor] += h.s_fi * h.weight
    fixed_geom = geom_penalty(res.located, s)
    dims = [len(o) for o in res.opts]
    K = max(1, int(s.keep_schemes))
    cache = {}

    def finish(rows):
        out = []
        for row in rows:
            sc = dict(row)
            ev = evaluate_hset(st, bv, scheme_hatoms(res, sc), s, params, cache)
            for k in ("bvF", "bvG", "maxF", "maxG", "total", "h_cell", "red", "mod", "sev", "dis", "cone",
                      "hh", "geom"):
                sc[k] = ev[k]
            out.append(sc)
        out.sort(key=lambda x: -x["total"])
        return out[:K]

    # H...H: geometry contacts on a sliding scale (+ disorder-only donor conflicts); water acceptor cones
    f_red, f_mod, _f_sev, _f_dis, f_pen = contacts_score(st, res.fixed_contacts, s)
    f_cone = len(cone_violations(st, res.located, s)) if res.located else 0
    loc_prim = {}
    for h in res.located:
        if h.primary:
            loc_prim.setdefault(h.donor, []).append(h)
    searched = {d["i"] for d in res.donors}
    fixed_cones = {i: _water_cone(st, lst[:2], s) for i, lst in loc_prim.items()
                   if i not in searched and len(lst) == 2}
    loc_in = {}
    for h in res.located:
        if h.cand is not None:
            loc_in.setdefault(h.cand.acceptor, []).append(h)

    def n_inside(hs_in, cone):
        return sum(1 for h in hs_in if cone is not None and cone_check(st, h, cone)[1])

    if nd == 0:
        if res.located:
            res.method = "located H only"
            res.n_total = 1
            res.schemes = finish([{"idx": (), "geom": fixed_geom}])
        return res

    incF, geom, sred, smod, ssev, spen, scone, opt_cones = [], [], [], [], [], [], [], []
    for d, ol in zip(res.donors, res.opts):
        m = len(ol)
        ia = d["i"]
        aF = np.zeros((m + 1, n))
        g, r_, md, sv, pn, cn = (np.zeros(m + 1) for _ in range(6))
        cones_d = []
        for k, o in enumerate(ol):
            for h in o.hs:
                aF[k, h.donor] -= h.s_fi * h.p
                aF[k, h.cand.acceptor] += h.s_fi * h.weight
            flips = getattr(o, "flips", [])
            for b in flips:                    # located-H branch that flips: that H-bond is lost
                aF[k, b.donor] += b.s_fi * b.p
                aF[k, b.cand.acceptor] -= b.s_fi * b.weight
            g[k] = o.geom + s.flip_penalty * sum(b.p for b in flips)
            cs = h_contacts(st, o.hs, cutoff=s.hh_mod)
            if res.located:
                cs = cs + h_contacts(st, o.hs, res.located, cutoff=s.hh_mod)
            r_[k], md[k], sv[k], _dis, pn[k] = contacts_score(st, cs, s)
            allh = loc_prim.get(ia, []) + [h for h in o.hs if getattr(h, "occf", 1.0) >= 0.999] + \
                [h for h in o.hs if getattr(h, "occf", 1.0) < 0.999][:1]
            cone = _water_cone(st, allh[:2], s) if len(allh) == 2 else None     # water-like donors only
            cones_d.append(cone)
            c_n = n_inside([h for h in loc_in.get(ia, []) if h.donor != ia], cone)
            for h in o.hs:                     # this option's H-bonds into waters whose H are all located
                if h.cand.acceptor in fixed_cones:
                    c_n += n_inside([h], fixed_cones[h.cand.acceptor])
            cn[k] = c_n
        incF.append(aF)
        geom.append(g)
        sred.append(r_)
        smod.append(md)
        ssev.append(sv)
        spen.append(pn)
        scone.append(cn)
        opt_cones.append(cones_d)

    env = st.environment(s.env_radius)
    reach = s.hh_mod + 2 * (max(s.dh_length, s.geom_dh_length) + 0.05)
    pairs = []
    for a in range(nd):
        ia = res.donors[a]["i"]
        for b in range(a + 1, nd):
            ib = res.donors[b]["i"]
            if not any(nb.j == ib and nb.d <= reach for nb in env[ia]):
                continue
            shape = (dims[a] + 1, dims[b] + 1)
            rm, mm, sm, pm, cm = (np.zeros(shape) for _ in range(5))
            for ka, oa in enumerate(res.opts[a]):
                to_b = [h for h in oa.hs if h.cand.acceptor == ib]
                for kb, ob in enumerate(res.opts[b]):
                    cs = h_contacts(st, oa.hs, ob.hs, cutoff=s.hh_mod)
                    if cs:
                        rm[ka, kb], mm[ka, kb], sm[ka, kb], _dis, pm[ka, kb] = contacts_score(st, cs, s)
                    to_a = [h for h in ob.hs if h.cand.acceptor == ia]
                    cm[ka, kb] = n_inside(to_b, opt_cones[b][kb]) + n_inside(to_a, opt_cones[a][ka])
            if rm.any() or mm.any() or pm.any() or cm.any():
                pairs.append((a, b, rm, mm, sm, pm, cm))

    # speed: only the scored sites that some option changes are carried per candidate scheme; the
    # others contribute a constant
    touched = np.zeros(n, dtype=bool)
    for aF in incF:
        touched |= np.any(aF != 0.0, axis=0)
    T = np.nonzero(anion & touched)[0]
    U = anion & ~touched
    wsite = site_weights(st, s)
    const_bv = -float((site_penalty(baseF[U], cF[U], bF[U], mF[U], s) * wsite[U]).sum()) if U.any() else 0.0
    incT, colT = [], []                        # per donor: only the columns its options change
    for aF in incF:
        a_T = aF[:, T]
        cols = np.nonzero(np.any(a_T != 0.0, axis=0))[0]
        colT.append(cols)
        incT.append(np.ascontiguousarray(a_T[:, cols]))
    baseT, cT, bT, mT, wT = baseF[T], cF[T], bF[T], mF[T], wsite[T]

    # speed: H...H + cone penalties pre-combined into one table per donor / donor pair (flattened, one
    # gather per pair); red / moderate counts only when asked for (the kept schemes are re-evaluated anyway)
    P = s.hh_red_penalty
    hh_d = [spen[d] + scone[d] * P for d in range(nd)]
    pairs_c = []
    for a, b, rm, mm, sm, pm, cm in pairs:
        mb = rm.shape[1]
        pairs_c.append((a, b, mb, (pm + cm * P).ravel(), sm.ravel() if sm.any() else None,
                        rm.ravel(), mm.ravel()))

    def score(I, full=False):
        B = I.shape[0]
        Fv = np.repeat(baseT[None, :], B, axis=0)
        gm = np.full(B, float(fixed_geom))
        hh = np.full(B, float(f_pen) + float(f_cone) * P)
        sev = np.zeros(B)
        red = np.full(B, float(f_red)) if full else None
        mod = np.full(B, float(f_mod)) if full else None
        for d in range(nd):
            col = I[:, d]
            if len(colT[d]):
                Fv[:, colT[d]] += incT[d][col]
            gm += geom[d][col]
            hh += hh_d[d][col]
            sev += ssev[d][col]
            if full:
                red += sred[d][col]
                mod += smod[d][col]
        for a, b, mb, hhp, sevp, redp, modp in pairs_c:
            flat = I[:, a] * mb + I[:, b]
            hh += hhp.take(flat)
            if sevp is not None:
                sev += sevp.take(flat)
            if full:
                red += redp.take(flat)
                mod += modp.take(flat)
        bvF = const_bv - (site_penalty(Fv, cT, bT, mT, s) * wT).sum(axis=1)
        total = s.w_bv * bvF + gm + hh
        if not s.show_severe:
            total = np.where(sev > 0, total - 1e12, total)
        return {"total": total, "geom": gm, "hh": hh, "red": red, "mod": mod, "sev": sev}

    W = K if s.rank_by == "fi" else max(3 * K, 1000)
    N = 1
    for x in dims:
        N *= x
    res.n_total = N
    if N <= s.exhaustive_limit:
        res.method = "exhaustive"
        best_I, best_t = None, None
        chunk = 50000
        for start in range(0, N, chunk):
            flat = np.arange(start, min(N, start + chunk))
            I = np.stack(np.unravel_index(flat, dims), axis=1)
            sc_ = score(I)
            t = sc_["total"]
            res.n_hidden += int((sc_["sev"] > 0).sum()) if not s.show_severe else 0
            if best_I is not None:
                I = np.concatenate([best_I, I])
                t = np.concatenate([best_t, t])
            if len(t) > W:
                keep = np.argpartition(-t, W - 1)[:W]
                I, t = I[keep], t[keep]
            best_I, best_t = I, t
            if progress:
                progress(min(N, start + chunk), N)
        I = best_I
    else:
        res.method = "beam (width %d) + refinement" % int(s.beam_width)
        res.n_hidden = None
        BW = max(W, int(s.beam_width))
        I = np.array([dims], dtype=int)
        for d in range(nd):
            B = I.shape[0]
            new = np.repeat(I, dims[d], axis=0)
            new[:, d] = np.tile(np.arange(dims[d]), B)
            if len(new) > BW:
                t = score(new)["total"]
                keep = np.argpartition(-t, BW - 1)[:BW]
                # diversity: also keep the best few rows of every option of this donor, so that an option
                # that looks best now cannot crowd out all alternatives before later donors reveal clashes
                r_ = max(3, BW // (4 * max(1, dims[d])))
                extra = []
                for v in range(dims[d]):
                    rows_v = np.nonzero(new[:, d] == v)[0]
                    if len(rows_v):
                        extra.append(rows_v[np.argsort(-t[rows_v])[:r_]])
                if extra:
                    keep = np.unique(np.concatenate([keep] + extra))
                new = new[keep]
            I = new
            if progress:
                progress(d + 1, nd)
        t = score(I)["total"]
        if len(t) > W:
            I = I[np.argpartition(-t, W - 1)[:W]]
        I_beam = I.copy()
        I = I.copy()
        for _sweep in range(20):
            changed = False
            for d in range(nd):
                m = dims[d]
                trial = np.repeat(I, m, axis=0)
                trial[:, d] = np.tile(np.arange(m), I.shape[0])
                tt = score(trial)["total"].reshape(I.shape[0], m)
                best = tt.argmax(axis=1)
                cur = tt[np.arange(I.shape[0]), I[:, d]]
                better = tt[np.arange(I.shape[0]), best] > cur + 1e-9
                if better.any():
                    I[better, d] = best[better]
                    changed = True
            if not changed:
                break
        I = np.unique(np.concatenate([I, I_beam]), axis=0)
        t = score(I)["total"]
        n_ok = int((t > -1e11).sum())
        if n_ok < W:
            # the sweeps often make many beam rows converge on the same few schemes: fill the list with
            # single-donor variations of the best ones (useful alternatives to browse / compare)
            top = I[np.argsort(-t)[:min(len(I), 60)]]
            neigh = []
            for d in range(nd):
                if dims[d] < 2:
                    continue
                rep = np.repeat(top, dims[d], axis=0)
                rep[:, d] = np.tile(np.arange(dims[d]), len(top))
                neigh.append(rep)
            if neigh:
                I = np.unique(np.concatenate([I] + neigh), axis=0)
                t = score(I)["total"]
        if len(t) > W:
            I = I[np.argpartition(-t, W - 1)[:W]]
    sc = score(I, full=True)
    order = np.argsort(-sc["total"])
    rows = [{"idx": tuple(int(x) for x in I[r]), "geom": float(sc["geom"][r]), "hh": float(sc["hh"][r]),
             "red": float(sc["red"][r]), "mod": float(sc["mod"][r]), "sev": float(sc["sev"][r])}
            for r in order if s.show_severe or sc["sev"][r] == 0]
    if progress:
        progress(0, 0)
    res.schemes = finish(rows)
    if not res.schemes and not s.show_severe:
        # every combination contains an impossible H...H contact: find the donors responsible
        s2 = replace(s, show_severe=True, keep_schemes=1)
        tmp = search_schemes(st, bv, donors, located, s2, params)
        if tmp.schemes:
            for c in evaluate_hset(st, bv, scheme_hatoms(tmp, tmp.schemes[0]), s2, params)["contacts"]:
                a, b, d = c[0], c[1], c[2]
                if c[2] < s.hh_severe and contact_is_geometry(st, c):
                    for h in (a, b):
                        if not h.located:
                            res.hard_fail.setdefault(h.donor, "every scheme has an H···H contact < %.2f Å "
                                                              "involving this donor" % s.hh_severe)
    return res


def scheme_hatoms(res, scheme, include_located=True):
    """H atoms of a scheme; located H that must flip away from a disordered acceptor are replaced by
    flipped copies (see resolve_flips)."""
    hs = list(res.located) if include_located else []
    hs += [h for d, k in enumerate(scheme["idx"]) if k < len(res.opts[d]) for h in res.opts[d][k].hs]
    if include_located and getattr(res, "bv", None) is not None:
        hs = resolve_flips(res.st, res.bv, hs, res.s)
    return hs


def consolidate_schemes(st, res):
    """Merge schemes with the same acceptor ATOMS for every donor: symmetry-equivalent copies (same score)
    and variants that use other symmetry copies of the same acceptors (other geometry and score). The
    best-scoring one is kept, with sc['equiv'] = number merged and sc['variants'] = how many of them
    scored differently. Returns the number removed."""
    if getattr(res, "consolidated", False):
        return 0
    out, index = [], {}
    for sc in sorted(res.schemes, key=lambda x: -x["total"]):
        # same acceptor ATOMS for every donor (the table shows them identically): merged, keeping the best.
        # Copies with the same score are symmetry-equivalent; copies with another score use other symmetry
        # copies of the same acceptors (other geometry) - counted as 'variants'
        key = scheme_summary(st, res, sc)
        hit = index.get(key)
        if hit is not None:
            hit["equiv"] = hit.get("equiv", 1) + 1
            if abs(hit["total"] - sc["total"]) >= 0.02:
                hit["variants"] = hit.get("variants", 0) + 1
            continue
        sc = dict(sc)
        sc["equiv"], sc["variants"] = 1, 0
        index[key] = sc
        out.append(sc)
    n = len(res.schemes) - len(out)
    res.schemes = out
    res.consolidated = True
    return n


def scheme_summary(st, res, scheme):
    parts = []
    for d, k in enumerate(scheme["idx"]):
        o = res.opts[d][k]
        parts.append("%s→%s" % (res.donors[d]["label"], ",".join(st.atoms[c.acceptor].label for c in o.cands)))
    return "; ".join(parts) if parts else "(located H only)"


# ============================================================================
# Refinement: rigid rotation of inferred H about their parent O
# ============================================================================

def _rotate(v, axis_angle_deg):
    """Rodrigues rotation of vector v by rotation vector (degrees)."""
    w = np.radians(np.asarray(axis_angle_deg, dtype=float))
    th = np.linalg.norm(w)
    v = np.asarray(v, dtype=float)
    if th < 1e-12:
        return v
    k = w / th
    return v * math.cos(th) + np.cross(k, v) * math.sin(th) + k * k.dot(v) * (1 - math.cos(th))


def donor_badness(st, bv, hs, s, params):
    """Per-donor contribution to misfit: its own and its acceptors' BV misfit (F&I outside band,
    GRG deviation), angle penalties of its H-bonds and H...H contacts of its H."""
    ev = evaluate_hset(st, bv, hs, s, params)
    r = ev["r"]
    bad = {}
    for h in hs:
        if h.located:
            continue
        i = h.donor
        b = bad.setdefault(i, 0.0)
        sites = [i] + ([h.cand.acceptor] if h.cand is not None else [])
        for j in sites:
            b += (r["outF"][j] / s.bv_scale) ** 2 + (r["devG"][j] / s.bv_scale) ** 2
        if h.cand is not None:
            b -= h.cand.adc_pen
        bad[i] = b
    for a, b2, d, _, _ in ev["contacts"]:
        for h in (a, b2):
            if not h.located:
                bad[h.donor] = bad.get(h.donor, 0.0) + ((s.hh_mod - d) / s.refine_sigma_hh) ** 2
    return bad, ev


def penalised_donors(st, bv, hs, s, params):
    """Donors with inferred H whose part of the scheme received any penalty."""
    bad, ev = donor_badness(st, bv, hs, s, params)
    r = ev["r"]
    out = set()
    by = {}
    for h in hs:
        if not h.located:
            by.setdefault(h.donor, []).append(h)
    for i, lst in by.items():
        if geom_penalty([h for h in hs if h.donor == i], s) < 0:
            out.add(i)
        for h in lst:
            sites = [i] + ([h.cand.acceptor] if h.cand is not None else [])
            if any(r["outF"][j] > 0 or abs(r["devG"][j]) > 0.10 for j in sites):
                out.add(i)
    for a, b, d, _, _ in ev["contacts"]:
        for h in (a, b):
            if not h.located:
                out.add(h.donor)
    return out


class RefinedScheme:
    pass


def pick_for_refinement(res, top=20):
    """Index of the scheme to refine automatically, among the `top` best-scoring schemes: the best one
    without any H...H geometry contact (< hh_mod) or acceptor-cone violation; if there is none, the one
    with the smallest H...H violation sum (least negative H...H penalty), ties broken by total score."""
    if not res.schemes:
        return None
    pool = range(min(top, len(res.schemes)))
    clean = [k for k in pool if res.schemes[k].get("red", 0) == 0 and res.schemes[k].get("mod", 0) == 0
             and res.schemes[k].get("cone", 0) == 0]
    if clean:
        return clean[0]
    return max(pool, key=lambda k: (res.schemes[k].get("hh", 0.0), res.schemes[k]["total"]))


def _rot_about(v, axis, deg):
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-9 or abs(deg) < 1e-12:
        return np.asarray(v, dtype=float)
    return _rotate(v, a / n * deg)


def refine_scheme(st, bv, hs, s, params, mode="all", max_deg=None, n_worst=None, progress=None):
    """Move the inferred H of the chosen donors about their parent O to minimise
      E = sum_anions ((GRG sum - ideal) / sigma_bv)^2
        + sum over H...H geometry contacts < hh_mod of x^2 + x^4,  x = (hh_mod - d) / sigma_hh
          (disorder-only donor conflicts count 0.25 x^2)
        + water acceptor-cone violations: 1 + ((half-angle - angle) / 5 deg)^2 each
        + refine_w_sym * ((s1 - s2) / 0.05)^2 for a water whose two H-bonds differ in strength
    Each water/hydroxyl turns rigidly by at most max_deg (a water with one located H turns about that
    O-H) and a water's H-D-H angle may open/close by up to +- refine_hdh_deg. Located H never move.
    With refine_dh, every inferred D-H length may also change by up to +- refine_dh_range, restrained by
      + refine_dh_weight * sum (change / sigma)^2,  sigma = refine_dh_range / 2
    and the GRG sums then use these lengths directly (no separate +- dh_range re-fit).
    F&I sums depend only on D...A distances and do not change.
    mode: 'all' | 'worst' (n_worst donors with the largest misfit) | 'penalised'.
    Bounded pattern search (derivative free). Returns a RefinedScheme."""
    max_deg = s.refine_max_deg if max_deg is None else max_deg
    n_worst = s.refine_n_worst if n_worst is None else n_worst
    hdh = max(0.0, s.refine_hdh_deg)
    hyp_by = {}
    for h in hs:
        if not h.located:
            hyp_by.setdefault(h.donor, []).append(h)
    if mode == "all":
        chosen = sorted(hyp_by)
    elif mode == "worst":
        bad, _ = donor_badness(st, bv, hs, s, params)
        chosen = sorted(sorted(bad, key=lambda i: -bad[i])[:max(1, int(n_worst))])
    else:
        chosen = sorted(penalised_donors(st, bv, hs, s, params))
    fixed = [h for h in hs if h.located or h.donor not in chosen]
    sym_img = h_site_images(st, hs)         # H pairs related by the donor's site symmetry stay related
    loc_axis = {}
    for h in hs:
        if h.located and h.primary and h.donor in chosen and h.donor not in loc_axis:
            loc_axis[h.donor] = np.array(h.vec) / np.linalg.norm(h.vec)
    # D-H refinement: one length parameter per inferred H, in units of 0.01 A (comparable step size to
    # the angles in degrees), bounded by +- refine_dh_range, Gaussian restraint with sigma = range / 2
    dh_on = bool(s.refine_dh) and s.refine_dh_range > 0
    dh_lim = s.refine_dh_range / 0.01 if dh_on else 0.0
    dh_sig = s.refine_dh_range / 2.0 if dh_on else 1.0
    blocks = []                       # (donor, n rotation parameters, has scissor parameter, n length params)
    for i in chosen:
        nrot = 1 if i in loc_axis else 3
        sc = hdh > 0 and (len(hyp_by[i]) == 2 or (i in loc_axis and len(hyp_by[i]) == 1)) and \
            not any(getattr(h, "occf", 1.0) < 0.999 for h in hyp_by[i])      # split H: rigid rotation only
        blocks.append((i, nrot, sc, len(hyp_by[i]) if dh_on else 0))
    nx = sum(b[1] + (1 if b[2] else 0) + b[3] for b in blocks)
    cache = {}
    an, _, _, _, cG = site_targets(st, bv, s)          # scored anion sites
    wsite = site_weights(st, s)                        # occupancy weights (as in the scheme score)
    s_bv = s                          # refined (fixlen) H keep their lengths in GRG; others are re-fitted

    def build(x):
        out, p = list(fixed), 0
        rots, sciss, dls = {}, {}, {}
        for i, nrot, has_sc, ndh in blocks:
            rv = loc_axis[i] * x[p] if nrot == 1 else np.array(x[p:p + 3])
            p += nrot
            dlt = x[p] if has_sc else 0.0
            p += 1 if has_sc else 0
            dl = [x[p + k] * 0.01 for k in range(ndh)]
            p += ndh
            rots[i], sciss[i], dls[i] = float(np.linalg.norm(rv)), float(dlt), dl
            vecs = [np.array(h.vec, dtype=float) for h in hyp_by[i]]
            if has_sc:
                if len(vecs) >= 2:              # open/close the H-D-H angle symmetrically
                    nrm = np.cross(vecs[0], vecs[1])
                    vecs[0] = _rot_about(vecs[0], nrm, -dlt / 2.0)
                    vecs[1] = _rot_about(vecs[1], nrm, dlt / 2.0)
                else:                           # one located + one inferred H: turn the inferred one
                    nrm = np.cross(loc_axis[i], vecs[0])
                    vecs[0] = _rot_about(vecs[0], nrm, dlt)
            fin = {}
            for k, (h, v) in enumerate(zip(hyp_by[i], vecs)):
                v = _rotate(v, rv)
                if dl:
                    n0 = np.linalg.norm(v)
                    v = v * ((n0 + dl[k]) / n0)
                fin[id(h)] = (v, dl[k] if dl else 0.0)
            for k, h in enumerate(hyp_by[i]):
                v, d_ = fin[id(h)]
                if id(h) in sym_img and id(sym_img[id(h)][0]) in fin:   # ordered on a special position:
                    r, Rc = sym_img[id(h)]                              # keep it the exact symmetry image
                    v, d_ = Rc @ fin[id(r)][0], fin[id(r)][1]
                g0 = getattr(h, "grg_len", None) or s.dh_length
                hn = _carry_split(h, _make_h(st, i, h.cand, v, s, params, grg_len=g0 + d_))
                hn.fixlen = bool(dl)
                out.append(hn)
        return out, rots, sciss, dls

    def energy(x):
        hs2, _, _, dls = build(x)
        r = bv_with_h(st, bv, hs2, s_bv, params, cache)
        e = float(np.sum(wsite[an] * (((r["totG"] - cG)[an]) / s.refine_sigma_bv) ** 2))
        if dh_on:                                # restraint: H resist moving along the bond
            e += s.refine_dh_weight * sum((d / dh_sig) ** 2 for dl in dls.values() for d in dl)
        for c in h_contacts(st, hs2, cutoff=s.hh_mod):
            xx = (s.hh_mod - c[2]) / s.refine_sigma_hh
            e += (xx * xx + xx ** 4) if contact_is_geometry(st, c) else 0.25 * xx * xx
        for _h, _w, ang, half in cone_violations(st, hs2, s):
            e += 1.0 + ((half - ang) / 5.0) ** 2
        if s.refine_w_sym:
            for i in chosen:
                own = [h for h in hs2 if h.donor == i and h.primary and h.cand is not None]
                if len(own) == 2:
                    s1, s2 = (r["sha"][r["hpos"][id(h)]] for h in own)
                    e += s.refine_w_sym * ((s1 - s2) / 0.05) ** 2
        return e

    def clamp(x):
        x = np.array(x, dtype=float)
        p = 0
        for _, nrot, has_sc, ndh in blocks:
            seg = x[p:p + nrot]
            n = np.linalg.norm(seg)
            if n > max_deg:
                x[p:p + nrot] = seg * (max_deg / n)
            p += nrot
            if has_sc:
                x[p] = max(-hdh, min(hdh, x[p]))
                p += 1
            for k in range(ndh):
                x[p + k] = max(-dh_lim, min(dh_lim, x[p + k]))
            p += ndh
        return x

    x = np.zeros(nx)
    e0 = e = energy(x) if nx else 0.0
    step, evals = max(max_deg, hdh, dh_lim) / 2.0, 0
    while nx and step > 0.05 and evals < 8000:
        improved = False
        for c in range(nx):
            for sgn in (1.0, -1.0):
                t = clamp(x + sgn * step * np.eye(nx)[c])
                et = energy(t)
                evals += 1
                if et < e - 1e-9:
                    x, e, improved = t, et, True
                    break
        if not improved:
            step /= 2.0
        if progress:
            progress(evals)
    hs_new, rots, sciss, dls = build(x)
    hs_new = resolve_flips(st, bv, [h for h in hs_new], s)
    rs = RefinedScheme()
    rs.hs, rs.rotations, rs.scissors, rs.mode, rs.max_deg = hs_new, rots, sciss, mode, max_deg
    rs.dh_changes = dls if dh_on else {}
    rs.donors = chosen
    rs.e_start, rs.e_end, rs.evals = e0, e, evals
    rs.eval = evaluate_hset(st, bv, hs_new, s, params)
    return rs


# ============================================================================
# Hypothetical-H CIF output
# ============================================================================

def _letters(k):
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        s = chr(97 + r) + s
    return s


def label_hydrogens(st, hs):
    """{id(h): label} - located H keep their CIF label; hypothetical H get short SHELXL-style names
    (H<number><letter>, <= 4 characters; see shelx_names), the same names used for SHELXL output."""
    labels = {id(h): h.located for h in hs if h.located}
    taken = _res_labels(st.res_text) if getattr(st, "res_text", None) else ()
    labels.update(shelx_names(st, hs, taken))
    return labels


def hypothetical_cif_path(source, optimized=False):
    """<name>_hypothetical.cif (or <name>_optimized_hypothetical.cif when atom coordinates were optimised),
    numbered 1, 2, ... if that exists."""
    base = os.path.splitext(source)[0] + ("_optimized_hypothetical" if optimized else "_hypothetical")
    path, n = base + ".cif", 1
    while os.path.exists(path):
        path = "%s%d.cif" % (base, n)
        n += 1
    return path


def _single_comp(h):
    """Component index of an H that belongs to exactly one component of a combined scheme, else None."""
    c = getattr(h, "comp", None)
    return next(iter(c)) if c is not None and len(c) == 1 else None


def _h_value(tag, h, st, labels, ordered=()):
    D = st.atoms[h.donor]
    u = 1.5 * D.uiso if D.uiso else 0.05
    # H off the donor's symmetry element: disordered copies - unless its copies are the donor's other H
    # (ordered on the site, `ordered` = ids of those representatives), then it is a whole atom
    occ = D.occ if id(h) in ordered else D.occ / h_symmetry_copies(st, h)
    occ *= getattr(h, "occf", 1.0)            # split H (towards one of two disordered acceptors)
    vals = {
        "_atom_site_label": labels[id(h)],
        "_atom_site_type_symbol": "H",
        "_atom_site_fract_x": "%.5f" % h.frac[0],
        "_atom_site_fract_y": "%.5f" % h.frac[1],
        "_atom_site_fract_z": "%.5f" % h.frac[2],
        "_atom_site_u_iso_or_equiv": "%.4f" % u,
        "_atom_site_b_iso_or_equiv": "%.3f" % (8 * math.pi ** 2 * u),
        "_atom_site_adp_type": "Uiso",
        "_atom_site_thermal_displace_type": "Uiso",
        "_atom_site_occupancy": ("%.4f" % occ).rstrip("0").rstrip(".") if occ != 1 else "1",
        "_atom_site_calc_flag": "calc",
        "_atom_site_symmetry_multiplicity": "1",
        "_atom_site_site_symmetry_multiplicity": "1",
        "_atom_site_site_symmetry_order": "1",
        "_atom_site_disorder_assembly": "H" if _single_comp(h) is not None else ".",
        "_atom_site_disorder_group": str(_single_comp(h) + 1) if _single_comp(h) is not None else ".",
    }
    if tag in vals:
        return vals[tag]
    if tag.startswith("_atom_site_refinement_flags"):
        return "."
    return "?"


def _rewrite_formula(lines, start, st, bv, hs):
    """Set _chemical_formula_sum (and _chemical_formula_weight) of the data block at `start` to the model:
    the sites (occupancy x multiplicity / Z) plus the H of the scheme `hs`; add a comment with the
    oxidation states and the charge balance."""
    f, z = scheme_formula(st, bv, hs)
    order = list(dict.fromkeys([e for e in (st.formula or {}) if e != "H"] + ["H"]))
    new_sum = formula_text(f, order=order)
    fw = formula_weight(f)
    q = formula_charge(st, bv, h_cell=h_per_cell(st, hs))
    end = next((k for k in range(start + 1, len(lines)) if lines[k].strip().lower().startswith("data_")), len(lines))

    def set_item(tag, value):
        nonlocal end
        for k in range(start + 1, end):
            t = lines[k].split(None, 1)
            if not t or t[0].lower() != tag:
                continue
            if len(t) > 1 and t[1].strip():
                lines[k] = "%s %s" % (t[0], value)
            else:                                  # value on the next line(s)
                m = k + 1
                while m < end and not lines[m].strip():
                    m += 1
                if m < end and lines[m].strip().startswith(";"):
                    e2 = m + 1
                    while e2 < end and not lines[e2].strip().startswith(";"):
                        e2 += 1
                    del lines[m:e2 + 1]
                    end -= e2 + 1 - m
                elif m < end:
                    del lines[m]
                    end -= 1
                lines[k] = "%s %s" % (t[0], value)
            return True
        return False

    old = st.formula_text if st.formula_text not in (None, "?") else "?"
    if not set_item("_chemical_formula_sum", "'%s'" % new_sum):
        lines.insert(start + 1, "_chemical_formula_sum '%s'" % new_sum)
        end += 1
    if fw is not None and not set_item("_chemical_formula_weight", "%.2f" % fw):
        lines.insert(start + 2, "_chemical_formula_weight %.2f" % fw)
    lines.insert(start + 1, "# formula updated by HydroGEN (sites + H of this scheme; was '%s'); oxidation states %s; "
                            "charge per formula unit %+.2f" % (old, oxidation_state_text(st, bv), q["total"]))
    return lines


def write_hypothetical_cif(st, hs, s, drop_cif_h=False, bv=None):
    """Copy of the source CIF with the H atoms `hs` appended to the _atom_site loop of
    this structure's data block (located H are already there and are skipped).
    drop_cif_h: remove the H atoms already in the CIF (atom_site and aniso rows), e.g. when the
    analysis ignored them. bv: when given, _chemical_formula_sum / _weight are set to the model (sites +
    the H of `hs`). Returns (path, labels)."""
    labels = label_hydrogens(st, hs)
    hs_all = list(hs)
    imgs = h_site_images(st, hs_all)            # H generated by the donor's own site symmetry: not written
    ordered = {id(r) for r, _ in imgs.values()}
    hs = [h for h in hs if not h.located and id(h) not in imgs]
    cif_h = {a.label.lower() for a in st.atoms if a.el == "H"} if drop_cif_h else set()
    with open(st.source, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()

    def first(l):
        t = l.split()
        return t[0].lower() if t else ""

    start = next((k for k, l in enumerate(lines) if first(l) == ("data_" + st.name).lower()), None)
    if start is None:
        raise ValueError("Data block 'data_%s' not found in %s" % (st.name, st.source))
    end = next((k for k in range(start + 1, len(lines)) if first(lines[k]).startswith("data_")), len(lines))

    if cif_h:      # drop the CIF's own H rows from the atom_site and atom_site_aniso loops
        keep, k, in_site_loop = [], start + 1, False
        out = lines[:start + 1]
        while k < end:
            l = lines[k]
            f = first(l)
            if f == "loop_":
                m, tg = k + 1, []
                while m < end and lines[m].strip().startswith("_"):
                    tg.append(first(lines[m]))
                    m += 1
                in_site_loop = "_atom_site_label" in tg or "_atom_site_aniso_label" in tg
                out.extend(lines[k:m])
                k = m
                continue
            if f.startswith("_") or f.startswith(("data_", "save_")):
                in_site_loop = False
            if not (in_site_loop and f in cif_h):
                out.append(l)
            k += 1
        removed = end - start - 1 - (len(out) - start - 1)
        lines = out + lines[end:]
        end -= removed

    tags, insert_at, k = None, None, start + 1
    while k < end:
        if first(lines[k]) == "loop_":
            m, tg = k + 1, []
            while m < end and lines[m].strip().startswith("_"):
                tg.append(first(lines[m]))
                m += 1
            if "_atom_site_label" in tg and "_atom_site_fract_x" in tg:
                last, q = m - 1, m
                while q < end:
                    t = lines[q].strip()
                    if t.startswith("_") or first(lines[q]) in ("loop_",) or \
                            first(lines[q]).startswith(("data_", "save_")):
                        break
                    if t and not t.startswith("#"):
                        last = q
                    q += 1
                tags, insert_at = tg, last + 1
                break
            k = m
            continue
        k += 1
    if tags is None:
        raise ValueError("No _atom_site loop with fractional coordinates found in data_%s" % st.name)

    rows = ["# hypothetical H (HydroGEN): D-H %.3f A, H-D-H %.1f deg, placed nearest the acceptor"
            " (refined H may be rotated)" % (s.geom_dh_length, s.hdh_angle)]
    if cif_h:
        rows.insert(0, "# the H atoms of the original CIF were removed (analysis ignored located H)")
    for h in sorted(hs, key=lambda x: labels[id(x)]):
        rows.append(" ".join(_h_value(t, h, st, labels, ordered) for t in tags))
    for k, (r, _) in sorted(imgs.items(), key=lambda kv: labels[id(kv[1][0])]):
        rows.append("# %s is the symmetry image of %s on the %s site (ordered; generated by the space group)" % (
            labels[k], labels[id(r)], st.atoms[r.donor].label))
    lines[insert_at:insert_at] = rows
    lines.insert(start + 1, "# %d hypothetical H atom(s) added by HydroGEN (positions not refined against "
                            "diffraction data)" % len(hs))
    opt = getattr(st, "optimized", None)
    if opt:
        lines = _rewrite_coordinates(lines, start, tags, opt)
        lines.insert(start + 2, "# atom coordinates OPTIMISED by HydroGEN (bond-valence driven shifts; %s) - not the "
                                "refined positions" % getattr(st, "opt_note", ""))
    if bv is not None:
        lines = _rewrite_formula(lines, start, st, bv, hs_all)
    path = hypothetical_cif_path(st.source, optimized=bool(opt))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(nl.join(lines) + nl)
    return path, labels


_CIF_TOK = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")


def _rewrite_coordinates(lines, start, tags, opt):
    """Replace _atom_site_fract_x/y/z of the sites in opt {label: (old, new)} in the atom_site loop of the
    data block starting at line `start` (rows are rewritten with single spaces)."""
    def first(l):
        t = l.split()
        return t[0].lower() if t else ""
    ix = [tags.index("_atom_site_fract_" + c) for c in "xyz"]
    il = tags.index("_atom_site_label")
    k, n = start + 1, len(lines)
    while k < n:
        if first(lines[k]) == "loop_":
            m, tg = k + 1, []
            while m < n and lines[m].strip().startswith("_"):
                tg.append(first(lines[m]))
                m += 1
            if tg == tags:
                q = m
                while q < n:
                    t = lines[q].strip()
                    if not t or t.startswith("#"):
                        q += 1
                        continue
                    if t.startswith("_") or first(lines[q]) == "loop_" or first(lines[q]).startswith(("data_", "save_")):
                        break
                    tok = _CIF_TOK.findall(lines[q])
                    if len(tok) == len(tags) and tok[il] in opt:
                        new = opt[tok[il]][1]
                        for c, j in enumerate(ix):
                            tok[j] = "%.6f" % new[c]
                        lines[q] = " ".join(tok)
                    q += 1
                return lines
            k = m
            continue
        if first(lines[k]).startswith("data_") and k > start:
            break
        k += 1
    return lines


# ============================================================================
# Bond-valence table (new-mineral style)
# ============================================================================

def _cation_cell(bv, a, c):
    """Bond valences from cation site c to anion site a, with x n-> (counted n times in the
    anion row) and x n (down arrow) (counted n times in the cation column)."""
    va = [bd.s for bd in bv.bonds[a] if bd.included and bd.nb.j == c]
    if not va:
        return ["-"]
    nc = sum(1 for bd in bv.bonds[c] if bd.included and bd.nb.j == a)
    groups = {}
    for v in va:
        groups.setdefault(round(v, 2), []).append(v)
    down = nc // len(va) if (nc > len(va) and nc % len(va) == 0) else 1
    lines = []
    for v in sorted(groups, reverse=True):
        txt = "%.2f" % v
        if len(groups[v]) > 1:
            txt += "×%d→" % len(groups[v])
        if down > 1:
            txt += "×%d↓" % down
        lines.append(txt)
    return lines


def bv_table(st, bv, hs, s, params, method="fi"):
    """Bond-valence table in the layout used in new-mineral descriptions.
    Rows: anions; columns: cations (not H), H, H bond, D (donated) or O-H, A (accepted), Σan;
    last row Σcat. method 'fi': Σan = cations + A - D (Ferraris & Ivaldi; own O-H not counted).
    method 'grg': Σan = cations + O-H + A (G&H 2015 H-O parameters, optimised D-H).
    Returns (title, header, rows, footnotes); each cell is a list of text lines."""
    r = bv_with_h(st, bv, hs, s, params)
    labels = label_hydrogens(st, hs)
    gr = getattr(bv, "nh4_gr", set())          # NH4+ bonded as a whole: a cation column
    cats = [i for i, a in enumerate(st.atoms) if bv.roles[i] == "cation" and a.el != "H"] + sorted(gr)
    ans = [i for i in range(len(st.atoms)) if bv.roles[i] == "anion" and i not in gr]
    fi = method == "fi"
    header = ["Atom"] + [st.atoms[c].label for c in cats] + ["H", "H bond", "D" if fi else "O–H", "A", "Σan"]
    don = {}
    acc = {}
    for h in hs:
        if h.donor in gr:                       # ammonium H: geometry only (its bonds are in the NH4+ column)
            continue
        don.setdefault(h.donor, []).append(h)
        if h.cand is not None:
            acc.setdefault(h.cand.acceptor, []).append(h)
    rows = []
    for a in ans:
        row = [[st.atoms[a].label]] + [_cation_cell(bv, a, c) for c in cats]
        hl, hb, dv = [], [], []
        for h in sorted(don.get(a, []), key=lambda x: (labels[id(x)], not x.primary)):
            hl.append((labels[id(h)] + ("" if h.located else "*")) if h.primary else "  ↳")
            if h.cand is not None:
                txt = "···" + st.atoms[h.cand.acceptor].label
                if h.p < 0.999:
                    txt += " (×%s)" % occ_text(h.p)
            elif getattr(h, "flipped", False):
                txt = "(flipped, ×%s)" % occ_text(h.pflip)
            else:
                txt = "(none)"
            hb.append(txt)
            k = r["hpos"][id(h)]
            if fi:
                dv.append("%.2f" % (h.s_fi * h.p) if h.cand is not None else "-")
            else:
                dv.append("%.2f" % r["sdh"][k] if h.primary else "")
        av = []
        for h in sorted(acc.get(a, []), key=lambda x: labels[id(x)]):
            k = r["hpos"][id(h)]
            v = (h.s_fi if fi else r["sha"][k]) * h.weight
            av.append("%.2f" % v)
        row += [hl or ["-"], hb or ["-"], dv or ["-"], av or ["-"],
                ["%.2f" % (r["totF"][a] if fi else r["totG"][a])]]
        rows.append(row)
    rows.append([["Σcat"]] + [["%.2f" % bv.bvs[c]] for c in cats] + [[""]] * 5)

    refs = {}
    for c in cats:
        for bd in bv.bonds[c]:
            if bd.included:
                key = ("NH4+–%s" % st.atoms[bd.nb.j].el) if c in gr else "%s%s–%s" % (
                    st.atoms[c].el, valence_text(bv.valence[c]) if bv.valence[c] else "", st.atoms[bd.nb.j].el)
                refs.setdefault(key, set()).add(bd.param.source)
    notes = ["Bond-valence parameters: " + "; ".join("%s %s" % (k, ", ".join(sorted(v)))
                                                      for k, v in sorted(refs.items())) + "."]
    mixed = [(c, v) for c, v in getattr(bv, "valence_spec", {}).items() if isinstance(v, tuple) and c in cats]
    if mixed:
        notes.append("Mixed valence: " + "; ".join("%s %.0f%% %s%d+ / %.0f%% %s%d+ (average %s)" % (
            st.atoms[c].label, 100 * v[3], st.atoms[c].el, v[1], 100 * (1 - v[3]), st.atoms[c].el, v[2],
            valence_text(valence_value(v))) for c, v in mixed) + " - bond valences are the same mixture.")
    if not hs:
        pass                                     # no H: no H-bond notes
    elif fi:
        notes.append("Hydrogen-bond valences after Ferraris & Ivaldi (1988), s = (D···A/%.2f)^-%.1f + %.2f; "
                     "D = donated, A = accepted; Σan = cations + A − D (O–H bonds to the parent O not counted)."
                     % (s.fi_d0, s.fi_n, s.fi_c))
    else:
        ds = ["%s %.3f Å" % (labels[id(h)], r["L"][r["hpos"][id(h)]]) for h in hs]
        notes.append("H–O parameters of Gagné & Hawthorne (2015) for O–H and H···A; each D–H allowed to "
                     "lengthen by up to %.2f Å to bring its parent O closest to its ideal sum. D–H used: %s. "
                     "Σan = cations + O–H + A." % (s.dh_range, ", ".join(ds)))
    if any(h.conditional and h.located for h in hs):
        notes.append("(×p): located H bonding to a disordered site bond to whichever alternative position is "
                     "present; p = occupancy of that position (½, ⅓ ... or decimal). D = p x s; the acceptor row "
                     "counts s as if the site were present.")
    if any(getattr(h, "compw", None) for h in hs):
        notes.append("Disordered (combined) scheme: components %s. (×p) = fraction of the cells in which that H "
                     "(and its H-bond) is present; D = p x s, the acceptor receives p x s, the O–H counts p x "
                     "s(O–H)." % ", ".join("%d ×%s" % (k + 1, occ_text(w)) for k, w in sorted(
                         next(h.compw for h in hs if getattr(h, "compw", None)).items())))
    elif any(getattr(h, "occf", 1.0) < 0.999 for h in hs):
        notes.append("(×p) on a hypothetical H: the H is split over two positions, one towards each alternative of a "
                     "disordered acceptor, each with fraction p (from the acceptors' occupancies), or a partial "
                     "hydroxyl's H fraction; D = p x s and the O–H counts p x s(O–H); the acceptor row counts s as if "
                     "the site were present.")
    if any(getattr(h, "flipped", False) for h in hs):
        notes.append("(flipped): a located H that would clash with the inferred H of a disordered site turns away "
                     "in the cells where that site is present (that fraction of its H-bond is lost).")
    if any(not h.located for h in hs):
        notes.append("* hypothetical H atom (placed at D–H %.3f Å, H–D–H %.0f°, nearest its acceptor; its GRG "
                     "valences use D–H %.3f Å along the same direction)."
                     % (s.geom_dh_length, s.hdh_angle, s.dh_length))
    if gr:
        notes.append("Ammonium %s: NH4+ as a whole (column), bonded to the anions with García-Rodríguez et al. "
                     "(2000) NH4+–X parameters; its H only set the geometry and are not listed."
                     % ", ".join(st.atoms[i].label for i in sorted(gr)))
    if any(a.occ != 1 for a in st.atoms):
        notes.append("Cation–anion values are unweighted; sums are weighted by site occupancy.")
    title = "Bond valence analysis for %s (vu) - %s" % (st.name, "Ferraris & Ivaldi H-bonds" if fi else
                                                         "GRG H-O parameters")
    if not hs:                                   # no H at all: a plain bond-valence table (no H / H-bond columns)
        drop = {k for k, h in enumerate(header) if h in ("H", "H bond", "D", "O–H", "A")}
        header = [h for k, h in enumerate(header) if k not in drop]
        rows = [[c for k, c in enumerate(r) if k not in drop] for r in rows]
        title = "Bond valence analysis for %s (vu)" % st.name
    return title, header, rows, notes


def bv_table_text(title, header, rows, notes):
    allrows = [[[h] for h in header]] + rows
    ncol = len(header)
    widths = [max(len(line) for row in allrows for line in row[c]) for c in range(ncol)]
    left = {0} | {c for c, h in enumerate(header) if h in ("H", "H bond")}   # Atom, H, H bond left-aligned

    def fmt_row(row):
        h = max(len(cell) for cell in row)
        out = []
        for k in range(h):
            parts = []
            for c, cell in enumerate(row):
                txt = cell[k] if k < len(cell) else ""
                parts.append(txt.ljust(widths[c]) if c in left else txt.rjust(widths[c]))
            out.append("  ".join(parts).rstrip())
        return out

    lines = [title, ""] + fmt_row(allrows[0])
    lines.append("-" * len(lines[-1]))
    for row in rows:
        lines += fmt_row(row)
    lines.append("")
    lines += notes
    return "\n".join(lines)


_FRACS = [(1 / 2, "½"), (1 / 3, "⅓"), (2 / 3, "⅔"), (1 / 4, "¼"), (3 / 4, "¾"), (1 / 5, "⅕"), (2 / 5, "⅖"),
          (3 / 5, "⅗"), (4 / 5, "⅘"), (1 / 6, "⅙"), (5 / 6, "⅚"), (1 / 8, "⅛"), (3 / 8, "⅜"), (5 / 8, "⅝"),
          (7 / 8, "⅞")]


def occ_text(x, tol=0.002):
    """1, ½, ⅓, ⅔, ¼ ... when x is (within tol) a simple fraction, otherwise two decimals."""
    if abs(x - 1.0) <= tol:
        return "1"
    for v, sym in _FRACS:
        if abs(x - v) <= tol:
            return sym
    return "%.2f" % x


# ============================================================================
# Hydrogen-bond details table
# ============================================================================

def hbond_details(st, bv, hs, s):
    """H-bond geometry table: D-H...A, D-H, D...A, H...A, angle D-H...A, symmetry code of A (apt_sym)
    and the occupancy of the H-bond (donor occupancy x probability that the acceptor position is
    present). Returns (title, header, rows, notes); each cell is a list of text lines."""
    labels = label_hydrogens(st, hs)
    header = ["D–H⋯A", "D–H", "D⋯A", "H⋯A", "∠DHA", "apt_sym", "Occ."]
    rows, codes, flipped = [], {}, []
    for h in sorted(hs, key=lambda x: (x.donor, labels[id(x)], not x.primary)):
        if h.cand is None:
            if getattr(h, "flipped", False):
                flipped.append(labels[id(h)])
            continue
        nb = h.cand.nb
        code = st.symop_code(nb.op, nb.shift)
        if code != ".":
            codes[code] = st.symop_text(nb.op, nb.shift)
        lab = labels[id(h)] + ("" if h.located else "*")
        occ = st.atoms[h.donor].occ * h.p
        rows.append([["%s–%s⋯%s" % (st.atoms[h.donor].label, lab, st.atoms[h.cand.acceptor].label)],
                     ["%.3f" % h.d_dh], ["%.3f" % nb.d], ["%.3f" % h.d_ha], ["%.1f" % h.angle],
                     [code if code != "." else ""], [occ_text(occ)]])
    notes = []
    if codes:
        notes.append("Symmetry codes (apt_sym): " + "; ".join("%s = %s" % (c, codes[c]) for c in sorted(codes)) + ".")
    notes.append("Occ. = occupancy of the H-bond: donor occupancy x occupancy of the acceptor position for H that "
                 "bond to whichever alternative of a disordered site is present.")
    if any(not h.located for h in hs):
        notes.append("* hypothetical H (D–H %.3f Å unless refined; positions not refined against diffraction data)."
                     % s.geom_dh_length)
    if flipped:
        notes.append("Located H turning away from a disordered acceptor in part of the cells: %s."
                     % ", ".join(sorted(set(flipped))))
    return "Hydrogen-bond details for %s (distances in Å, angles in °)" % st.name, header, rows, notes


def table_text(title, header, rows, notes, left=(0,)):
    """Plain-text rendering of a table whose cells are lists of lines."""
    allrows = [[[h] for h in header]] + rows
    ncol = len(header)
    widths = [max(len(line) for row in allrows for line in row[c]) for c in range(ncol)]

    def fmt_row(row):
        out = []
        for k in range(max(len(cell) for cell in row)):
            parts = []
            for c, cell in enumerate(row):
                txt = cell[k] if k < len(cell) else ""
                parts.append(txt.ljust(widths[c]) if c in left else txt.rjust(widths[c]))
            out.append("  ".join(parts).rstrip())
        return out

    lines = [title, ""] + fmt_row(allrows[0])
    lines.append("-" * len(lines[-1]))
    for row in rows:
        lines += fmt_row(row)
    return "\n".join(lines + [""] + notes)


# ============================================================================
# SHELXL output: H atom lines, DFIX restraints, .ins from a .res
# ============================================================================

SHELX_CMDS = set("""TITL CELL ZERR LATT SYMM SFAC DISP UNIT LAUE REM MORE TIME END HKLF OMIT SHEL BASF TWIN EXTI
SWAT HOPE MERG SPEC RESI MOVE ANIS AFIX HFIX FRAG FEND EXYZ EADP EQIV CONN PART BIND FREE DFIX DANG BUMP SAME SADI
CHIV FLAT DELU SIMU DEFS ISOR NCSY SUMP L.S. CGLS BLOC DAMP STIR WGHT FVAR BOND CONF MPLA RTAB HTAB LIST ACTA SIZE
TEMP WPDB FMAP GRID PLAN MOLE NEUT ABIN ANSC ANSR XNPD RIGU WIGL LONE ZERR SHEL PRIG""".split())
RESTRAINT_CMDS = {"DFIX", "DANG", "SADI", "HTAB", "BIND", "FREE", "CONN", "EADP", "EXYZ", "ISOR", "SIMU",
                  "DELU", "RIGU", "CHIV", "FLAT", "BUMP", "SAME", "MPLA", "RTAB", "CONF", "AFIX", "HFIX"}


def _res_atom(line):
    """(label, sfac, occ-field) if `line` is a SHELX atom line, else None."""
    if not line or line[0] in " \t":
        return None
    t = line.split()
    if len(t) < 6 or t[0].upper() in SHELX_CMDS or len(t[0]) > 4:
        return None
    try:
        int(t[1])
        float(t[2]); float(t[3]); float(t[4]); float(t[5])
    except ValueError:
        return None
    return t[0], t[1], t[5]


def _res_labels(res_text):
    return {a[0].upper() for a in (_res_atom(l) for l in (res_text or "").splitlines()) if a}


def shelx_names(st, hs, taken=()):
    """{id(h): name} for the inferred H: H<number><letter> (<= 4 characters, SHELXL limit), using the
    donor's number when free (OW5 -> H5A, H5B) or continuing a located H of the same donor (H8A -> H8B),
    otherwise the next free number."""
    used = {t.upper() for t in taken} | {a.label.upper() for a in st.atoms}
    nums = {int(m.group(1)) for t in used for m in [re.match(r"^H(\d+)", t)] if m}
    by = {}
    for h in hs:
        if not h.located and h.primary:
            by.setdefault(h.donor, []).append(h)
    loc_on = {}
    for h in hs:
        if h.located and h.primary:
            loc_on.setdefault(h.donor, []).append(h.located.upper())
    letters = "ABCDEFGH"
    names = {}
    for i in sorted(by):
        lst = sorted(by[i], key=lambda x: x.cand.ordinal if x.cand else 0)
        trial = []
        for L in loc_on.get(i, []):          # continue the numbering of a located H on the same donor
            m = re.match(r"^H(\d+)([A-Z])$", L)
            if m:
                start = letters.find(m.group(2)) + 1
                trial.append((int(m.group(1)), max(start, 0)))
        dig = re.findall(r"\d+", st.atoms[i].label)
        if dig:
            trial.append((int(dig[0]), 0))
        n_ok = None
        for n, k0 in trial:
            cand = ["H%d%s" % (n, letters[k0 + j]) for j in range(len(lst)) if k0 + j < len(letters)]
            if len(cand) == len(lst) and all(len(c) <= 4 and c not in used for c in cand):
                n_ok = cand
                break
        if n_ok is None:
            n = 1
            while True:
                cand = ["H%d%s" % (n, letters[j]) for j in range(len(lst))]
                if n not in nums and all(len(c) <= 4 and c not in used for c in cand):
                    break
                n += 1
            n_ok = cand
        for h, nm in zip(lst, n_ok):
            names[id(h)] = nm
            used.add(nm)
            m = re.match(r"^H(\d+)", nm)
            nums.add(int(m.group(1)))
    return names


def shelx_export(st, hs, s, res_text=None, drop_cif_h=False):
    """SHELXL lines for the inferred H in `hs`.
    Returns {'text': command lines for the text box, 'ins': full .ins text (only with a .res),
             'names': {id(h): name}, 'warnings': [...]}.
    H lines: name sfac x y z occ -1.2 (isotropic, Uiso = 1.2 x Ueq of the preceding atom = the parent).
    Occupancy copies the parent's field from the .res (keeps free-variable links), else 10 + occupancy.
    Restraints: DFIX 0.82 0.07 parent H, and DFIX 1.33 0.07 H H for two H on one parent."""
    imgs = h_site_images(st, hs)            # generated by the donor's own site symmetry: not written
    ordered = {id(r) for r, _ in imgs.values()}
    hyp = [h for h in hs if not h.located and h.primary and h.cand is not None and id(h) not in imgs]
    res_lines = res_text.splitlines() if res_text else None
    taken = _res_labels(res_text) if res_text else set()
    names = shelx_names(st, hs, taken)
    warnings = []
    sfac_h, parent = None, {}
    if res_lines:
        sfac = []
        for l in res_lines:
            t = l.split()
            if t and t[0].upper() == "SFAC":
                rest = t[1:]
                if len(rest) > 1 and all(re.match(r"^[A-Za-z]{1,2}$", x) for x in rest):
                    sfac += [x.capitalize() for x in rest]
                elif rest:
                    sfac.append(rest[0].capitalize())
        if "H" in sfac:
            sfac_h = str(sfac.index("H") + 1)
        for k, l in enumerate(res_lines):
            a = _res_atom(l)
            if a and a[0].upper() not in parent:
                parent[a[0].upper()] = (k, a[2])
    by = {}
    for h in hyp:
        by.setdefault(h.donor, []).append(h)
    h_lines, dfix = {}, []
    for i in sorted(by):
        plab = st.atoms[i].label
        lst = sorted(by[i], key=lambda x: names[id(x)])
        copies_any = False
        for h in lst:
            copies = 1 if id(h) in ordered else h_symmetry_copies(st, h)
            split = getattr(h, "occf", 1.0) < 0.999
            if id(h) in ordered and not split:          # whole H in a general position (parent on the axis
                occf = "%.5f" % (10.0 + st.atoms[i].occ)  # has a reduced SHELX site occupancy - don't copy it)
                h_lines.setdefault(plab.upper(), []).append(
                    "%-5s%-3s %10.6f %10.6f %10.6f %11s %10.5f" % (names[id(h)], sfac_h or "1", h.frac[0],
                                                                h.frac[1], h.frac[2], occf, -1.2))
                dfix.append("DFIX %.2f 0.07 %s %s" % (0.87 if st.atoms[i].el == "N" else 0.82, plab, names[id(h)]))
                continue
            alab = st.atoms[h.cand.acceptor].label.upper()
            if split and getattr(h, "alt", None) and not getattr(h, "compw", None) and res_lines and \
                    alab in parent and copies == 1:
                occf = parent[alab][1]           # split H: present with its acceptor -> follow its occupancy
            elif split:
                occf = "%.5f" % (10.0 + st.atoms[i].occ * h.occf / copies)
            elif res_lines and plab.upper() in parent and copies == 1:
                occf = parent[plab.upper()][1]
            else:
                occf = "%.5f" % (10.0 + st.atoms[i].occ / copies)
                copies_any = copies_any or copies > 1
            h_lines.setdefault(plab.upper(), []).append(
                "%-5s%-3s %10.6f %10.6f %10.6f %11s %10.5f" % (names[id(h)], sfac_h or "1", h.frac[0], h.frac[1],
                                                            h.frac[2], occf, -1.2))
            dfix.append("DFIX %.2f 0.07 %s %s" % (0.87 if st.atoms[i].el == "N" else 0.82, plab, names[id(h)]))
        if copies_any:
            warnings.append("H on %s sits off its parent's symmetry element: occupancy divided by the number of "
                            "copies (fixed, not linked to a free variable)." % plab)
        sym_img = [names[k] for k, (r, _) in imgs.items() if r.donor == i and k in names]
        if sym_img:
            warnings.append("H on %s are ordered on its site: %s generated by symmetry from %s (one H written, "
                            "full occupancy; no H...H DFIX)." % (
                                plab, ", ".join(sym_img), ", ".join(names[id(h)] for h in lst if id(h) in ordered)))
        whole =[names[id(h)] for h in lst if getattr(h, "occf", 1.0) >= 0.999] + \
                [h.located for h in hs if h.located and h.primary and h.donor == i]
        splits = [names[id(h)] for h in lst if getattr(h, "occf", 1.0) < 0.999]
        if any(getattr(h, "compw", None) for h in lst):     # combined scheme: H...H within each component
            allh = lst + [h for h in hs if h.located and h.primary and h.donor == i]
            seen_p = set()
            for k in next(h.compw for h in lst if getattr(h, "compw", None)):
                mem = [h for h in allh if getattr(h, "comp", None) is None or k in h.comp]
                nm = [names[id(h)] if not h.located else h.located for h in mem]
                for a_, b_ in combinations(nm, 2):
                    if (a_, b_) not in seen_p and len(mem) <= 2:
                        seen_p.add((a_, b_))
                        dfix.append("DFIX 1.33 0.07 %s %s" % (a_, b_))
        elif splits and whole:                   # the whole H with each alternative of the split one
            dfix += ["DFIX 1.33 0.07 %s %s" % (whole[0], x) for x in splits]
        elif len(whole) == 2:
            dfix.append("DFIX 1.33 0.07 %s %s" % (whole[0], whole[1]))
        elif len(whole) >= 3:                    # NH4+ / NH3 / H3O+: tetrahedral H...H for every pair
            hh = 2 * (0.87 if st.atoms[i].el == "N" else 0.82) * math.sin(math.radians(109.47 / 2))
            dfix += ["DFIX %.2f 0.07 %s %s" % (hh, a, b) for a, b in combinations(whole, 2)]
    head = ["REM HydroGEN: %d hypothetical H for %s (positions from the H-bond scheme, not refined)" % (len(hyp), st.name),
            "REM Uiso = -1.2 x Ueq of the PRECEDING atom: always place each H line directly after its parent atom.",
            "REM DFIX 0.82 0.07 = parent-H; DFIX 1.33 0.07 = H...H on the same parent."]
    if not res_lines:
        head.append("REM No .res available: '1' is used as the SFAC number of H - change it to match your SFAC "
                    "line; occupancy 11.0 = fixed full occupancy (10 + occ for partial sites; link to your "
                    "free variables if the parent's occupancy is refined).")
    body = list(dfix) + [""]
    order = {a.label.upper(): k for k, a in enumerate(st.atoms)}
    for p in sorted(h_lines, key=lambda x: parent[x][0] if x in parent else 10 ** 6 + order.get(x, 0)):
        body.append("REM after %s:" % next(a.label for a in st.atoms if a.label.upper() == p))
        body += h_lines[p]
    out = {"text": "\n".join(head + [""] + body), "ins": None, "names": names, "warnings": warnings}
    if not res_lines:
        return out

    # ---- build the .ins ------------------------------------------------------------
    lines = list(res_lines)
    opt = {k.upper(): v for k, v in (getattr(st, "optimized", None) or {}).items()}
    if opt:                                      # optimised coordinates replace the refined ones
        n_rep = 0
        for k, l in enumerate(lines):
            a = _res_atom(l)
            if a and a[0].upper() in opt:
                t = l.split()
                new = opt[a[0].upper()][1]
                t[2:5] = ["%.6f" % v for v in new]
                lines[k] = " ".join(t)
                n_rep += 1
        warnings.append("Coordinates of %d atom(s) are the OPTIMISED (bond-valence) positions, not the refined "
                        "ones." % n_rep)
    drop = {a.label.upper() for a in st.atoms if a.el == "H"} if drop_cif_h else set()
    if drop:
        new, skip_cont = [], False
        for l in lines:
            if skip_cont:
                skip_cont = l.rstrip().endswith("=")
                continue
            a = _res_atom(l)
            if a and a[0].upper() in drop:
                skip_cont = l.rstrip().endswith("=")
                continue
            t = l.split()
            if t and t[0].upper() in RESTRAINT_CMDS and any(x.split("_")[0].upper() in drop for x in t[1:]):
                new.append("REM " + l)           # restraint on an H that was removed
                continue
            new.append(l)
        lines = new
    n_scheme = int(round(h_per_cell(st, hs))) if hs else 0
    if sfac_h is not None and n_scheme:          # H already in SFAC: its UNIT count = the H of this scheme
        k_h = int(sfac_h) - 1
        for k, l in enumerate(lines):
            t = l.split()
            if t and t[0].upper() == "UNIT" and len(t) > k_h + 1:
                if t[k_h + 1] != str(n_scheme):
                    warnings.append("UNIT for H changed from %s to %d (H per cell of this scheme)." % (t[k_h + 1], n_scheme))
                    t[k_h + 1] = str(n_scheme)
                    lines[k] = " ".join(t)
                break
    if sfac_h is None:                           # add H to SFAC and UNIT
        n_h = n_scheme or formula_h_per_cell(st)
        n_h = int(round(n_h)) if n_h else len(hyp) * 4
        for k, l in enumerate(lines):
            t = l.split()
            if t and t[0].upper() == "SFAC" and len(t) > 2 and all(re.match(r"^[A-Za-z]{1,2}$", x) for x in t[1:]):
                lines[k] = l.rstrip() + " H"
                sfac_h = str(len(t))
            elif t and t[0].upper() == "UNIT":
                lines[k] = l.rstrip() + " %d" % n_h
        warnings.append("H was not in SFAC: added as element %s with UNIT %d (check UNIT)." % (sfac_h, n_h))
        for p in h_lines:                        # fix the SFAC number in the H lines
            h_lines[p] = [x[:5] + ("%-3s" % sfac_h) + x[8:] for x in h_lines[p]]
    # insert H lines after their parents (and their continuation lines), last first
    idx = {}
    for k, l in enumerate(lines):
        a = _res_atom(l)
        if a and a[0].upper() not in idx:
            idx[a[0].upper()] = k
    missing = [p for p in h_lines if p not in idx]
    if missing:
        warnings.append("Parent atom(s) not found in the .res: %s - their H are added before HKLF." % ", ".join(missing))
    for p in sorted([p for p in h_lines if p in idx], key=lambda x: -idx[x]):
        k = idx[p]
        while lines[k].rstrip().endswith("=") and k + 1 < len(lines):
            k += 1
        lines[k + 1:k + 1] = h_lines[p]
    hk = next((k for k, l in enumerate(lines) if l.split()[:1] and l.split()[0].upper() == "HKLF"), len(lines))
    for p in missing:
        lines[hk:hk] = h_lines[p]
    # DFIX block before WGHT / FVAR
    fv = next((k for k, l in enumerate(lines) if l.split()[:1] and l.split()[0].upper() in ("WGHT", "FVAR")), None)
    if fv is None:
        fv = next((k for k, l in enumerate(lines) if _res_atom(l)), len(lines))
    lines[fv:fv] = ["REM HydroGEN hypothetical H restraints"] + dfix
    lines.insert(1, "REM HydroGEN: hypothetical H added for refinement%s" %
                 (" (H of the original model removed)" if drop else ""))
    out["ins"] = "\n".join(lines) + "\n"
    return out


def shelx_output_paths(source, optimized=False):
    """(<name>_HydroGEN.ins, .hkl) - <name>_optimized_HydroGEN for optimised coordinates - numbered
    _HydroGEN1, 2 ... so earlier files are never overwritten."""
    base = os.path.splitext(source)[0] + ("_optimized_HydroGEN" if optimized else "_HydroGEN")
    stem, n = base, 1
    while os.path.exists(stem + ".ins") or os.path.exists(stem + ".hkl"):
        stem = "%s%d" % (base, n)
        n += 1
    return stem + ".ins", stem + ".hkl"


# ============================================================================
# Geometric H-bonds (H atoms located in the CIF)
# ============================================================================

def located_hbonds(st, donors=("N", "O"), acceptors=("N", "O", "F", "Cl", "S"),
                   max_ha=2.7, max_da=3.5, min_angle=120.0, max_dh=1.3, env_radius=6.0):
    donors = {e.capitalize() for e in donors if e}
    acceptors = {e.capitalize() for e in acceptors if e}
    env = st.environment(max(env_radius, max_ha + 0.1))
    rows, n_orphan = [], 0
    hyd = [i for i, a in enumerate(st.atoms) if a.el == "H"]
    for i in hyd:
        heavy = [n for n in env[i] if st.atoms[n.j].el != "H"]
        if not heavy or heavy[0].d > max_dh:
            n_orphan += 1
            continue
        dn = heavy[0]
        D = st.atoms[dn.j]
        if D.el not in donors:
            continue
        seen = set()
        for nb in env[i]:
            if nb.d > max_ha:
                break
            A = st.atoms[nb.j]
            if A.el not in acceptors:
                continue
            v_da = (nb.vec[0] - dn.vec[0], nb.vec[1] - dn.vec[1], nb.vec[2] - dn.vec[2])
            d_da = math.sqrt(sum(x * x for x in v_da))
            if d_da < 0.1 or d_da > max_da:
                continue
            ang = angle_between(dn.vec, nb.vec)
            if ang < min_angle or nb.key in seen:
                continue
            seen.add(nb.key)
            rows.append({"donor": D.label, "hydrogen": st.atoms[i].label, "acceptor": A.label,
                         "d_DH": dn.d, "d_HA": nb.d, "d_DA": d_da, "angle": ang,
                         "symop": st.symop_text(nb.op, nb.shift), "code": st.symop_code(nb.op, nb.shift)})
    notes = []
    if not hyd:
        notes.append("No H atoms in the CIF - see the 'Inferred H-bonds' tab.")
    if n_orphan:
        notes.append("%d H atom(s) had no parent atom within %.2f A." % (n_orphan, max_dh))
    rows.sort(key=lambda r: (r["donor"], r["hydrogen"], r["d_HA"]))
    return rows, notes


# ============================================================================
# Lower symmetry: describe the structure in an index-2 subgroup (disorder across a symmetry element
# can then be modelled as order: e.g. an H disordered over a mirror becomes two independent sites)
# ============================================================================

def _op_arrays(st):
    return [(np.array(R, dtype=float), np.array(t, dtype=float) % 1.0) for R, t in st.symops]


def _op_key(R, t):
    return (tuple(int(round(x)) for x in np.asarray(R).ravel()),
            tuple(int(round((x % 1.0) * 48)) % 48 for x in t))


def _compose(a, b):
    """(a o b)(x) = Ra (Rb x + tb) + ta."""
    return a[0] @ b[0], (a[0] @ b[1] + a[1]) % 1.0


def _closure(gens, ident):
    elems = {_op_key(*ident): ident}
    frontier = [ident]
    while frontier:
        new = []
        for x in frontier:
            for g in gens:
                y = _compose(g, x)
                k = _op_key(*y)
                if k not in elems:
                    elems[k] = y
                    new.append(y)
        frontier = new
        if len(elems) > 400:
            break
    return elems


def index2_subgroups(st):
    """All subgroups of index 2 of the space group (operations taken modulo the CIF cell).
    Returns a list of lists of indices into st.symops."""
    ops = _op_arrays(st)
    keys = [_op_key(*o) for o in ops]
    n = len(ops)
    if n < 2 or n % 2:
        return []
    if n == 2:                                 # e.g. P-1 -> P1, Pc -> P1
        return [[k for k in range(n) if is_identity(st.symops[k][0], st.symops[k][1])]]
    ident = (np.eye(3), np.zeros(3))
    target = n // 2
    found = {}
    nonid = [k for k in range(n) if not is_identity(ops[k][0].tolist(), ops[k][1].tolist())]
    sizes = (1, 2, 3) if n <= 16 else (1, 2)
    for m in sizes:
        for comb in combinations(nonid, m):
            el = _closure([ops[k] for k in comb], ident)
            if len(el) != target:
                continue
            idx = tuple(sorted(k for k in range(n) if keys[k] in el))
            if len(idx) == target:
                found[idx] = True
    return [list(k) for k in found]


def _op_symbol(R, t):
    """Short Hermann-Mauguin-like symbol of one operation's rotational part (1, -1, 2, m, 3, 4 ...)."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float)
    det, tr = round(np.linalg.det(R)), round(np.trace(R))
    if det > 0:
        sym = {3: "1", -1: "2", 0: "3", 1: "4", 2: "6"}.get(tr, "?")
        if sym == "1" and np.any(np.abs((t + 0.5) % 1.0 - 0.5) > 1e-6):
            sym = "t"                          # pure translation (centring)
    else:
        sym = {-3: "-1", 1: "m", 0: "-3", -1: "-4", -2: "-6"}.get(tr, "?")
    if sym in ("2", "m"):                      # intrinsic translation: screw axis / glide plane
        w = (t + R @ t) / 2.0
        w = (w + 0.5) % 1.0 - 0.5
        if np.any(np.abs(w) > 1e-6):
            if sym == "2":
                sym = "2_1"
            else:
                nz = [k for k in range(3) if abs(w[k]) > 1e-6]
                sym = "abc"[nz[0]] if len(nz) == 1 else ("n" if len(nz) == 2 else "d")
    return sym


def subgroup_name(st, idx):
    """Readable name of a subgroup: lattice letter + point-group symbols, e.g. 'C -1', 'P 2/m'."""
    ops = [st.symops[k] for k in idx]
    syms = {_op_symbol(R, t) for R, t in ops}
    n_t = sum(1 for R, t in ops if _op_symbol(R, t) == "t")
    n_t_all = sum(1 for R, t in st.symops if _op_symbol(R, t) == "t")
    letter = (st.spacegroup.strip()[:1] or "P").upper()
    if n_t_all and n_t < n_t_all:
        letter = "P" if n_t == 0 else letter + "'"
    pg = syms - {"t", "1"}
    if n_t:                                    # centred: 2 + centring = 2_1, m + centring = glide
        if "2" in pg:
            pg.discard("2_1")
        if "m" in pg:
            pg -= {"a", "b", "c", "n", "d"}
    rot = [x for x in ("2_1", "2") if x in pg]
    mir = [x for x in ("m", "a", "b", "c", "n", "d") if x in pg]
    if pg <= {"2", "2_1", "m", "a", "b", "c", "n", "d", "-1"} and len(rot) <= 1 and len(mir) <= 1:
        if rot and mir:
            pgn = "%s/%s" % (rot[0], mir[0])
        elif rot or mir:
            pgn = (rot or mir)[0]
        else:
            pgn = "-1" if "-1" in pg else "1"
    else:
        pgn = "{%s}" % ",".join(sorted(pg))
    return "%s %s" % (letter, pgn)


def symmetry_disorder_pairs(st, s):
    """Partly occupied sites that clash with their own symmetry copy (closer than split_dist):
    [(site index, symop index, distance)] - disorder across a symmetry element."""
    out = []
    env = st.environment(max(s.split_dist, 0.1))
    for i, a in enumerate(st.atoms):
        if a.occ >= 0.999:
            continue
        for nb in env[i]:
            if nb.j == i and 0.05 < nb.d < s.split_dist:
                out.append((i, nb.op, nb.d))
    return out


def subgroup_report(st, s):
    """[(op indices, name, n disordered H self-pairs separated, n sites split, recommended)] for every
    index-2 subgroup; recommended = separates the most symmetry-related disordered pairs (ties: the
    one that keeps the lattice centring, then the fewest split sites)."""
    subs = index2_subgroups(st)
    pairs = symmetry_disorder_pairs(st, s)
    ops = _op_arrays(st)
    keys = [_op_key(*o) for o in ops]
    rows = []
    for idx in subs:
        kept = {keys[k] for k in idx}
        # only H disordered across a symmetry element can be re-inferred as ordered; split non-H sites
        # (e.g. a half-occupied water over an inversion centre) stay half occupied in the subgroup
        sep = sum(1 for i, op, d in pairs if keys[op] not in kept and st.atoms[i].el == "H")
        nsplit = sum(1 for orbs in (_orbits(st, a.f, idx) for a in st.atoms) if len(orbs) > 1)
        name = subgroup_name(st, idx)
        rows.append([idx, name, sep, nsplit, False])
    if rows:
        best = max(rows, key=lambda r: (r[2], "'" not in r[1] and not r[1].startswith("P") or
                                        st.spacegroup.strip().upper().startswith("P"), -r[3]))
        if best[2] > 0:
            best[4] = True
    rows.sort(key=lambda r: (-r[4], -r[2], r[3]))
    return [tuple(r) for r in rows], pairs


def _orbits(st, f, idx):
    """Split the orbit of position f under the full group into orbits of the subgroup idx.
    Returns a list of lists of fractional positions (mod 1); the first contains f."""
    ops = _op_arrays(st)
    f = np.array(f, dtype=float)
    pos = []
    for R, t in ops:
        p = (R @ f + t) % 1.0
        if not any(np.allclose((p - q + 0.5) % 1.0 - 0.5, 0, atol=2e-3) for q in pos):
            pos.append(p)
    sub = [ops[k] for k in idx]
    left, orbs = list(pos), []
    while left:
        p0 = left.pop(0)
        orb = [p0]
        for R, t in sub:
            q = (R @ p0 + t) % 1.0
            for k in range(len(left) - 1, -1, -1):
                if np.allclose((left[k] - q + 0.5) % 1.0 - 0.5, 0, atol=2e-3):
                    orb.append(left.pop(k))
        orbs.append(orb)
    return orbs


def lower_symmetry_cif(st, idx, drop_h="partial", s=None):
    """CIF text of the structure in subgroup idx (same cell): every site whose orbit splits becomes
    two sites (label + 'a' / 'b'); the second copy is generated by one operation of the lost coset
    from the original coordinates, so molecules stay together.
    drop_h: 'none' keep all H; 'partial' remove H whose occupancy is lower than their parent's (the
    disordered H - they are re-inferred, now ordered); 'all' remove every H.
    Returns (text, summary lines)."""
    ops = _op_arrays(st)
    keys = [_op_key(*o) for o in ops]
    kept = {keys[k] for k in idx}
    coset = next(o for o, k in zip(ops, keys) if k not in kept)
    name = subgroup_name(st, idx)
    parent_occ = {}
    if drop_h == "partial":
        env = st.environment(1.3)
        for i, a in enumerate(st.atoms):
            if a.el == "H":
                par = next((nb.j for nb in env[i] if st.atoms[nb.j].el != "H" and nb.d < 1.2), None)
                parent_occ[i] = st.atoms[par].occ if par is not None else 1.0
    rows, dropped, split = [], [], []
    for i, a in enumerate(st.atoms):
        if a.el == "H" and (drop_h == "all" or (drop_h == "partial" and a.occ < parent_occ.get(i, 1.0) - 0.01)):
            dropped.append(a.label)
            continue
        orbs = _orbits(st, a.f, idx)
        if len(orbs) == 1:
            rows.append((a.label, a, np.array(a.f, dtype=float)))
        else:
            f2 = coset[0] @ np.array(a.f, dtype=float) + coset[1]
            rows.append((a.label + "a", a, np.array(a.f, dtype=float)))
            rows.append((a.label + "b", a, f2))
            split.append(a.label)
    sg_items = [("_cell_length_a", st.a), ("_cell_length_b", st.b), ("_cell_length_c", st.c),
                ("_cell_angle_alpha", st.alpha), ("_cell_angle_beta", st.beta), ("_cell_angle_gamma", st.gamma)]
    L = ["data_%s_%s" % (st.name, re.sub(r"[^A-Za-z0-9]+", "", name)),
         "_audit_creation_method 'HydroGEN: %s described in subgroup %s'" % (st.spacegroup, name),
         "_space_group_name_H-M_alt '%s'" % name,
         "_chemical_formula_sum '%s'" % (st.formula_text if st.formula_text not in (None, "?") else "?")]
    if st.Z:
        L.append("_cell_formula_units_Z %g" % st.Z)
    L += ["%s %.5f" % (k, v) for k, v in sg_items]
    L += ["", "loop_", "_space_group_symop_operation_xyz"]
    for k in idx:
        R, t = st.symops[k]
        L.append("'%s'" % format_symop(R, [x % 1.0 for x in t], (0, 0, 0)))
    L += ["", "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
          "_atom_site_fract_z", "_atom_site_U_iso_or_equiv", "_atom_site_occupancy",
          "_atom_site_disorder_assembly", "_atom_site_disorder_group"]
    for lab, a, f in rows:
        L.append("%-8s %-3s %9.5f %9.5f %9.5f %8s %7.4f %s %s" % (
            lab, a.el, f[0], f[1], f[2], ("%.4f" % a.uiso) if a.uiso else "?", a.occ, a.asm or ".", a.grp or "."))
    summary = ["%s described in %s (index 2, same cell)." % (st.spacegroup, name),
               "Kept operations: " + "; ".join(format_symop(st.symops[k][0], [x % 1.0 for x in st.symops[k][1]],
                                                            (0, 0, 0)) for k in idx)]
    if split:
        summary.append("Sites split into two independent sites (a, b): " + ", ".join(split))
    if dropped:
        summary.append("H removed (re-inferred by the analysis): " + ", ".join(dropped))
    return "\n".join(L) + "\n", summary


def lower_symmetry_path(source, name):
    base = os.path.splitext(source)[0] + "_" + re.sub(r"[^A-Za-z0-9]+", "", name)
    path, n = base + ".cif", 1
    while os.path.exists(path):
        path = "%s_%d.cif" % (base, n)
        n += 1
    return path


# ============================================================================
# Structure optimisation: small, bond-valence-driven shifts of atom positions ahead of the H-bond
# analysis (cations satisfied; anions without H not over/under-bonded; moving an atom costs more the
# further it goes, steeply near the allowed radius)
# ============================================================================

ANION_BAND_BY_CLASS = {CLASS_NOT: 0.05, CLASS_POSSIBLE: 0.15, CLASS_PROBABLE: 0.30}


def default_movable(st, bv, s):
    """Atoms that move by default: anions with at least one summed bond, and partly occupied cations.
    Fully occupied cations and H stay (H follow their parent anion)."""
    out = set()
    for i, a in enumerate(st.atoms):
        if a.el == "H":
            continue
        if bv.roles[i] == "anion" and any(bd.included and not bd.to_h for bd in bv.bonds[i]):
            out.add(i)
        elif bv.roles[i] == "cation" and a.occ < 0.999:
            out.add(i)
    return out


def _site_projector(st, i):
    """Cartesian projector onto the displacements allowed by the site symmetry of atom i."""
    f = np.array(st.atoms[i].f, dtype=float)
    M, Mi = st.M, np.linalg.inv(st.M)
    Ps = []
    for R, t in st.symops:
        R = np.array(R, dtype=float)
        g = R @ f + np.array(t, dtype=float) - f
        if np.all(np.abs(g - np.round(g)) < 2e-3):
            Ps.append(M @ R @ Mi)
    return np.mean(Ps, axis=0) if Ps else np.eye(3)


def optimize_structure(st, bv, s, movable, radius=0.03, sigma=0.05, w_move=1.0, steep=3.0,
                       max_evals=30000, progress=None):
    """Shift the movable atoms (each within `radius` A of its refined position, along directions
    allowed by its site symmetry) to minimise
        sum over cations ((BVS - V) / sigma)^2            (partly occupied cations x occupancy)
      + w_move * sum over moved groups (exp(steep * (d / radius)^2) - 1)
    Only the CATION sums are fitted: anion (donor or acceptor) sums are not targeted - they are only
    reported (their H and H-bonds settle them later)."""
    return _optimize_structure(st, bv, s, movable, radius, sigma, w_move, steep, max_evals, progress)


def _optimize_structure(st, bv, s, movable, radius, sigma, w_move, steep, max_evals, progress):
    """(see optimize_structure)
    The movement cost is cheap for small shifts and steep near the radius, so a cation's deficit is
    shared among its anions (e.g. several waters bonded only to it) rather than met by moving one
    far. Partners of a mixed site move together; located H move with their parent.
    Bonds = those summed in the BV analysis (the bond list is kept fixed). A misfit beyond 0.25 vu
    grows only linearly (a site that small shifts cannot fix does not dominate).
    Default radius 0.03 A: in tests on 7 structures the mean shift was ~1/3 of the radius and most of
    the achievable cation misfit was removed (e.g. palisadeite rms 0.058 -> 0.008 vu, mean shift 0.003 A).
    Returns a dict (new fractional coordinates, shifts, BVS before / after, costs ...)."""
    n = len(st.atoms)
    M, Mi = st.M, np.linalg.inv(st.M)
    Rc = np.array([M @ np.array(R, dtype=float) @ Mi for R, _t in st.symops])
    # groups (mixed-site partners together; a group with an unticked member stays fixed)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in mixed_sites(st):
        parent[find(a)] = find(b)
    members = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    groups = [m for m in members.values() if all(i in movable for i in m) and st.atoms[m[0]].el != "H"]
    gof = {}
    for g, m in enumerate(groups):
        for i in m:
            gof[i] = g
    hpar = {k: p for k, p in getattr(bv, "h_parent", {}).items() if p in gof}
    P = [_site_projector(st, m[0]) for m in groups]
    # bonds
    bi, bj, vec, op, r0, bb, occ = [], [], [], [], [], [], []
    for i in range(n):
        for bd in bv.bonds[i]:
            if not bd.included or bd.to_h or st.atoms[bd.nb.j].el == "H":
                continue
            bi.append(i); bj.append(bd.nb.j); vec.append(bd.nb.vec); op.append(bd.nb.op)
            r0.append(bd.param.ro); bb.append(bd.param.b); occ.append(bd.occ)
    bi, bj, op = np.array(bi, dtype=int), np.array(bj, dtype=int), np.array(op, dtype=int)
    vec, r0, bb, occ = (np.array(x, dtype=float) for x in (vec, r0, bb, occ))
    # targets
    hrole = getattr(bv, "hrole", {})
    cat = np.array([bv.roles[i] == "cation" and bool(bv.ideal[i]) and st.atoms[i].el != "H" for i in range(n)],
                   dtype=bool)
    V = np.array([float(bv.ideal[i] or 0.0) for i in range(n)])
    an_lo = np.full(n, np.nan)
    for i in range(n):
        if bv.roles[i] != "anion" or i in hrole or st.atoms[i].el == "H":
            continue
        c = bv.cls[i]
        if c in DONOR_CLASSES or bv.n_loc[i] > 0:
            continue
        an_lo[i] = V[i] - ANION_BAND_BY_CLASS.get(c, 0.15)
    an = ~np.isnan(an_lo)
    an_hi = V + 0.05
    w_cat = np.array([st.atoms[i].occ if cat[i] else 0.0 for i in range(n)])   # partly occupied cations count less
    ng = len(groups)

    def disp(x):
        D = np.zeros((n, 3))
        for g in range(ng):
            dg = P[g] @ x[3 * g:3 * g + 3]
            for i in groups[g]:
                D[i] = dg
        return D

    def bvs(D):
        if not len(bi):
            return np.zeros(n)
        v = vec + np.einsum("bij,bj->bi", Rc[op], D[bj]) - D[bi]
        d = np.sqrt((v * v).sum(1))
        sv = np.exp((r0 - d) / bb) * occ
        return np.bincount(bi, weights=sv, minlength=n)

    huber = 0.25                                 # vu: beyond this a misfit grows only linearly, so one site that
                                                 # small shifts cannot fix (wrong valence ...) does not dominate

    def loss(dev):
        a = np.abs(dev)
        return np.where(a <= huber, a * a, 2 * huber * a - huber * huber) / (sigma * sigma)

    def parts(x):
        D = disp(x)
        S = bvs(D)
        e_cat = float(np.sum(w_cat[cat] * loss(S[cat] - V[cat])))
        e_an = 0.0                               # only the CATION sums are fitted (anion sums are not targeted)
        dist = np.array([np.linalg.norm(D[m[0]]) for m in groups]) if ng else np.zeros(0)
        e_mv = float(w_move * np.sum(np.exp(steep * (dist / radius) ** 2) - 1.0)) if ng else 0.0
        return e_cat + e_an + e_mv, e_cat, e_an, e_mv, S, dist

    def clamp(x):
        x = x.copy()
        for g in range(ng):
            dg = P[g] @ x[3 * g:3 * g + 3]
            nd = np.linalg.norm(dg)
            if nd > radius:
                x[3 * g:3 * g + 3] *= radius / nd
        return x

    x = np.zeros(3 * ng)
    e0, c0, a0, m0, S0, _ = parts(x)
    e = e0
    evals = 0
    step = radius / 2.0
    while ng and step > radius / 400.0 and evals < max_evals:
        improved = False
        for k in range(3 * ng):
            for sgn in (1.0, -1.0):
                t = x.copy()
                t[k] += sgn * step
                t = clamp(t)
                et = parts(t)[0]
                evals += 1
                if et < e - 1e-12:
                    x, e, improved = t, et, True
                    break
        if not improved:
            step /= 2.0
        if progress:
            progress(evals)
    e1, c1, a1, m1, S1, dist = parts(x)
    D = disp(x)
    for k, p in hpar.items():                           # located H ride with their parent
        D[k] = D[p]
    new_f = {i: tuple(float(v) for v in np.array(st.atoms[i].f) + Mi @ D[i])
             for i in range(n) if np.linalg.norm(D[i]) > 1e-6}
    return {"new_f": new_f, "shift": {i: float(np.linalg.norm(D[i])) for i in range(n)},
            "bvs0": S0, "bvs1": S1, "cost0": (e0, c0, a0, m0), "cost1": (e1, c1, a1, m1),
            "evals": evals, "groups": groups, "cat": cat, "an": an, "V": V, "an_lo": an_lo, "an_hi": an_hi,
            "radius": radius}


def optimized_structure(st, new_f, note=""):
    """Copy of structure st with the given fractional coordinates {site: (x, y, z)}; remembers the
    changes (st2.optimized = {label: (old, new)}) so outputs can say so."""
    import copy
    st2 = copy.copy(st)
    st2.atoms = [Atom(a.label, a.el, list(new_f.get(i, a.f)), a.occ, a.uiso, a.asm, a.grp)
                 for i, a in enumerate(st.atoms)]
    st2._env = {}
    for attr in ("_mult", "_mixed", "_Minv", "excl", "_alt_sites"):
        if hasattr(st2, attr):
            setattr(st2, attr, None)
    st2._disordered = set()
    st2.optimized = {st.atoms[i].label: (tuple(st.atoms[i].f), tuple(f)) for i, f in new_f.items()}
    st2.opt_note = note
    st2.tag = " [optimized]"
    return st2


# ============================================================================
# Subschemes: the ordered H-bond networks contained in a disordered scheme. The structure is expanded
# to P1 (same cell); every way of occupying the disordered positions that is allowed (no two positions
# that exclude each other, none left empty that could be occupied) is a configuration; the scheme's H
# that fit a configuration (donor present, acceptor present) form its network. A network is valid when
# every donor gets all its H and no H clashes. Symmetry-equivalent configurations can be consolidated.
# ============================================================================

SUB_MAX_CONFIGS = 4000      # configurations enumerated (randomly sampled beyond this)
SUB_MAX_EVAL = 60           # valid configurations scored at once (most frequent first; others when selected)


def _wrap(d):
    return (d + 0.5) % 1.0 - 0.5


class P1Cell:
    """Non-H sites of st expanded to P1: copies [(site, frac mod 1)], by_site {site: [copy]} (the copy at
    the CIF position first) and, per symmetry operation, the permutation of the copies."""

    def __init__(self, st):
        self.st = st
        self.ops = _op_arrays(st)
        self.M, self.Mi = st.M, np.linalg.inv(st.M)
        self.copies, self.by_site = [], {}
        for i, a in enumerate(st.atoms):
            if a.el == "H":
                continue
            f = np.array(a.f, dtype=float)
            f0 = f % 1.0
            pos = []
            for R, t in self.ops:
                p = (R @ f + t) % 1.0
                if not any(np.all(np.abs(_wrap(p - q)) < 2e-3) for q in pos):
                    pos.append(p)
            pos.sort(key=lambda q: 0 if np.all(np.abs(_wrap(q - f0)) < 2e-3) else 1)
            self.by_site[i] = []
            for p in pos:
                self.by_site[i].append(len(self.copies))
                self.copies.append((i, p))
        self.perms = [[self.find(i, (R @ p + t) % 1.0) for i, p in self.copies] for R, t in self.ops]

    def find(self, i, p):
        for c in self.by_site.get(i, ()):
            if np.all(np.abs(_wrap(self.copies[c][1] - p)) < 3e-3):
                return c
        return None

    def dist(self, p, q):
        return float(np.linalg.norm(self.M @ _wrap(np.asarray(p) - np.asarray(q))))

    def label(self, c):
        i, _p = self.copies[c]
        lst = self.by_site[i]
        lab = self.st.atoms[i].label
        return lab if len(lst) == 1 else "%s_%d" % (lab, lst.index(c) + 1)


def _rotation_between(u, w):
    """Rotation matrix that turns the direction of u into that of w."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    w = np.asarray(w, float) / np.linalg.norm(w)
    v, c = np.cross(u, w), float(u @ w)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def _maximal_independent_sets(nodes, adj, limit=2000):
    """All maximal sets of mutually compatible nodes (Bron-Kerbosch on the compatibility graph)."""
    nodes = list(nodes)
    S = set(nodes)
    comp = {u: S - adj[u] - {u} for u in nodes}
    out = []

    def bk(r, p, x):
        if len(out) >= limit:
            return
        if not p and not x:
            out.append(frozenset(r))
            return
        piv = max(p | x, key=lambda u: len(comp[u] & p))
        for v in list(p - comp[piv]):
            bk(r | {v}, p & comp[v], x & comp[v])
            p = p - {v}
            x = x | {v}
    bk(set(), set(nodes), set())
    return out


def _expand_hs(cell, hs):
    """Every symmetry copy of the scheme's H atoms in the P1 cell: [{'donor': copy, 'vec': cartesian D->H,
    'acc': acceptor copy or None, 'accvec': cartesian D->A, 'h': the scheme's H}]."""
    st = cell.st
    recs, seen = [], set()
    for h in hs:
        if not h.primary or h.donor not in cell.by_site:
            continue
        i = h.donor
        p0 = cell.copies[cell.by_site[i][0]][1]
        for R, t in cell.ops:
            Rc = cell.M @ R @ cell.Mi
            pd = (R @ p0 + t) % 1.0
            cd = cell.find(i, pd)
            if cd is None:
                continue
            v = Rc @ np.asarray(h.vec, dtype=float)
            key = (cd, tuple(np.round(v, 2)))
            if key in seen:
                continue
            seen.add(key)
            ca = av = None
            if h.cand is not None:
                av = Rc @ np.asarray(h.cand.nb.vec, dtype=float)
                ca = cell.find(h.cand.acceptor, (pd + cell.Mi @ av) % 1.0)
            recs.append({"donor": cd, "vec": v, "acc": ca, "accvec": av, "h": h})
    return recs


def find_subschemes(st, bv, hs, s, progress=None):
    """Configurations of the disordered structure st and the network of scheme hs (H list) in each.
    Returns {'cell': P1Cell, 'groups': [...], 'note': text, 'n_conf': n, 'sampled': bool}; a group =
    symmetry-equivalent configurations {'members': [config...], 'weight': summed frequency estimate,
    'valid': bool, 'reasons': [...], 'key'}; a config = {'present': frozenset of copies, 'hrecs': [...],
    'weight', 'desc'}. A combined (disordered) scheme is split into its components first."""
    comps = component_hsets(hs) if hs else None
    if comps:
        out = None
        for k, lst in sorted(comps.items()):
            r = find_subschemes(st, bv, lst, s, progress)
            for g in r["groups"]:
                g["component"] = k + 1
            if out is None:
                out = r
            else:
                out["groups"] += r["groups"]
                out["n_conf"] += r["n_conf"]
                out["sampled"] = out["sampled"] or r["sampled"]
        out["groups"].sort(key=lambda g: (not g["valid"], -g["weight"]))
        out["note"] = "Combined scheme: subschemes of each of its %d components. " % len(comps) + out["note"]
        return out
    cell = P1Cell(st)
    excl = st.excl
    partial = [c for c, (i, _p) in enumerate(cell.copies) if st.atoms[i].occ < 0.999]
    reach = max(excl.reach, s.group_dist) + 0.1
    adj = {c: set() for c in partial}
    for x, c1 in enumerate(partial):
        i1, p1 = cell.copies[c1]
        for c2 in partial[x + 1:]:
            i2, p2 = cell.copies[c2]
            d = cell.dist(p1, p2)
            if d <= reach and excl(i1, i2, d):
                adj[c1].add(c2)
                adj[c2].add(c1)
    # connected components of the exclusion graph: each choice is a maximal set of compatible positions;
    # an isolated partly occupied position is present or vacant, and a pair whose occupancies add up to
    # clearly less than 1 may also be vacant altogether
    always = {c for c, (i, _p) in enumerate(cell.copies) if st.atoms[i].occ >= 0.999}
    occ_of = {c: st.atoms[cell.copies[c][0]].occ for c in partial}
    # cations sharing one position (e.g. Mg / Zn): the same for the H-bond network - the majority atom
    for c in partial:
        if not adj[c]:
            continue
        same = [u for u in adj[c] if cell.dist(cell.copies[c][1], cell.copies[u][1]) < 0.05]
        grp = [c] + same
        if same and len(adj[c]) == len(same) and all(bv.roles[cell.copies[u][0]] == "cation" for u in grp) \
                and all(set(adj[u]) <= set(grp) for u in grp):
            keep = max(grp, key=lambda u: (occ_of[u], -u))
            for u in grp:
                adj[u] = set()
                occ_of[u] = 1.0 if u == keep else 0.0
    seen, components = set(), []
    for c in partial:
        if not adj[c]:
            o = occ_of[c]
            if o >= 0.75:                    # isolated, mostly occupied: present; mostly vacant: absent
                always.add(c)
            elif o > 0.25:
                components.append([(frozenset([c]), o), (frozenset(), 1.0 - o)])
            continue
        if c in seen:
            continue
        stack, comp = [c], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(adj[u] - comp)
        seen |= comp
        mis = _maximal_independent_sets(comp, adj)
        w = [float(np.prod([occ_of[u] for u in m])) for m in mis]
        if len(comp) == 2:
            vac = 1.0 - sum(occ_of[u] for u in comp)
            if vac > 0.05:
                mis.append(frozenset())
                w.append(vac)
        tot = sum(w) or 1.0
        components.append(list(zip(mis, [x / tot for x in w])))
    if not components:
        return {"cell": cell, "groups": [], "n_conf": 0, "sampled": False,
                "note": "No disorder that changes the H-bond network (no alternative or partly vacant positions; "
                        "cations sharing a position count as one) - no subschemes."}
    n_all = 1
    for c in components:
        n_all *= len(c)
    sampled = n_all > SUB_MAX_CONFIGS
    if not sampled:
        combos = product(*components)
    else:
        # sampled in proportion to the occupancy weights (the most frequent configurations first)
        rng = np.random.default_rng(12345)
        picked = {tuple(int(np.argmax([w for _m, w in c])) for c in components)}
        probs = [np.array([w for _m, w in c]) / sum(w for _m, w in c) for c in components]
        for _ in range(SUB_MAX_CONFIGS * 5):
            picked.add(tuple(int(rng.choice(len(pr), p=pr)) for pr in probs))
            if len(picked) >= SUB_MAX_CONFIGS:
                break
        combos = ([components[k][j] for k, j in enumerate(t)] for t in sorted(picked))
    # the scheme's H in P1, with every clash each of them would have (with an atom / another H) if present
    recs = _expand_hs(cell, hs)
    hrole = getattr(bv, "hrole", {}) or {}
    # an H-bond to a disordered acceptor position that is vacant in a configuration goes to the occupied
    # alternative position instead (e.g. a water split over a 2-fold axis, 0.4 A apart): the same H, turned
    # towards it - used only when the original acceptor is absent and the new one is a usable acceptor
    extra = []
    for r in recs:
        if r["h"].located or r["acc"] is None or r["acc"] not in adj:
            continue
        pa = cell.copies[r["donor"]][1] + cell.Mi @ r["accvec"]
        for c2 in adj[r["acc"]]:
            j2 = cell.copies[c2][0]
            if c2 == r["donor"] or bv.roles[j2] != "anion" or bv.cls[j2] in NON_ACCEPTOR_CLASSES or \
                    hrole.get(j2) == "exclude":
                continue
            nv = r["accvec"] + cell.M @ _wrap(cell.copies[c2][1] - pa)
            dn = float(np.linalg.norm(nv))
            if not (s.da_min <= dn <= s.da_max) or float(np.linalg.norm(nv - r["accvec"])) > 1.5:
                continue
            Rm = _rotation_between(r["accvec"], nv)
            extra.append({"donor": r["donor"], "vec": Rm @ r["vec"], "acc": c2, "accvec": nv, "h": r["h"],
                          "alt_of": r})
    recs += extra
    by_donor = {}
    P = np.array([p for _i, p in cell.copies])
    lim = np.array([s.h_min_cation if bv.roles[i] == "cation" else s.h_min_anion for i, _p in cell.copies])
    HP = np.array([cell.copies[r["donor"]][1] + cell.Mi @ r["vec"] for r in recs]) if recs else np.zeros((0, 3))
    for k, r in enumerate(recs):
        r["k"] = k
        by_donor.setdefault(r["donor"], []).append(r)
        D = np.linalg.norm(_wrap(HP[k][None, :] - P) @ cell.M.T, axis=1)
        hit = np.nonzero(D < lim)[0]
        # (located H are as found in the CIF: their contacts are reported by the analysis, not judged here)
        r["clash"] = [] if r["h"].located else [(int(c), float(D[c])) for c in hit if c != r["donor"]]
    hh = {}
    if len(recs) > 1:
        DH = np.linalg.norm(_wrap(HP[:, None, :] - HP[None, :, :]) @ cell.M.T, axis=2)
        for k1, k2 in zip(*np.nonzero(np.triu(DH < s.hh_severe, 1))):
            if recs[k1]["h"].located and recs[k2]["h"].located:
                continue
            if recs[k1]["donor"] != recs[k2]["donor"]:
                hh.setdefault(int(k1), []).append((int(k2), float(DH[k1, k2])))

    def need(i):
        if hrole.get(i):
            return 0
        c = bv.cls[i]
        if c == CLASS_POH:
            return None                      # partial hydroxyl: with or without its H
        if c in DONOR_CLASSES:
            return H_PER_CLASS[c]
        return sum(1 for h in hs if h.primary and h.located and h.donor == i)

    varying = set().union(*[set().union(*[m for m, _w in c]) for c in components]) if components else set()
    roles = bv.roles
    groups, index = [], {}
    n_conf = 0
    for combo in combos:
        n_conf += 1
        if progress and n_conf % 200 == 0:
            progress(n_conf, min(n_all, SUB_MAX_CONFIGS))
        present = set(always)
        weight = 1.0
        for m, w in combo:
            present |= m
            weight *= w
        chosen, reasons = [], []
        for cd in sorted(present):
            i = cell.copies[cd][0]
            if st.atoms[i].el == "H":
                continue
            lst = by_donor.get(cd, [])
            n = need(i)
            if n == 0 and not lst:
                continue
            loc = [r for r in lst if r["h"].located]
            hyp = [r for r in lst if not r["h"].located and (r["acc"] is None or r["acc"] in present)
                   and ("alt_of" not in r or r["alt_of"]["acc"] not in present)]
            hyp.sort(key=lambda r: ("alt_of" in r, -(getattr(r["h"], "occf", 1.0) * r["h"].p)))
            kept = []
            for r in hyp:
                if all(np.linalg.norm(r["vec"] - q["vec"]) > 0.35 for q in kept + loc):
                    kept.append(r)
            room = (1 if n is None else n) - len(loc)
            kept = kept[:max(0, room)]
            if n is not None and len(loc) + len(kept) < n:
                reasons.append("%s lacks %d H (its acceptor is absent here, no occupied alternative can take the "
                               "H-bond)" % (cell.label(cd), n - len(loc) - len(kept)))
            chosen += loc + kept
        # clashes of the chosen H with the atoms present and with each other
        ks = {r["k"] for r in chosen}
        for r in chosen:
            for c, d in r["clash"]:
                if c in present:
                    reasons.append("H on %s %.2f Å from %s" % (cell.label(r["donor"]), d, cell.label(c)))
            for k2, d in hh.get(r["k"], ()):
                if k2 in ks:
                    reasons.append("H···H %.2f Å (%s / %s)" % (d, cell.label(r["donor"]),
                                                               cell.label(recs[k2]["donor"])))
        var = sorted(present & varying)
        key = min(tuple(sorted(perm[c] for c in var)) for perm in cell.perms)
        cnt = {}
        for c in var:
            lab = st.atoms[cell.copies[c][0]].label
            cnt[lab] = cnt.get(lab, 0) + 1
        desc = ", ".join("%s%s" % (k, " ×%d" % v if v > 1 else "") for k, v in sorted(cnt.items()))
        moved = sorted({"%s→%s" % (cell.label(r["donor"]), cell.label(r["acc"])) for r in chosen if "alt_of" in r})
        conf = {"present": frozenset(present), "hrecs": chosen, "weight": weight, "desc": desc,
                "reasons": sorted(set(reasons)), "moved": moved}
        g = index.get(key)
        if g is None:
            g = {"key": key, "members": [], "weight": 0.0, "valid": not reasons, "reasons": conf["reasons"]}
            index[key] = g
            groups.append(g)
        g["members"].append(conf)
        g["weight"] += weight
    groups.sort(key=lambda g: (not g["valid"], -g["weight"]))
    for g in groups:
        g["members"].sort(key=lambda m: -m["weight"])
    nv = sum(1 for g in groups if g["valid"])
    note = ("%s configuration(s) of the disordered positions%s; %d symmetry-distinct, %d of them valid networks."
            % ("{:,}".format(n_conf), " (randomly sampled from %s)" % "{:,}".format(n_all) if sampled else "",
               len(groups), nv))
    return {"cell": cell, "groups": groups, "note": note, "n_conf": n_conf, "sampled": sampled}


def subscheme_structure(st, bv, cell, conf, number, hs_parent_located=True):
    """Ordered P1 structure of one configuration: the atoms present (occupancy 1), the located H of its
    network as atoms; the inferred H go into st2.sub_spec (re-created by subscheme_analysis). Classes and
    valences are those of the disordered parent (st2.forced_classes / forced_valences / forced_roles)."""
    import copy
    st2 = copy.copy(st)
    atoms, idx = [], {}
    for c in sorted(conf["present"]):
        i, p = cell.copies[c]
        a = st.atoms[i]
        idx[c] = len(atoms)
        atoms.append(Atom(cell.label(c), a.el, [float(x) for x in p], 1.0, a.uiso, ".", "."))
    with_h = {}
    for r in conf["hrecs"]:
        with_h[r["donor"]] = with_h.get(r["donor"], 0) + 1
    spec = []
    for r in conf["hrecs"]:
        h = r["h"]
        if h.located:
            f = cell.copies[r["donor"]][1] + cell.Mi @ r["vec"]
            lab = h.located if len(cell.by_site[h.donor]) == 1 else "%s_%d" % (
                h.located, cell.by_site[h.donor].index(r["donor"]) + 1)
            atoms.append(Atom(lab, "H", [float(x) for x in f], 1.0, None, ".", "."))
        else:
            spec.append((idx[r["donor"]], tuple(float(x) for x in r["vec"]), idx.get(r["acc"]),
                         tuple(float(x) for x in r["accvec"]) if r["accvec"] is not None else None,
                         bool(getattr(h, "fixlen", False)), getattr(h, "grg_len", None)))
    st2.atoms = atoms
    st2.symops = [([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.0, 0.0, 0.0])]
    st2.spacegroup = "P 1"
    st2._env = {}
    for attr in ("_mult", "_mixed", "_Minv", "excl", "_alt_sites", "_alt_key", "optimized", "opt_note",
                 "sub_subs"):
        if hasattr(st2, attr):
            setattr(st2, attr, None)
    st2._disordered = set()
    st2.formula, st2.formula_text = None, "?"
    st2.res_text = st2.hkl_text = None
    st2.sym_warning = None
    st2.tag = " [subscheme %d]" % number
    st2.sub_number = number
    st2.sub_parent = st.title
    st2.sub_desc = conf["desc"]
    st2.sub_spec = spec
    st2.sub_reasons = list(conf.get("reasons", []))
    st2.sub_moved = list(conf.get("moved", []))
    st2.sub_units = len(cell.ops)          # asymmetric units in the P1 cell (scores per unit = P1 / this)
    fc, fv, fr = {}, {}, {}
    for c, k in idx.items():
        i = cell.copies[c][0]
        if bv.roles[i] == "anion" and bv.cls[i]:
            cls = bv.cls[i]
            if cls == CLASS_POH:
                cls = CLASS_OH if with_h.get(c) else CLASS_PROBABLE
            fc[k] = cls
            if getattr(bv, "hrole", {}).get(i):
                fr[k] = bv.hrole[i]
        elif bv.roles[i] == "cation" and bv.valence_spec.get(i) is not None:
            fv[k] = bv.valence_spec[i]
    st2.forced_classes, st2.forced_valences, st2.forced_roles = fc, fv, fr
    return st2


def _single_option(d, cands, hs, s):
    o = Option()
    o.donor, o.cands, o.hs = d, tuple(cands), hs
    o.angle = angle_between(cands[0].nb.vec, cands[1].nb.vec) if len(cands) == 2 else None
    o.hdh = angle_between(hs[0].vec, hs[1].vec) if len(hs) == 2 else None
    o.adc_pen = sum(h.cand.adc_pen for h in hs)
    o.geom = geom_penalty(list(d["located"]) + hs, s)
    o.angle_pen = o.geom - o.adc_pen
    o.incomplete, o.passes, o.impossible, o.flips, o.number = False, True, "", [], 1
    o.donorF = d["bvs"] - sum(h.s_fi * h.p for h in list(d["located"]) + hs)
    o.local = o.geom
    return o


def subscheme_analysis(st1, params, s, overrides=None, class_overrides=None, valence_overrides=None,
                       role_overrides=None):
    """Analysis of a subscheme structure (subscheme_structure) with ITS network as the only scheme:
    returns (bv, located H used, donors, scheme result, all located H)."""
    rov = dict(getattr(st1, "forced_roles", {}) or {})
    rov.update(role_overrides or {})
    bv1 = analyse_bv(st1, params, s, overrides, class_overrides, valence_overrides)
    apply_site_roles(st1, bv1, s, rov)
    loc_all = located_hatoms(st1, bv1, params, s)
    loc = [] if s.ignore_located else loc_all
    donors = infer_hbonds(st1, bv1, params, s, loc)
    env = st1.environment(s.env_radius)
    spec = {}
    for sp in getattr(st1, "sub_spec", []) or []:
        spec.setdefault(sp[0], []).append(sp)
    missing = []
    for d in donors:
        if not d["n_missing"]:
            continue
        cands, hs = [], []
        for (i, vec, acc, accvec, fixlen, grg) in spec.get(d["i"], []):
            if acc is None or accvec is None:
                continue
            c = next((c for c in d["cands"] if c.nb.j == acc and math.dist(c.nb.vec, accvec) < 0.05), None)
            if c is None:
                nb = next((nb for nb in env[i] if nb.j == acc and math.dist(nb.vec, accvec) < 0.05), None)
                if nb is None:
                    missing.append(st1.atoms[i].label)
                    continue
                c = _make_candidate(st1, bv1, s, params, nb, d["cations"])
            h = _make_h(st1, i, c, vec, s, params, grg_len=grg if fixlen else None)
            h.fixlen = fixlen
            cands.append(c)
            hs.append(h)
        d["options"] = [_single_option(d, cands, hs, s)] if hs else []
    res = search_schemes(st1, bv1, donors, loc, s, params)
    if not res.schemes:
        res = search_schemes(st1, bv1, donors, loc, replace(s, show_severe=True), params)
    if missing:
        bv1.notes.append("⚠ Subscheme H that could not be re-created: on %s." % ", ".join(sorted(set(missing))))
    bv1.notes.insert(0, "Subscheme %d of %s (ordered P1 configuration; occupied disordered sites: %s). Classes, "
                        "valences and H-bond roles as in the disordered structure." % (
                            getattr(st1, "sub_number", 0), getattr(st1, "sub_parent", "?"),
                            getattr(st1, "sub_desc", "") or "-"))
    if getattr(st1, "sub_moved", None):
        bv1.notes.insert(1, "H-bonds moved to the occupied alternative acceptor position: " + ", ".join(st1.sub_moved))
    if getattr(st1, "sub_reasons", None):
        bv1.notes.insert(1, "⚠ INVALID network (shown for inspection): " + "; ".join(st1.sub_reasons))
    return bv1, loc, donors, res, loc_all


def subscheme_cif_text(st1, bv1, hs, note=""):
    """P1 CIF of a subscheme structure with its H (located H are atoms of st1, inferred H from hs)."""
    labels = label_hydrogens(st1, hs)
    f, z = scheme_formula(st1, bv1, hs)
    fw = formula_weight(f)
    q = formula_charge(st1, bv1, h_cell=h_per_cell(st1, hs))
    name = re.sub(r"[^A-Za-z0-9_]+", "_", "%s_subscheme%d" % (st1.name, getattr(st1, "sub_number", 0)))
    L = ["data_%s" % name,
         "# HydroGEN subscheme %d of %s: one ordered configuration of the disordered structure in P1" % (
             getattr(st1, "sub_number", 0), getattr(st1, "sub_parent", "?")),
         "# occupied disordered sites: %s" % (getattr(st1, "sub_desc", "") or "-")] + \
        (["# H-bonds moved to the occupied alternative acceptor position: %s" % ", ".join(st1.sub_moved)]
         if getattr(st1, "sub_moved", None) else []) + \
        (["# INVALID network (exported for inspection): %s" % r for r in st1.sub_reasons]
         if getattr(st1, "sub_reasons", None) else []) + [
         "# oxidation states %s; charge per formula unit %+.2f%s" % (oxidation_state_text(st1, bv1), q["total"],
                                                                      ("; " + note) if note else ""),
         "# hypothetical H (calc) are not refined against diffraction data",
         "_audit_creation_method 'HydroGEN subscheme (P1)'",
         "_symmetry_space_group_name_H-M 'P 1'",
         "_space_group_name_H-M_alt 'P 1'",
         "_chemical_formula_sum '%s'" % formula_text(f, order=[e for e in f if e != "H"] + ["H"])]
    if fw is not None:
        L.append("_chemical_formula_weight %.2f" % fw)
    if st1.Z:
        L.append("_cell_formula_units_Z %g" % st1.Z)
    L += ["_cell_length_a %.5f" % st1.a, "_cell_length_b %.5f" % st1.b, "_cell_length_c %.5f" % st1.c,
          "_cell_angle_alpha %.4f" % st1.alpha, "_cell_angle_beta %.4f" % st1.beta,
          "_cell_angle_gamma %.4f" % st1.gamma, "",
          "loop_", "_space_group_symop_operation_xyz", "'x, y, z'", "",
          "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
          "_atom_site_fract_z", "_atom_site_U_iso_or_equiv", "_atom_site_occupancy", "_atom_site_calc_flag"]
    for a in st1.atoms:
        u = a.uiso if a.uiso else (0.05 if a.el == "H" else None)
        L.append("%-10s %-3s %9.5f %9.5f %9.5f %8s 1 %s" % (a.label, a.el, a.f[0], a.f[1], a.f[2],
                                                           ("%.4f" % u) if u else "?", "."))
    for h in sorted((h for h in hs if not h.located and h.primary), key=lambda x: labels[id(x)]):
        D = st1.atoms[h.donor]
        u = 1.5 * D.uiso if D.uiso else 0.05
        L.append("%-10s %-3s %9.5f %9.5f %9.5f %8.4f 1 calc" % (labels[id(h)], "H", h.frac[0], h.frac[1],
                                                               h.frac[2], u))
    return "\n".join(L) + "\n"


def subscheme_paths(source, numbers, optimized=False):
    """Output paths: one subscheme -> <name>_hypothetical_subscheme<n>.cif next to the source; several ->
    a new folder <name>_hypothetical_subschemes (numbered if it exists) with <...>_subscheme<n>.cif.
    Returns (folder or None, {n: path})."""
    base = os.path.splitext(source)[0] + ("_optimized_hypothetical" if optimized else "_hypothetical")
    if len(numbers) == 1:
        n = numbers[0]
        p, k = "%s_subscheme%d.cif" % (base, n), 1
        while os.path.exists(p):
            p = "%s_subscheme%d_%d.cif" % (base, n, k)
            k += 1
        return None, {n: p}
    folder, k = base + "_subschemes", 1
    while os.path.exists(folder):
        folder = "%s%d_subschemes" % (base, k)
        k += 1
    stem = os.path.basename(folder)[:-len("_subschemes")]
    return folder, {n: os.path.join(folder, "%s_subscheme%d.cif" % (stem, n)) for n in numbers}


def network_dimensionality(st, bv, hs):
    """Dimensionality of the periodic net formed by the summed cation-anion bonds (ammonium N...X bonds
    included) and the H-bonds of hs (donor -> acceptor) in an ORDERED structure in P1 (a subscheme).
    Returns (rank 0-3 of the largest component's translation lattice, text). 3 = linked in all
    directions; 2 = sheets/layers that the bonds and H-bonds do not link to each other (the normal of the
    sheets is given), 1 = chains, 0 = isolated groups."""
    n = len(st.atoms)
    Mi = np.linalg.inv(st.M)
    F = np.array([a.f for a in st.atoms], dtype=float)
    edges = [[] for _ in range(n)]

    def add(i, j, vec):
        t = np.round(F[i] + Mi @ np.asarray(vec, float) - F[j]).astype(int)
        edges[i].append((j, t))
        edges[j].append((i, -t))
    for i in range(n):
        for bd in bv.bonds[i]:
            if bd.included and not bd.to_h and i < bd.nb.j:
                add(i, bd.nb.j, bd.nb.vec)
            elif bd.included and not bd.to_h and i == bd.nb.j:
                add(i, i, bd.nb.vec)
    for h in hs:
        if h.cand is not None and h.p > 0:
            add(h.donor, h.cand.acceptor, h.cand.nb.vec)
    off = [None] * n
    best = (0, -1, [])
    for s0 in range(n):
        if off[s0] is not None or st.atoms[s0].el == "H":
            continue
        off[s0] = np.zeros(3, int)
        comp, vecs, stack = [s0], [], [s0]
        while stack:
            i = stack.pop()
            for j, t in edges[i]:
                if off[j] is None:
                    off[j] = off[i] + t
                    comp.append(j)
                    stack.append(j)
                else:
                    d = off[i] + t - off[j]
                    if d.any():
                        vecs.append(d)
        rank = int(np.linalg.matrix_rank(np.array(vecs, float))) if vecs else 0
        if (rank, len(comp)) > (best[0], best[1]):
            best = (rank, len(comp), vecs)
    rank, _size, vecs = best
    if rank == 3:
        return 3, "3D: linked in all directions"
    if rank == 2:
        V = np.array(vecs, float)
        u = V[np.argmax(np.abs(V).sum(1))]
        w = next(v for v in V if np.linalg.matrix_rank(np.array([u, v])) == 2)
        nrm = np.cross(u, w)                     # sheet normal in reciprocal-lattice indices (h k l)
        g = np.gcd.reduce(np.abs(nrm).astype(int)) or 1
        hkl = tuple(int(x) for x in (nrm / g))
        return 2, "2D: sheets parallel to (%d %d %d) are not linked to each other by bonds or H-bonds" % hkl
    if rank == 1:
        return 1, "1D: chains only"
    return 0, "0D: isolated units"


def subscheme_connectivity(st1, bv1, hs1, donors):
    """(dimensionality of the network of hs1, dimensionality reachable with ALL passing candidate H-bonds of
    the donors, text). A network that links fewer directions than its candidates could - e.g. a water
    double layer whose H-bonds all point to the sheets, so the sheets are not held together - is not
    a plausible arrangement."""
    import types
    dim, txt = network_dimensionality(st1, bv1, hs1)
    pseudo = [types.SimpleNamespace(donor=d["i"], cand=c, p=1.0) for d in donors for c in d["cands"] if c.passes]
    pot, _t = network_dimensionality(st1, bv1, list(hs1) + pseudo)
    return dim, pot, txt


def research_subschemes(st, bv, s, params, sub, top=3, max_conf=40, progress=None):
    """'Search each arrangement anew': instead of inheriting the H of the source scheme, every symmetry-
    distinct arrangement of sub (find_subschemes; the most frequent max_conf of them) is built in P1 and
    its own best H-bond networks are searched (classes, valences and roles of the disordered structure).
    The top networks of each become subscheme groups (with their evaluation). Returns the new groups."""
    import copy
    cell = sub["cell"]
    base = sorted(sub["groups"], key=lambda g: -g["weight"])[:max_conf]
    out = []
    for k, g in enumerate(base):
        if progress:
            progress(k + 1, len(base))
        conf = dict(g["members"][0])
        conf["hrecs"] = [r for r in conf["hrecs"] if r["h"].located]     # keep located H only
        conf["reasons"], conf["moved"] = [], []
        st0 = subscheme_structure(st, bv, cell, conf, 0)
        st0.sub_spec = None                                             # searched like any structure
        bv1, loc, donors, res, loc_all = analyse_and_search(st0, params, s, None, None, None,
                                                            dict(st0.forced_roles))
        bv1.notes.insert(0, "Subscheme (arrangement searched anew) of %s: occupied disordered sites %s." % (
            st.title, conf["desc"] or "-"))
        if not res.schemes:
            why = {}
            for i, txt in res.hard_fail.items():
                why.setdefault(txt, []).append(st0.atoms[i].label)
            out.append({"key": (g["key"], 0), "members": [conf], "weight": g["weight"], "equiv": len(g["members"]),
                        "valid": False, "reasons": ["no H-bond network found for this arrangement" + (
                            ": " + "; ".join("%s - %s" % (", ".join(sorted(labs)), txt) for txt, labs in why.items())
                            if why else "")], "searched": True})
            continue
        picked = []                              # distinct networks (equal scores = symmetry-equivalent copies)
        for r, sc in enumerate(res.schemes):
            if all(abs(sc["total"] - res.schemes[q]["total"]) > 0.01 for q in picked):
                picked.append(r)
            if len(picked) >= top:
                break
        for nr, r in enumerate(picked):
            st1 = st0 if nr == 0 else copy.copy(st0)
            hs1 = scheme_hatoms(res, res.schemes[r])
            ev = evaluate_hset(st0, bv1, hs1, s, params)
            reasons = []
            if res.hard_fail:
                reasons.append("; ".join("%s: %s" % (st0.atoms[i].label, txt) for i, txt in res.hard_fail.items()))
            if ev["sev"]:
                reasons.append("H···H below %.2f Å" % s.hh_severe)
            dim, pot, txt = subscheme_connectivity(st0, bv1, hs1, donors)
            if getattr(s, "sub_require_link", True) and dim < pot:
                reasons.append("layers not linked - %s (the candidate H-bonds could link it in %dD)" % (txt, pot))
            c2 = dict(conf)
            c2["desc"] = conf["desc"] + ("" if nr == 0 else "  [network %d]" % (nr + 1))
            out.append({"key": (g["key"], r), "members": [c2], "weight": g["weight"], "equiv": len(g["members"]),
                        "valid": not reasons, "reasons": reasons, "searched": True, "dim": dim,
                        "eval": {"st": st1, "bv": bv1, "loc": loc, "donors": donors, "res": res, "loc_all": loc_all,
                                 "hs": hs1, "ev": ev, "scheme_index": r}})
    out.sort(key=lambda g: (not g["valid"], -g["weight"]))
    return out


def cell_contents(st, bv):
    """Unit-cell contents of the model as on the Bond valence tab: every non-H site occupancy x multiplicity,
    H from the D/A classes (as set by hand: water, hydroxyl, partial hydroxyl x, NH4+ ...). Independent of Z."""
    mult = site_multiplicities(st)
    cell = {}
    for i, a in enumerate(st.atoms):
        if a.el != "H":
            cell[a.el] = cell.get(a.el, 0.0) + a.occ * mult[i]
    h = class_h_per_cell(st, bv)
    if h > 1e-6:
        cell["H"] = h
    return cell


def effective_z(st, bv):
    """(Z, note): formula units per cell for comparisons. With a _chemical_formula_sum, Z is derived from the
    sites (cell contents / formula, median over the non-H elements in both, rounded when close to a whole
    number); if that differs from the CIF's Z, the derived value is used and the note says so."""
    cell = cell_contents(st, bv)
    zc = st.Z
    if st.formula:
        ratios = sorted(cell[e] / n for e, n in st.formula.items() if e != "H" and n > 0 and cell.get(e, 0) > 0)
        if ratios:
            z = ratios[len(ratios) // 2]
            if abs(z - round(z)) <= 0.15 and round(z) >= 1:
                z = float(round(z))
            if not zc or abs(z - zc) > 0.05 * max(z, 1):
                return z, "Z renormalised from %s to %g (cell contents / formula)" % (
                    ("%g" % zc) if zc else "none", z)
            return float(zc), ""
    return float(zc or 1.0), ("" if zc else "no Z in the CIF - 1 assumed")


def subscheme_composition(st1, hs1):
    """Contents of a subscheme's P1 cell: its atoms (occupancy 1) plus the H of its network."""
    out = {}
    for a in st1.atoms:
        if a.el != "H":
            out[a.el] = out.get(a.el, 0.0) + 1.0
    nh = sum(1 for a in st1.atoms if a.el == "H")
    nh += sum(1 for h in hs1 if h.primary and not h.located)
    if nh:
        out["H"] = float(nh)
    return out


def composition_check(ref_cell, sub_cell, z):
    """(ok, text): every element within 1 atom per formula unit (z atoms per cell) of the reference cell."""
    diffs = []
    ok = True
    for e in sorted(set(ref_cell) | set(sub_cell)):
        d = sub_cell.get(e, 0.0) - ref_cell.get(e, 0.0)
        if abs(d) > z + 1e-6:
            ok = False
        if abs(d) > 1e-3:
            diffs.append("%s %+.2f" % (e, d / z))
    return ok, ", ".join(diffs) if diffs else "same"
