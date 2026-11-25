#!/usr/bin/env python3
"""
HydroGEN (hbond_gui.py) - bond-valence and hydrogen-bond analysis of CIF files.

    python hbond_gui.py        (or double-click HydroGEN.exe)

Tabs
  Bond valence      BV sum per site (with and without H); click a site to see its bonds,
                    double-click a bond to force it in / out of that atom's sum.
  Inferred H-bonds  donors (water / hydroxyl by BV w/o H, or O carrying located H), their
                    located H, candidate acceptors and options for the missing H.
  Schemes           whole-structure combinations ranked by bond-valence compatibility
                    (located H always included), F&I and GRG sums side by side.
  Refinement        rotate the inferred H of a scheme (max. N degrees) to fit GRG sums and
                    H...H distances; compare refined / unrefined schemes and the CIF's own H.
  Subschemes        disordered structures: the ordered P1 arrangements and the H-bond network
                    of a scheme in each; open one as a structure, refine, export P1 CIFs.
  BV table          new-mineral style bond-valence table of the selected scheme, comparison
                    entry or Finalize selection; save as .txt / .csv or copy.
  Located H-bonds   geometric D-H...A search for H atoms in the CIF.
  Environment       neighbours within 6 A of a site, and angles.
  Finalize & CIF    tick the hypothetical H atoms to keep and write <name>_hypothetical.cif.

Calculations: hbond_core.py; bond-valence parameters: bv_params.txt.
"""

import csv
import datetime
import json
import os
import re
import sys
import threading
import traceback
import urllib.request
import webbrowser

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

import numpy as np

import hbond_core as hc

RANK_CHOICES = {"F&I only": "fi", "GRG only": "grg", "GRG + F&I (mean)": "both"}
CHECK, UNCHECK, LOCKED = "☑", "☐", "■"
TAB_DISORDER, TAB_SCHEMES, TAB_REFINE, TAB_SUBS, TAB_DETAILS, TAB_TABLE, TAB_FINAL = 2, 3, 4, 5, 8, 9, 10
SUB_SRC_SCHEME = "Schemes tab: selected scheme"
SUB_SRC_FINAL = "Finalize: current selection"
SUB_SRC_CMP = "Refinement: "
APP_VERSION = "1.3"                       # must match the GitHub release tag (V1.3) of this build
GITHUB_REPO = "sigogglinite/HydroGEN"     # Update downloads the latest release's HydroGEN.zip from here
UPDATE_ZIP = "HydroGEN.zip"               # release asset: the zipped program folder made by build_exe.ps1
APP_TITLE = "HydroGEN %s - A program for bond valence and H-bond analysis" % APP_VERSION
DIS_MODES = {"donors": "donor", "acceptors only": "acceptor", "excluded": "exclude"}
ROLE_TEXT = {"normal": "normal", "acceptor": "acceptor only", "exclude": "excluded from H-bond search"}
APP_NAME = "HydroGEN"
REFINE_MODES = {"All inferred H": "all", "Worst-fitting donor(s)": "worst", "Only penalised parts": "penalised"}
PROJECT_EXT = ".hgen"
# tags whose background is a pastel colour (need dark text in dark mode) / text-colour tags and their dark variants
BG_TAGS = ("water", "hydroxyl", "probable", "possible", "red", "mod", "fail", "donor", "warn", "bad", "group")
FG_DARK = {"#888": "#9ca3af", "#999": "#9ca3af", "#555": "#9ca3af", "#b45309": "#fbbf24",
           "#1d4ed8": "#93c5fd", "#b91c1c": "#f87171"}


import base64
import io
import pickle
import zlib


# ================================================================== updates (GitHub releases)
def version_tuple(v):
    """'V1.10' / 'v1.2.3' / '1.1' -> (1, 10) / (1, 2, 3) / (1, 1); trailing zeros dropped (1.0 == 1)."""
    nums = [int(x) for x in re.findall(r"\d+", str(v))] or [0]
    while len(nums) > 1 and nums[-1] == 0:
        nums.pop()
    return tuple(nums)


def _gh_request(url):
    return urllib.request.Request(url, headers={"User-Agent": "HydroGEN-updater",
                                                "Accept": "application/vnd.github+json"})


def _ssl_context():
    """HTTPS certificates: the certifi bundle shipped with HydroGEN (Python's own OpenSSL cannot always verify
    GitHub from the Windows store alone), else the system default."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _urlopen(req, timeout):
    return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())


def latest_release(repo=None, timeout=15):
    """Latest published release of the GitHub repository: {'tag', 'page', 'notes', 'asset': {'url', 'size'} or
    None} - the asset is the release's HydroGEN.zip (the zipped program folder made by build_exe.ps1)."""
    repo = repo or GITHUB_REPO
    url = "https://api.github.com/repos/%s/releases/latest" % repo
    with _urlopen(_gh_request(url), timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    if "tag_name" not in d:
        raise RuntimeError(d.get("message") or "no published release found")
    a = next((x for x in d.get("assets", []) if x.get("name", "").lower() == UPDATE_ZIP.lower()), None)
    return {"tag": d.get("tag_name", "?"), "page": d.get("html_url", "https://github.com/%s/releases" % repo),
            "notes": d.get("body") or "",
            "asset": {"url": a["browser_download_url"], "size": int(a.get("size") or 0)} if a else None}


def download_file(url, path, size, state):
    """Download url to path (state['done'] = bytes so far; state['cancel'] stops it). The file must have the
    release's size, otherwise it is deleted and an error raised."""
    tmp = path + ".part"
    try:
        with _urlopen(urllib.request.Request(url, headers={"User-Agent": "HydroGEN-updater"}), 30) as r, \
                open(tmp, "wb") as fh:
            while True:
                if state.get("cancel"):
                    raise RuntimeError("cancelled")
                chunk = r.read(65536)
                if not chunk:
                    break
                fh.write(chunk)
                state["done"] += len(chunk)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    got = os.path.getsize(tmp)
    if size and got != size:
        os.remove(tmp)
        raise RuntimeError("incomplete download (%d of %d bytes)" % (got, size))
    os.replace(tmp, path)


def app_folder():
    """Folder of the installed (folder-build) program - the one holding HydroGEN.exe and _internal - or None
    when running from the Python source or from an old single-file build."""
    if not getattr(sys, "frozen", False):
        return None
    d = os.path.dirname(os.path.abspath(sys.executable))
    return d if os.path.isdir(os.path.join(d, "_internal")) else None


def _zip_program_root(z):
    """Folder inside the update zip that holds HydroGEN.exe and _internal/ ('' or e.g. 'HydroGEN/')."""
    names = z.namelist()
    for n in names:
        parts = n.split("/")
        if parts[-1].lower() == "hydrogen.exe" and len(parts) <= 2:
            root = n[:len(n) - len(parts[-1])]
            if any(m.startswith(root + "_internal/") for m in names):
                return root
    raise RuntimeError("the downloaded zip does not contain HydroGEN.exe and its _internal folder")


def install_update_zip(zpath, app_dir):
    """Replace the program in app_dir (HydroGEN.exe + _internal) with the one in the update zip. The current
    program files are moved into app_dir/HydroGEN_old (an older backup there is deleted first), then the new
    files moved in. Windows allows renaming files of a running program (not deleting them), so this works while
    HydroGEN is open - but the running HydroGEN keeps reading its code from HydroGEN.exe, so it must be closed
    right afterwards. Anything else in app_dir (your own files) is left alone. If a step fails, every file is
    moved back and the error raised. Returns the backup folder."""
    import shutil
    import zipfile
    stage, old = os.path.join(app_dir, "HydroGEN_update"), os.path.join(app_dir, "HydroGEN_old")
    for d in (stage, old):
        if os.path.exists(d):
            shutil.rmtree(d)
    with zipfile.ZipFile(zpath) as z:
        root = _zip_program_root(z)
        bad = z.testzip()
        if bad:
            raise RuntimeError("the downloaded zip is damaged (%s)" % bad)
        z.extractall(stage)
    src = os.path.join(stage, *root.split("/"))

    def files(base, tops):
        out = []
        for t in tops:
            p = os.path.join(base, t)
            if os.path.isfile(p):
                out.append(t)
            elif os.path.isdir(p):
                for dp, _dn, fn in os.walk(p):
                    out += [os.path.relpath(os.path.join(dp, f), base) for f in fn]
        return out

    tops = sorted(set(os.listdir(src)) | {"HydroGEN.exe", "_internal"})
    moves = []

    def mv(a, b):
        os.makedirs(os.path.dirname(b), exist_ok=True)
        os.rename(a, b)
        moves.append((a, b))

    try:
        for rel in files(app_dir, tops):
            mv(os.path.join(app_dir, rel), os.path.join(old, rel))
        for rel in files(src, tops):
            mv(os.path.join(src, rel), os.path.join(app_dir, rel))
    except BaseException:
        for a, b in reversed(moves):
            try:
                os.makedirs(os.path.dirname(a), exist_ok=True)
                os.rename(b, a)
            except OSError:
                pass
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
        raise
    shutil.rmtree(stage, ignore_errors=True)
    for dp, _dn, _fn in sorted(os.walk(os.path.join(app_dir, "_internal")), key=lambda w: -len(w[0])):
        try:
            os.rmdir(dp)                     # leftover empty folders of the old version (non-empty ones stay)
        except OSError:
            pass
    return old

STATE_VERSION = 1


class _StateUnpickler(pickle.Unpickler):
    """Loads a saved analysis state: only HydroGEN's own classes, numpy arrays and plain containers - never
    arbitrary code (a .hgen file may come from someone else)."""
    _NUMPY = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
              ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
              ("numpy._core.multiarray", "scalar"), ("numpy.core.numeric", "_frombuffer"),
              ("numpy._core.numeric", "_frombuffer")}
    _BUILTINS = {"set", "frozenset", "complex", "slice", "range"}

    def find_class(self, module, name):
        if module == "hbond_core" and not name.startswith("_") and isinstance(getattr(hc, name, None), type):
            return getattr(hc, name)
        if (module, name) in self._NUMPY or (module == "numpy" and name.startswith("dtypes")):
            return super().find_class(module, name)
        if module == "numpy.dtypes" and name.endswith("DType"):
            return super().find_class(module, name)
        if module == "builtins" and name in self._BUILTINS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError("not allowed in a HydroGEN analysis file: %s.%s" % (module, name))


def dump_state(obj):
    return base64.b64encode(zlib.compress(pickle.dumps(obj, protocol=4), 6)).decode("ascii")


def load_state(text):
    return _StateUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(text)))).load()


def _prefs_path():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "HydroGEN", "prefs.json")


def load_prefs():
    try:
        with open(_prefs_path(), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_prefs(p):
    try:
        os.makedirs(os.path.dirname(_prefs_path()), exist_ok=True)
        with open(_prefs_path(), "w", encoding="utf-8") as fh:
            json.dump(p, fh, indent=1)
    except OSError:
        pass


def _resource(name):
    """File shipped with the program: next to the .exe / script, or bundled inside the .exe."""
    import sys
    places = []
    if getattr(sys, "frozen", False):
        places += [os.path.dirname(os.path.abspath(sys.executable)), getattr(sys, "_MEIPASS", "")]
    places.append(os.path.dirname(os.path.abspath(__file__)))
    for d in places:
        p = os.path.join(d, name)
        if d and os.path.exists(p):
            return p
    return None


def _fmt(x, nd=3):
    return "" if x is None else ("%.*f" % (nd, x))


def _val_text(v, role):
    if v is None:
        return "?"
    if abs(v - round(v)) > 1e-6:                 # mixed valence: average site charge
        return ("%.2f+" % v) if role == "cation" else ("%.2f-" % abs(v))
    return ("%d+" % round(v)) if role == "cation" else ("%d-" % abs(round(v)))


def accent_button(parent, text, command, font=None):
    """Prominent (dark blue) action button; plain tk so the colour also shows in the native light theme."""
    return tk.Button(parent, text=text, command=command, bg="#1e40af", fg="white", activebackground="#2563eb",
                     activeforeground="white", disabledforeground="#9ca3af", relief="flat", padx=10, pady=2,
                     cursor="hand2", font=font)


class ScrolledTree(ttk.Frame):
    """Treeview + scrollbars. columns: list of (key, heading, width, anchor)."""
    instances = []

    def __init__(self, master, columns, tree_heading=None, tree_width=160, height=8, selectmode="browse"):
        super().__init__(master)
        ScrolledTree.instances.append(self)
        show = "tree headings" if tree_heading else "headings"
        self.tree = ttk.Treeview(self, columns=[c[0] for c in columns], show=show, height=height,
                                 selectmode=selectmode)
        if tree_heading:
            self.tree.heading("#0", text=tree_heading)
            self.tree.column("#0", width=tree_width, stretch=False)
        for key, head, width, anchor in columns:
            self.tree.heading(key, text=head)
            self.tree.column(key, width=width, anchor=anchor, stretch=False)
        ys = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        xs = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def clear(self):
        self.tree.delete(*self.tree.get_children())


class BVTable(ScrolledTree):
    """Side-by-side BV sums with H: F&I (cations + accepted − donated, own O–H not counted)
    vs GRG (G&H 2015 H–O parameters, D–H optimised within the allowed range)."""

    def __init__(self, master, height=8):
        super().__init__(master, [
            ("site", "Site", 70, "w"), ("el", "El", 30, "w"), ("cls", "Class / H", 150, "w"),
            ("base", "BVS w/o H", 72, "e"),
            ("fdon", "F&I: donated", 88, "e"), ("facc", "F&I: accepted", 92, "e"),
            ("ftot", "F&I: Σ", 65, "e"), ("ftar", "F&I: target", 85, "center"), ("fout", "F&I: outside", 85, "e"),
            ("gdh", "GRG: O–H", 75, "e"), ("gacc", "GRG: accepted", 95, "e"),
            ("gtot", "GRG: Σ", 65, "e"), ("gdev", "GRG: Δ", 60, "e")], height=height)
        self.tree.tag_configure("warn", background="#fef3c7")
        self.tree.tag_configure("bad", background="#fecaca")
        self.tree.tag_configure("hrow", foreground="#555")

    def fill(self, st, bv, hs, s, params, labels=None):
        self.clear()
        r = hc.bv_with_h(st, bv, hs, s, params)
        for i, a in enumerate(st.atoms):
            if not r["anion"][i]:
                continue
            out, dg = r["outF"][i], abs(r["devG"][i])
            tag = ("bad",) if (out >= 0.15 or dg >= 0.25) else (("warn",) if (out >= 0.05 or dg >= 0.10) else ())
            tar = ("%.2f ± %.2f" % (r["cF"][i], r["bF"][i])) if r["bF"][i] else "%.2f" % r["cF"][i]
            self.tree.insert("", "end", tags=tag, values=(
                a.label, a.el, bv.cls[i] or bv.roles[i], _fmt(r["base"][i]),
                ("−%.3f" % r["F_don"][i]) if r["F_don"][i] else "", _fmt(r["F_acc"][i]) if r["F_acc"][i] else "",
                _fmt(r["totF"][i]), tar, _fmt(r["outF"][i]) if r["outF"][i] else "",
                _fmt(r["G_dh"][i]) if r["G_dh"][i] else "", _fmt(r["G_acc"][i]) if r["G_acc"][i] else "",
                _fmt(r["totG"][i]), "%+.3f" % r["devG"][i]))
        for h in sorted(hs, key=lambda x: (x.donor, x.located or "~", x.cand.ordinal if x.cand else 0)):
            k = r["hpos"][id(h)]
            lab = (labels or {}).get(id(h), "H")
            acc = st.atoms[h.cand.acceptor].label if h.cand is not None else "none"
            self.tree.insert("", "end", tags=("hrow",), values=(
                lab + ("" if h.located else "*"), "H", "on %s → %s" % (st.atoms[h.donor].label, acc),
                "D–H %.3f" % r["L"][k], _fmt(h.s_fi) if h.cand is not None else "", "", "", "", "",
                _fmt(r["sdh"][k]), _fmt(r["sha"][k]) if h.cand is not None else "",
                _fmt(r["sdh"][k] + r["sha"][k]), "%+.3f" % (r["sdh"][k] + r["sha"][k] - 1)))
        return r


class SchemeBuilder:
    """Build an H-bond scheme by hand: for every donor pick the acceptor of each missing H from a menu
    (alphabetical, with D...A, symmetry code and the change in total score this choice would make);
    the score, flags and BV sums are recomputed after every choice."""

    GOOD, NEUTRAL, MILD, BAD = "#15803d", "#374151", "#c2410c", "#b91c1c"

    def __init__(self, app):
        self.app = app
        self.st, self.bv, self.s, self.params = app.st, app.bv, app.s, app.params
        self.located = list(app.loc_h)
        self.donors = [d for d in app.donors if d["n_missing"] > 0]
        self.choice = {d["i"]: [None] * d["n_missing"] for d in self.donors}
        self.cache = {}
        self.grg_cache = {}
        self.show_hard = tk.BooleanVar(value=False)
        self.n_made = 0
        st = self.st
        win = self.win = tk.Toplevel(app.root)
        win.title("%s - build a scheme (%s)" % (APP_NAME, st.name))
        win.geometry("1150x780")
        top = ttk.Frame(win, padding=(8, 6))
        top.pack(fill="x")
        self.score_lbl = ttk.Label(top, font=app.bold)
        self.score_lbl.pack(anchor="w")
        self.detail_var = tk.StringVar()
        ttk.Label(top, textvariable=self.detail_var, wraplength=1100, justify="left").pack(anchor="w")
        self.flag = tk.Label(top, text="", anchor="w", justify="left", padx=6, pady=2, wraplength=1100)
        self.flag.pack(fill="x", pady=(3, 0))
        bar = ttk.Frame(win, padding=(8, 0))
        bar.pack(fill="x")
        ttk.Checkbutton(bar, text="Show hard-fail acceptors", variable=self.show_hard).pack(side="right", anchor="n")
        hint = ttk.Label(bar, foreground="#555", justify="left", wraplength=800, text=(
            "Pick the acceptor of each missing H. Menus list the candidates alphabetically with D···A, symmetry "
            "code and the CHANGE IN TOTAL SCORE that choice makes to the present scheme (green better, red worse); "
            "classes and roles: right-click on the Bond valence tab, then reopen."))
        hint.pack(side="left", fill="x", expand=True)
        hint.bind("<Configure>", lambda e: hint.configure(wraplength=max(300, e.width - 4)))
        pw = ttk.PanedWindow(win, orient="vertical")
        pw.pack(fill="both", expand=True, padx=8, pady=4)
        # scrollable donor list
        outer = ttk.Frame(pw)
        pw.add(outer, weight=3)
        canvas = tk.Canvas(outer, highlightthickness=0, background=app.root.cget("background"))
        ys = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        rows = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=rows, anchor="nw")
        rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        def wheel(e):                          # scroll the donor list (not the BV table below it)
            if str(e.widget).startswith(str(canvas)):
                canvas.yview_scroll(int(-e.delta / 120), "units")
        win.bind("<MouseWheel>", wheel)
        for col, txt in enumerate(("Donor", "Class", "Located H", "Missing H: acceptor (D···A, symmetry code)")):
            ttk.Label(rows, text=txt, font=app.bold).grid(row=0, column=col, sticky="w", padx=4, pady=(0, 4),
                                                          columnspan=1 if col < 3 else 4)
        self.mbs = {}
        fixed = [d for d in app.donors if d["n_missing"] == 0]
        r = 1
        for d in self.donors:
            i = d["i"]
            cls = d["cls"] + (" (x=%.2f)" % self.bv.poh_x.get(i, 0.5) if self.bv.cls[i] == hc.CLASS_POH else "")
            ttk.Label(rows, text=d["label"], font=app.bold).grid(row=r, column=0, sticky="w", padx=4)
            ttk.Label(rows, text=cls).grid(row=r, column=1, sticky="w", padx=4)
            loc = ", ".join("%s→%s" % (h.located, st.atoms[h.cand.acceptor].label if h.cand else "–")
                            for h in d["located"] if h.primary)
            ttk.Label(rows, text=loc or "–", foreground="#1d4ed8").grid(row=r, column=2, sticky="w", padx=4)
            for k in range(d["n_missing"]):
                mb = tk.Menubutton(rows, text="(choose)", relief="raised", indicatoron=True, anchor="w", width=34,
                                   font=("Consolas", 9))
                menu = tk.Menu(mb, tearoff=0, font=("Consolas", 9))
                menu.configure(postcommand=lambda m=menu, d=d, k=k: self.fill_menu(m, d, k))
                mb.configure(menu=menu)
                mb.grid(row=r, column=3 + k, sticky="w", padx=3, pady=2)
                self.mbs[(i, k)] = mb
            r += 1
        if fixed:
            ttk.Label(rows, foreground="#555", text="All H located (fixed): " + ", ".join(
                d["label"] for d in fixed)).grid(row=r, column=0, columnspan=6, sticky="w", padx=4, pady=(6, 0))
        if not self.donors:
            ttk.Label(rows, text="No donor with missing H (no water / hydroxyl ... class, or all H located).").grid(
                row=r + 1, column=0, columnspan=6, sticky="w", padx=4)
        self.bvt = BVTable(pw, height=8)
        pw.add(self.bvt, weight=2)
        bb = ttk.Frame(win, padding=8)
        bb.pack(fill="x")
        if app.res is not None and app.res.schemes:
            ttk.Button(bb, text="Load selected scheme", command=self.load_selected).pack(side="left")
        ttk.Button(bb, text="Clear all", command=self.clear).pack(side="left", padx=4)
        accent_button(bb, "  Add to Refinement comparison  ", self.to_cmp, app.bold).pack(side="left", padx=(12, 4))
        ttk.Button(bb, text="Finalize ▶", command=self.to_finalize).pack(side="left")
        ttk.Button(bb, text="Export CIF...", command=self.export_cif).pack(side="left", padx=4)
        ttk.Button(bb, text="Close", command=win.destroy).pack(side="right")
        if app.theme.get() == "dark":
            app.apply_theme()
            win.configure(background="#1f2329")
        self.refresh()

    # ---- scheme from the current choices
    def hs_of(self, choice):
        hs = list(self.located)
        for d in self.donors:
            hs += hc.hs_for_choice(self.st, self.bv, d, choice[d["i"]], self.params, self.s)
        return hc.resolve_flips(self.st, self.bv, hs, self.s)

    def _key(self, choice):
        return tuple((i, tuple(id(c) if c is not None else 0 for c in v)) for i, v in sorted(choice.items()))

    def evaluate(self, choice):
        key = self._key(choice)
        if key not in self.cache:
            hs = self.hs_of(choice)
            self.cache[key] = (hs, hc.evaluate_hset(self.st, self.bv, hs, self.s, self.params, self.grg_cache))
        return self.cache[key]

    def refresh(self):
        st = self.st
        hs, ev = self.evaluate(self.choice)
        n_need = sum(d["n_missing"] for d in self.donors)
        n_set = sum(1 for v in self.choice.values() for c in v if c is not None)
        hf = self.bv.h_formula
        self.score_lbl.configure(text="Total score %.1f      (%d of %d missing H placed)" % (ev["total"], n_set, n_need))
        self.detail_var.set(
            "BV score F&I %.2f   GRG %.2f   |   angle pen. %.1f   |   H···H + cone pen. %.0f (red %d, moderate %d, "
            "< %.2f Å %d, cone %d)   |   max outside F&I %.3f, max |Δ| GRG %.3f   |   H/cell %.2f%s" % (
                ev["bvF"], ev["bvG"], ev["geom"], ev["hh"], ev["red"], ev["mod"], self.s.hh_severe, ev["sev"],
                ev["cone"], ev["maxF"], ev["maxG"], ev["h_cell"],
                "" if hf is None else " (formula %.2f)" % hf))
        labels = hc.label_hydrogens(st, hs)
        self.app._flag_box(self.flag, hc.h_contacts(st, hs, cutoff=self.s.hh_mod), labels, hs)
        self.bvt.fill(st, self.bv, hs, self.s, self.params, labels)
        for d in self.donors:
            for k, c in enumerate(self.choice[d["i"]]):
                self.mbs[(d["i"], k)].configure(text=self._text(c) if c is not None else "(choose)")

    def _text(self, c):
        return "%-6s %5.3f Å  %-8s" % (self.st.atoms[c.acceptor].label, c.nb.d,
                                       self.st.symop_code(c.nb.op, c.nb.shift))

    # ---- menus
    def fill_menu(self, menu, d, k):
        menu.delete(0, "end")
        st, i = self.st, d["i"]
        cur = self.choice[i]
        base = dict(self.choice)
        base[i] = list(cur)
        base[i][k] = None
        t0 = self.evaluate(base)[1]["total"]
        others = [c for j, c in enumerate(cur) if j != k and c is not None]
        entries = []
        for c in sorted(d["cands"], key=lambda c: (st.atoms[c.acceptor].label, c.nb.d)):
            if any(c is o for o in others):
                continue
            trial = dict(base)
            trial[i] = list(base[i])
            trial[i][k] = c
            prob = hc.choice_problem(st, self.bv, d, [x for x in trial[i] if x is not None], self.params, self.s,
                                     self.located)
            delta = self.evaluate(trial)[1]["total"] - t0
            entries.append((c, prob, delta))
        ok = [e for e in entries if not e[1]]
        hard = [e for e in entries if e[1]]
        menu.add_command(label="(none)" + ("  - current" if cur[k] is None else ""),
                         command=lambda: self.set(d, k, None))
        menu.add_separator()
        if not ok and hard:
            menu.add_command(label="All acceptors are hard fails - shown anyway:", state="disabled")
            show = entries
        elif self.show_hard.get():
            show = entries
        else:
            show = ok
        for c, prob, delta in show:
            col = self.GOOD if delta > 0.5 else (self.NEUTRAL if delta > -5 else
                                                  (self.MILD if delta > -200 else self.BAD))
            lab = "%s%s  %+9.1f%s" % ("✖ " if prob else "", self._text(c), delta,
                                     "   ← current" if cur[k] is c else "")
            menu.add_command(label=lab, foreground=col, activeforeground=col,
                             command=lambda c=c: self.set(d, k, c))
        if hard and ok and not self.show_hard.get():
            menu.add_separator()
            menu.add_command(label="(%d hard-fail acceptor(s) hidden - tick 'Show hard-fail acceptors')" % len(hard),
                             state="disabled")

    def set(self, d, k, c):
        self.choice[d["i"]][k] = c
        self.refresh()

    def clear(self):
        self.choice = {d["i"]: [None] * d["n_missing"] for d in self.donors}
        self.refresh()

    def load_selected(self):
        app = self.app
        r = app.selected_scheme()
        if r is None:
            r = 0
        sc = app.res.schemes[r]
        byi = {d["i"]: d for d in self.donors}
        for dd, kk in zip(app.res.donors, sc["idx"]):
            o = app.res.opts[app.res.donors.index(dd)][kk]
            if dd["i"] in byi:
                n = byi[dd["i"]]["n_missing"]
                cands = list(o.cands)[:n]
                self.choice[dd["i"]] = cands + [None] * (n - len(cands))
        self.refresh()

    def load_hs(self, hs, name=""):
        """Start from an existing H set (a scheme, refined or custom entry, the H in the CIF ...): every
        H-bond of a donor with missing H is matched to that donor's candidate of the same acceptor
        position. H are re-placed ideally (refined rotations are not kept); of a combined (disordered)
        scheme the component with the largest weight is used."""
        comps = hc.component_hsets(hs)
        if comps:
            compw = next(h.compw for h in hs if getattr(h, "compw", None))
            hs = comps[max(comps, key=lambda k: compw[k])]
        byi = {d["i"]: d for d in self.donors}
        new = {i: [] for i in byi}
        for h in hs:
            if h.cand is None or h.donor not in byi or not getattr(h, "primary", True):
                continue
            if h.located and any(h is x for x in self.located):
                continue                                   # a located H that stays fixed
            d = byi[h.donor]
            c = next((c for c in d["cands"] if tuple(c.nb.key) == tuple(h.cand.nb.key)), None)
            if c is not None and all(c is not x for x in new[h.donor]):
                new[h.donor].append(c)
        n_unmatched = 0
        for i, d in byi.items():
            lst = new[i][:d["n_missing"]]
            n_unmatched += max(0, len(new[i]) - d["n_missing"])
            self.choice[i] = lst + [None] * (d["n_missing"] - len(lst))
        self.source = name
        if name:
            self.win.title("%s - adjust scheme: %s (%s)" % (APP_NAME, name, self.st.name))
        self.refresh()
        return n_unmatched

    # ---- output
    def current(self):
        hs, ev = self.evaluate(self.choice)
        return hs, ev

    def to_cmp(self):
        hs, ev = self.current()
        self.n_made += 1
        st = self.st
        info = "; ".join("%s→%s" % (d["label"], ",".join(st.atoms[c.acceptor].label for c in self.choice[d["i"]]
                                                          if c is not None) or "-") for d in self.donors)
        name = ("Adjusted %s" % self.source) if getattr(self, "source", "") else "Custom scheme %d" % self.n_made
        if any(c["name"] == name for c in self.app.cmp):
            name += " (%d)" % self.n_made
        self.app._add_cmp(name, hs, ev, info, tag=("red",) if ev["red"] else (("mod",) if ev["mod"] else ()))
        self.app.ref_status.set("Added '%s' from the scheme builder." % name)
        self.app.nb.select(TAB_REFINE)

    def to_finalize(self):
        hs, _ev = self.current()
        self.app.load_hatoms(hs, "custom scheme")
        self.app.nb.select(TAB_FINAL)

    def export_cif(self):
        hs, _ev = self.current()
        if not any(not h.located for h in hs):
            messagebox.showinfo(APP_NAME, "No H placed yet.", parent=self.win)
            return
        try:
            path = self.app.write_cif(self.st, self.app.bv, hs)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror(APP_NAME, "Could not write the CIF:\n%s" % e, parent=self.win)
            return
        messagebox.showinfo(APP_NAME, "Wrote %s" % path, parent=self.win)


class HBondApp:
    def __init__(self, root):
        self.root = root
        self.structures = []
        self.st = self.bv = self.res = None
        self.donors, self.loc_h, self.located = [], [], ([], [])
        self.overrides = {}          # structure title -> {(site, bond key): bool}
        self.class_overrides = {}    # structure title -> {site index: class}
        self.valence_overrides = {}  # structure title -> {site index: {'v': n} | {'avg': q} | {'mix': ...}}
        self.role_overrides = {}     # structure title -> {site index: 'normal' | 'acceptor' | 'exclude'}
        self.role_reason = {}        # structure title -> {site index: why the role was set}
        self.params = None
        self._worker = None
        self._result = None
        self._progress = ""
        self.fin_h, self.fin_pass, self.fin_info, self.checked = {}, set(), {}, set()
        self.fin_rowkey, self.fin_by_key, self.fin_opts_of = {}, {}, {}
        self.fin_loc, self.fin_parent, self._fin_n = {}, {}, 0
        self.table_data = None
        self.loc_all = []
        self.cmp = []                # comparison entries: {"name", "hs", "ev", "info"}
        self._bg_thread = None
        self.subs = None             # subschemes found (see find_subschemes)
        self.sub_rows = []           # displayed rows: (group index, member index or None)
        self._sub_thread = None
        self._cache = {}             # structure title -> snapshot of its analysis state (switch back / save)

        root.title(APP_TITLE)
        ico = _resource("HydroGEN.ico")
        if ico:
            try:
                root.iconbitmap(default=ico)
            except tk.TclError:
                pass
        root.geometry("1250x900")
        root.minsize(950, 640)
        self.bold = tkfont.nametofont("TkDefaultFont").copy()
        self.bold.configure(weight="bold")
        self.italic = tkfont.nametofont("TkDefaultFont").copy()
        self.italic.configure(slant="italic")

        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)

        frow = ttk.Frame(outer)
        frow.pack(fill="x")
        ttk.Label(frow, text="Structure:").pack(side="left")
        self.struct_var = tk.StringVar()
        self.struct_box = ttk.Combobox(frow, textvariable=self.struct_var, state="disabled", width=60)
        self.struct_box.pack(side="left", padx=6, fill="x", expand=True)
        self.struct_box.bind("<<ComboboxSelected>>", lambda e: self.on_structure_change())
        ttk.Button(frow, text="Browse...", command=self.browse).pack(side="left")
        self.info_var = tk.StringVar(value="Select one or more CIF files to begin.")
        ttk.Label(outer, textvariable=self.info_var, foreground="#555").pack(fill="x", pady=(4, 0))

        # collapsible settings: click the bar to open / hide (state remembered)
        self.prefs = load_prefs()
        ttk.Style().configure("Expander.TButton", anchor="w", padding=(8, 3))
        self.toggle_btn = ttk.Button(outer, style="Expander.TButton", command=self.toggle_settings)
        self.toggle_btn.pack(fill="x", pady=(6, 0))
        self.settings_frame = ttk.Frame(outer, padding=6)
        self._build_settings(self.settings_frame)

        arow = ttk.Frame(outer)
        arow.pack(fill="x", pady=(6, 0))
        self.after_settings = arow
        self._settings_open = bool(self.prefs.get("settings_open", False))
        self._show_settings(self._settings_open)
        self._pending = []
        self.run_btn = accent_button(arow, "  Run analysis  ", self.run, self.bold)
        self.run_btn.configure(state="disabled")
        self.run_btn.pack(side="left")
        self.export_btn = ttk.Button(arow, text="Export CSV...", command=self.export, state="disabled")
        self.export_btn.pack(side="left", padx=6)
        ttk.Button(arow, text="Build scheme...", command=self.open_builder).pack(side="left")
        self.show_severe = tk.BooleanVar(value=False)
        self.ignore_loc = tk.BooleanVar(value=False)
        ttk.Checkbutton(arow, text="Show impossible configurations", variable=self.show_severe,
                        command=self.rerun_if_done).pack(side="left", padx=(12, 4))
        ttk.Checkbutton(arow, text="Ignore located H", variable=self.ignore_loc,
                        command=self.rerun_if_done).pack(side="left", padx=4)
        ttk.Label(arow, text="Disordered water/OH:").pack(side="left", padx=(12, 3))
        self.dis_mode = tk.StringVar(value="donors")
        cb = ttk.Combobox(arow, textvariable=self.dis_mode, values=list(DIS_MODES), state="readonly", width=14)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.rerun_if_done())
        self.hf_btn = ttk.Button(arow, text="Re-evaluate without hard-fail donors", command=self.drop_hard_fails,
                                 state="disabled")
        self.hf_btn.pack(side="left", padx=(8, 0))
        self.status_var = tk.StringVar()
        self.status_lbl = ttk.Label(outer, textvariable=self.status_var)
        self.status_lbl.pack(fill="x", pady=(3, 0))

        self.nb = ttk.Notebook(outer)
        self.nb.pack(fill="both", expand=True, pady=(6, 0))
        self._build_bv_tab()
        self._build_env_tab()
        self._build_disorder_tab()
        self._build_schemes_tab()
        self._build_refine_tab()
        self._build_subs_tab()
        self._build_located_tab()
        self._build_inferred_tab()
        self._build_details_tab()
        self._build_table_tab()
        self._build_finalize_tab()

        try:
            self.params = hc.BVParams()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Bond-valence parameters", str(e))

        self.project_path = None
        self._pending_restore = None
        self.theme = tk.StringVar(value=self.prefs.get("theme", "light"))
        self._build_menu()
        self.apply_theme()

    # ======================================================= menus and theme
    def _build_menu(self):
        mb = tk.Menu(self.root)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Open CIF(s)...", command=self.browse, accelerator="Ctrl+O")
        fm.add_separator()
        fm.add_command(label="Open analysis...", command=self.open_analysis, accelerator="Ctrl+Shift+O")
        fm.add_command(label="Save analysis", command=self.save_analysis, accelerator="Ctrl+S")
        fm.add_command(label="Save analysis as...", command=lambda: self.save_analysis(ask=True))
        fm.add_separator()
        fm.add_command(label="Load inputs...", command=self.load_inputs)
        fm.add_command(label="Save inputs...", command=self.save_inputs)
        fm.add_command(label="Restore default inputs", command=self.restore_defaults)
        fm.add_command(label="Clear manual overrides (classes, roles, bonds)", command=self.clear_overrides)
        fm.add_separator()
        fm.add_command(label="Export CSV...", command=self.export)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.root.destroy)
        mb.add_cascade(label="File", menu=fm)
        pm = tk.Menu(mb, tearoff=0)
        pm.add_radiobutton(label="Light mode", variable=self.theme, value="light", command=self.set_theme)
        pm.add_radiobutton(label="Dark mode", variable=self.theme, value="dark", command=self.set_theme)
        mb.add_cascade(label="Preferences", menu=pm)
        hm = tk.Menu(mb, tearoff=0)
        hm.add_command(label="Open user manual", command=self.open_manual, accelerator="F1")
        hm.add_separator()
        hm.add_command(label="About %s" % APP_NAME, command=lambda: messagebox.showinfo(
            "About %s" % APP_NAME, "%s\n\nBond-valence sums (Gagné & Hawthorne 2015; Brese & O'Keeffe 1991), "
            "H-atom inference and H-bond scheme scoring (Ferraris & Ivaldi; GRG), refinement and CIF / SHELXL "
            "export." % APP_TITLE))
        mb.add_cascade(label="Help", menu=hm)
        mb.add_command(label="Update", command=self.check_for_update)
        self.root.config(menu=mb)
        self.root.bind_all("<F1>", lambda e: self.open_manual())
        self.root.bind_all("<Control-o>", lambda e: self.browse())
        self.root.bind_all("<Control-O>", lambda e: self.open_analysis())
        self.root.bind_all("<Control-s>", lambda e: self.save_analysis())

    # ============================================================== updates
    def check_for_update(self):
        """Menu 'Update': ask GitHub for the latest release; if it is newer than APP_VERSION, download its
        HydroGEN.zip into the program folder, move the new files into place (the old ones are kept in
        HydroGEN_old) and close; the user then reopens HydroGEN to start the new version."""
        if getattr(self, "_upd_busy", False):
            return
        self._upd_busy = True
        self.status_var.set("Checking GitHub for a newer version...")
        box = {}

        def work():
            try:
                box["rel"] = latest_release()
            except Exception as e:  # noqa: BLE001
                box["err"] = str(e)

        th = threading.Thread(target=work, daemon=True)
        th.start()

        def poll():
            if th.is_alive():
                self.root.after(150, poll)
                return
            self._upd_busy = False
            if "err" in box:
                self.status_var.set("Update check failed.")
                messagebox.showerror(APP_NAME, "Could not reach GitHub to check for updates:\n%s" % box["err"])
                return
            self._offer_update(box["rel"])

        self.root.after(150, poll)

    def _offer_update(self, rel):
        tag, url = rel["tag"], rel["page"]
        if version_tuple(tag) <= version_tuple(APP_VERSION):
            self.status_var.set("HydroGEN %s is the latest version." % APP_VERSION)
            messagebox.showinfo(APP_NAME, "You have the latest version (%s).\nLatest release on GitHub: %s"
                                % (APP_VERSION, tag))
            return
        notes = (rel.get("notes") or "").strip()
        if len(notes) > 1200:
            notes = notes[:1200] + " ..."
        app_dir = app_folder()
        if app_dir is None or rel.get("asset") is None:
            why = ("You are running HydroGEN from the Python source - download the new files there."
                   if not getattr(sys, "frozen", False) else
                   "This copy of HydroGEN cannot update itself - download the release's %s, unzip it and use "
                   "the HydroGEN folder in it." % UPDATE_ZIP if app_dir is None else
                   "The release has no %s attached." % UPDATE_ZIP)
            if messagebox.askyesno(APP_NAME, "HydroGEN %s is available (you have %s).\n\n%s\n\n%s\n\nOpen the release "
                                   "page in your browser?" % (tag, APP_VERSION, notes or "(no release notes)", why)):
                webbrowser.open(url)
            return
        if not messagebox.askyesno(APP_NAME, "HydroGEN %s is available (you have %s).\n\n%s\n\nDownload it (%.1f MB) "
                                   "and install it? HydroGEN closes once it is installed; you then open it again." % (
                                       tag, APP_VERSION, notes or "(no release notes)",
                                       rel["asset"]["size"] / 1048576.0)):
            return
        self._download_update(rel, os.path.join(app_dir, "HydroGEN_update.zip"), app_dir)

    def _download_update(self, rel, new, app_dir):
        win = tk.Toplevel(self.root)
        win.title("Updating %s" % APP_NAME)
        win.transient(self.root)
        win.resizable(False, False)
        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        msg = tk.StringVar(value="Downloading HydroGEN %s..." % rel["tag"])
        ttk.Label(frm, textvariable=msg, width=60).pack(anchor="w")
        bar = ttk.Progressbar(frm, length=380, maximum=max(1, rel["asset"]["size"]))
        bar.pack(fill="x", pady=(8, 8))
        state = {"done": 0, "cancel": False}
        cancel = ttk.Button(frm, text="Cancel", command=lambda: state.update(cancel=True))
        cancel.pack(anchor="e")
        win.protocol("WM_DELETE_WINDOW", lambda: state.update(cancel=True))
        win.grab_set()

        def work():
            try:
                download_file(rel["asset"]["url"], new, rel["asset"]["size"], state)
            except Exception as e:  # noqa: BLE001
                state["err"] = str(e)
                try:
                    os.remove(new)
                except OSError:
                    pass

        th = threading.Thread(target=work, daemon=True)
        th.start()

        def poll():
            if th.is_alive():
                bar["value"] = state["done"]
                msg.set("Downloading HydroGEN %s...  %.1f / %.1f MB" % (
                    rel["tag"], state["done"] / 1048576.0, rel["asset"]["size"] / 1048576.0))
                self.root.after(100, poll)
                return
            win.grab_release()
            win.destroy()
            if "err" in state:
                self.status_var.set("Update not installed.")
                if not state["cancel"]:
                    messagebox.showerror(APP_NAME, "The download failed - nothing was changed:\n%s" % state["err"])
                return
            self._install_update(rel, new, app_dir)

        self.root.after(100, poll)

    def _install_update(self, rel, zpath, app_dir):
        """Save first (optional), move the new program files into place, then close: the running HydroGEN reads
        its code from HydroGEN.exe, so it cannot keep running once that file is the new version's."""
        def drop_zip():
            try:
                os.remove(zpath)
            except OSError:
                pass
        if self.st is not None:
            ans = messagebox.askyesnocancel(APP_NAME, "HydroGEN %s is downloaded. HydroGEN closes after installing "
                                                      "it.\n\nSave the current analysis first?" % rel["tag"])
            if ans is None:
                drop_zip()
                self.status_var.set("Update cancelled - nothing was changed.")
                return
            if ans:
                self.save_analysis()
        self.status_var.set("Installing HydroGEN %s..." % rel["tag"])
        win = tk.Toplevel(self.root)
        win.title("Updating %s" % APP_NAME)
        win.transient(self.root)
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", lambda: None)          # cannot be interrupted half-way
        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Installing HydroGEN %s - this takes a few seconds..." % rel["tag"], width=60).pack(anchor="w")
        bar = ttk.Progressbar(frm, length=380, mode="indeterminate")
        bar.pack(fill="x", pady=(8, 0))
        bar.start(15)
        win.grab_set()
        box = {}

        def work():
            try:
                box["old"] = install_update_zip(zpath, app_dir)
            except Exception as e:  # noqa: BLE001
                box["err"] = str(e)

        th = threading.Thread(target=work, daemon=True)
        th.start()

        def poll():
            if th.is_alive():
                self.root.after(150, poll)
                return
            bar.stop()
            win.grab_release()
            win.destroy()
            drop_zip()
            if "err" in box:
                self.status_var.set("Update not installed.")
                messagebox.showerror(APP_NAME, "The update could not be installed - nothing was changed:\n%s\n\n(Is "
                                               "another HydroGEN window open from this folder, or is the folder "
                                               "read-only?)" % box["err"])
                return
            messagebox.showinfo(APP_NAME, "HydroGEN %s is installed. HydroGEN will now close - open it again to start "
                                          "the new version.\n\nThe previous version is kept in the folder %s (you "
                                          "can delete it)." % (rel["tag"], os.path.basename(box["old"])))
            self.root.destroy()

        self.root.after(150, poll)

    def open_manual(self):
        path = _resource("HydroGEN_User_Guide.txt")
        if not path:
            messagebox.showerror(APP_NAME, "The user manual (HydroGEN_User_Guide.txt) was not found next to the "
                                           "program.")
            return
        try:
            os.startfile(path)                     # opens in the default .txt viewer (Notepad)
        except (OSError, AttributeError) as e:
            messagebox.showerror(APP_NAME, "Could not open the user manual:\n%s\n\n%s" % (path, e))

    def set_theme(self):
        self.prefs["theme"] = self.theme.get()
        save_prefs(self.prefs)
        self.apply_theme()

    def apply_theme(self):
        style = ttk.Style()
        dark = self.theme.get() == "dark"
        if dark:
            style.theme_use("clam")
            bg, fg, field, head, sel, edge = "#1f2329", "#e5e7eb", "#2a2f36", "#323843", "#2563eb", "#3a3f47"
            style.configure(".", background=bg, foreground=fg, fieldbackground=field, bordercolor=edge,
                            lightcolor=head, darkcolor=bg, troughcolor=field, insertcolor=fg,
                            selectbackground=sel, selectforeground="#ffffff")
            style.configure("Treeview", background=field, fieldbackground=field, foreground=fg)
            style.map("Treeview", background=[("selected", sel)], foreground=[("selected", "#ffffff")])
            style.configure("Treeview.Heading", background=head, foreground=fg, relief="flat")
            style.map("Treeview.Heading", background=[("active", edge)])
            style.configure("TNotebook", background=bg, bordercolor=edge)
            style.configure("TNotebook.Tab", background=head, foreground=fg, padding=(10, 3))
            style.map("TNotebook.Tab", background=[("selected", bg)], foreground=[("selected", "#ffffff")])
            style.configure("TButton", background=head, foreground=fg)
            style.map("TButton", background=[("active", edge), ("disabled", bg)],
                      foreground=[("disabled", "#6b7280")])
            style.configure("TEntry", fieldbackground=field, foreground=fg)
            style.configure("TCombobox", fieldbackground=field, background=head, foreground=fg, arrowcolor=fg)
            style.map("TCombobox", fieldbackground=[("readonly", field)], foreground=[("readonly", fg)])
            for w in ("TCheckbutton", "TRadiobutton", "TLabel", "TFrame", "TLabelframe", "TPanedwindow"):
                style.configure(w, background=bg, foreground=fg)
            # hover highlight: dark bluish grey (clam's default is near-white, which hid the white text)
            hover = "#3a4556"
            for w in ("TCheckbutton", "TRadiobutton"):
                style.configure(w, indicatorbackground=field, indicatorforeground=fg, upperbordercolor=edge,
                                lowerbordercolor=edge)
                style.map(w, background=[("active", hover)], foreground=[("disabled", "#6b7280"), ("active", fg)],
                          indicatorbackground=[("pressed", hover), ("disabled", bg), ("active", field)])
            style.map("TButton", background=[("pressed", "#2c3442"), ("active", hover), ("disabled", bg)],
                      foreground=[("disabled", "#6b7280"), ("active", fg)])
            style.map("TNotebook.Tab", background=[("selected", bg), ("active", hover)],
                      foreground=[("selected", "#ffffff"), ("active", fg)])
            style.map("Treeview.Heading", background=[("active", hover)], foreground=[("active", fg)])
            style.map("TCombobox", fieldbackground=[("readonly", field)], foreground=[("readonly", fg)],
                      background=[("active", hover), ("pressed", hover)], arrowcolor=[("disabled", "#6b7280")])
            style.configure("TScrollbar", background=head, troughcolor=field, arrowcolor=fg, bordercolor=edge)
            style.map("TScrollbar", background=[("active", hover), ("pressed", hover)])
            style.configure("TLabelframe.Label", background=bg, foreground=fg)
            self.root.option_add("*TCombobox*Listbox.background", field)
            self.root.option_add("*TCombobox*Listbox.foreground", fg)
            self.root.configure(background=bg)
            text_bg, text_fg, tag_fg = field, fg, "#111111"
        else:
            try:
                style.theme_use("vista")
            except tk.TclError:
                style.theme_use("default")
            self.root.option_add("*TCombobox*Listbox.background", "white")
            self.root.option_add("*TCombobox*Listbox.foreground", "black")
            self.root.configure(background="SystemButtonFace")
            text_bg, text_fg, tag_fg = "#f6f6f6", "#111111", ""
        for t in (self.bv_notes, self.table_text, self.det_text):
            t.configure(background=text_bg, foreground=text_fg, insertbackground=text_fg)
        self.fin_hc.configure(background=self.root.cget("background"))
        self.status_lbl.configure(foreground="#e5e7eb" if dark else "#1f2937")
        self._ok_fg, self._bad_fg = ("#4ade80", "#f87171") if dark else ("#166534", "#b91c1c")
        self._plain_fg = "#e5e7eb" if dark else "#1f2937"
        if self.st is not None:
            self.refresh_finalize()
        for lbl in (self.sch_flag, self.cmp_flag, self.fin_flag, self.sub_flag):
            if not lbl.cget("text"):
                lbl.configure(background=self.root.cget("background"))
        def alive(w):
            try:
                return bool(w.winfo_exists())
            except tk.TclError:
                return False
        ScrolledTree.instances = [t for t in ScrolledTree.instances if alive(t)]
        for st_ in ScrolledTree.instances:
            tree = st_.tree
            if not hasattr(st_, "_fg_light"):
                st_._fg_light = {}
            for tag in BG_TAGS:
                if str(tree.tag_configure(tag, "background") or ""):
                    tree.tag_configure(tag, foreground=tag_fg)
            for tag in ("off", "fail", "bad", "hrow", "manual", "located", "cif", "hbad"):
                if str(tree.tag_configure(tag, "background") or ""):
                    continue
                cur = str(tree.tag_configure(tag, "foreground") or "")
                if tag not in st_._fg_light:
                    st_._fg_light[tag] = cur
                light = st_._fg_light[tag]
                if light:
                    tree.tag_configure(tag, foreground=FG_DARK.get(light, light) if dark else light)

    # ================================================== inputs (save / load)
    def _input_vars(self):
        d = dict(self.vars)
        d.update({"rank_by": self.rank_var, "show_impossible": self.show_severe, "ignore_located": self.ignore_loc,
                  "disorder_mode": self.dis_mode, "refine_mode": self.ref_mode, "refine_n_worst": self.ref_n,
                  "refine_max_deg": self.ref_max, "refine_sigma_bv": self.ref_sbv, "refine_sigma_hh": self.ref_shh,
                  "refine_hdh_deg": self.ref_hdh, "refine_w_sym": self.ref_wsym, "refine_auto_pick": self.ref_auto,
                  "refine_dh": self.ref_dh, "refine_dh_range": self.ref_dhr, "refine_dh_weight": self.ref_dhw,
                  "grg_norm_located": self.grg_norm, "consolidate_schemes": self.consol_var,
                  "occ_weight": self.occ_w, "mix_balance": self.mix_bal, "nh4_gr": self.nh4_gr,
                  "angle_radius": self.angle_radius})
        return d

    def default_inputs(self):
        s = hc.Settings()
        d = {k: getattr(s, k) for k in self.vars if hasattr(s, k)}
        d.update({"loc_donors": "N O", "loc_acceptors": "N O F Cl S", "rank_by": "F&I only",
                  "show_impossible": False, "ignore_located": False, "disorder_mode": "donors",
                  "refine_mode": "All inferred H", "refine_n_worst": s.refine_n_worst,
                  "refine_max_deg": s.refine_max_deg, "refine_sigma_bv": s.refine_sigma_bv,
                  "refine_sigma_hh": s.refine_sigma_hh, "refine_hdh_deg": s.refine_hdh_deg,
                  "refine_w_sym": s.refine_w_sym, "refine_auto_pick": s.refine_auto_pick,
                  "refine_dh": s.refine_dh, "refine_dh_range": s.refine_dh_range,
                  "refine_dh_weight": s.refine_dh_weight, "grg_norm_located": s.grg_norm_located,
                  "consolidate_schemes": True, "occ_weight": s.occ_weight, "mix_balance": s.mix_balance, "nh4_gr": s.nh4_gr,
                  "angle_radius": "3.0"})
        return d

    def get_inputs(self):
        return {k: v.get() for k, v in self._input_vars().items()}

    def set_inputs(self, d):
        """Set inputs from a dict of strings/values; returns the unknown keys."""
        vars_ = self._input_vars()
        unknown = []
        for k, val in d.items():
            var = vars_.get(k)
            if var is None:
                unknown.append(k)
            elif isinstance(var, tk.BooleanVar):
                var.set(str(val).strip().lower() in ("1", "true", "yes", "on"))
            else:
                var.set(str(val))
        return unknown

    def restore_defaults(self):
        if not messagebox.askyesno(APP_NAME, "Restore every input (settings, checkboxes, selectors, refinement "
                                             "settings) to its default value?\n\nManual class / role / bond "
                                             "overrides are kept (File → Clear manual overrides removes them)."):
            return
        self.set_inputs(self.default_inputs())
        self.status_var.set("All inputs restored to their defaults - press 'Run analysis' to apply them.")

    def clear_overrides(self):
        if self.st is None and not self.structures:
            return
        if not messagebox.askyesno(APP_NAME, "Remove all hand-set classes, valences, H-bond roles and bond overrides "
                                             "for every loaded structure?"):
            return
        self.overrides, self.class_overrides, self.role_overrides, self.role_reason = {}, {}, {}, {}
        self.valence_overrides = {}
        if self.st is not None:
            self._mark_pending("all manual classes, valences, roles and bond overrides removed")

    def save_inputs(self):
        path = filedialog.asksaveasfilename(title="Save inputs", defaultextension=".txt",
                                            initialfile="HydroGEN_inputs.txt", filetypes=[("Text", "*.txt")])
        if not path:
            return
        lines = ["# HydroGEN inputs - saved %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "# One 'key = value' per line; lines starting with # are ignored. Load with File > Load inputs.",
                 "# Keys that are left out keep their current value when the file is loaded.", ""]
        defaults = self.default_inputs()
        for k, v in self.get_inputs().items():
            lab = self.var_labels.get(k, k.replace("_", " "))
            note = "" if str(defaults.get(k)) == str(v) else "   # changed (default %s)" % defaults.get(k)
            lines.append("# %s" % lab)
            lines.append("%s = %s%s" % (k, str(v).lower() if isinstance(v, bool) else v, note))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        self.status_var.set("Saved inputs to %s" % path)

    def load_inputs(self, path=None):
        path = path or filedialog.askopenfilename(title="Load inputs", filetypes=[("Text", "*.txt"),
                                                                                    ("All files", "*.*")])
        if not path:
            return
        d = {}
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip() if not line.lstrip().startswith("#") else ""
                if "=" in line:
                    k, v = line.split("=", 1)
                    d[k.strip()] = v.strip()
        unknown = self.set_inputs(d)
        msg = "Loaded %d input(s) from %s." % (len(d) - len(unknown), os.path.basename(path))
        if unknown:
            msg += " Unknown keys ignored: %s." % ", ".join(unknown)
        self.status_var.set(msg + " Press 'Run analysis' to apply them.")

    # ===================================================== analysis files
    @staticmethod
    def _hspec(h):
        sp = {"donor": h.donor, "vec": list(h.vec), "acc": list(h.cand.nb.key)}
        if getattr(h, "fixlen", False):
            sp["fixlen"] = True
            sp["grg"] = getattr(h, "grg_len", None)
        if getattr(h, "occf", 1.0) < 0.999 or getattr(h, "compw", None):     # split / partial / combined H
            sp["w"] = {"occf": h.occf, "p": h.p, "accf": getattr(h, "accf", 1.0), "cond": bool(h.conditional),
                       "alt": sorted(getattr(h, "alt", ()) or ()), "key": list(h.key)}
            if getattr(h, "compw", None):
                sp["w"].update({"comp": None if h.comp is None else sorted(h.comp),
                                "compw": {str(k): v for k, v in h.compw.items()}, "base": list(h.base)})
        return sp

    def _from_spec(self, sp):
        """Rebuild an inferred H from a saved spec (donor, vector, acceptor neighbour key)."""
        acc = tuple(sp["acc"])
        for d in self.donors:
            if d["i"] != sp["donor"]:
                continue
            for c in d["cands"]:
                if tuple(c.nb.key) == acc or list(c.nb.key) == sp["acc"]:
                    h = hc._make_h(self.st, d["i"], c, tuple(sp["vec"]), self.s, self.params,
                                   grg_len=sp.get("grg"))
                    h.fixlen = bool(sp.get("fixlen", False))
                    w = sp.get("w")
                    if w:
                        h.occf, h.p, h.accf, h.conditional = w["occf"], w["p"], w.get("accf", 1.0), w["cond"]
                        h.alt = frozenset(w.get("alt", []))
                        h.key = tuple(tuple(x) if isinstance(x, list) else x for x in w["key"])
                        if "compw" in w:
                            h.comp = None if w["comp"] is None else frozenset(w["comp"])
                            h.compw = {int(k): v for k, v in w["compw"].items()}
                            h.base = tuple(w["base"])
                    return h
        return None

    def save_analysis(self, ask=False):
        if not self.structures:
            messagebox.showinfo(APP_NAME, "Nothing to save - open a CIF first.")
            return
        path = self.project_path
        if ask or not path:
            base = os.path.splitext(os.path.basename(self.structures[0].source))[0]
            path = filedialog.asksaveasfilename(title="Save analysis", defaultextension=PROJECT_EXT,
                                                initialfile=base + PROJECT_EXT,
                                                initialdir=os.path.dirname(self.structures[0].source),
                                                filetypes=[("HydroGEN analysis", "*" + PROJECT_EXT)])
            if not path:
                return
        files, seen = [], set()
        for s_ in self.structures:
            if s_.source in seen:
                continue
            seen.add(s_.source)
            try:
                with open(s_.source, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                text = None
            files.append({"path": s_.source, "text": text})

        def tkey(dct):
            return {t: v for t, v in dct.items() if v}

        data = {"format": "HydroGEN analysis", "version": 1,
                "saved": datetime.datetime.now().isoformat(timespec="seconds"),
                "inputs": self.get_inputs(), "files": files, "current": self.struct_var.get(),
                "bond_overrides": {t: [[i, list(k), v] for (i, k), v in ov.items()]
                                   for t, ov in tkey(self.overrides).items()},
                "class_overrides": {t: {str(i): c for i, c in ov.items()} for t, ov in tkey(self.class_overrides).items()},
                "valence_overrides": {t: {str(i): v for i, v in ov.items()}
                                      for t, ov in tkey(self.valence_overrides).items()},
                "role_overrides": {t: {str(i): r for i, r in ov.items()} for t, ov in tkey(self.role_overrides).items()},
                "role_reason": {t: {str(i): r for i, r in ov.items()} for t, ov in tkey(self.role_reason).items()},
                "ran": self.st is not None,
                "optimized": [{"file": os.path.basename(x.source), "name": x.name, "note": getattr(x, "opt_note", ""),
                               "coords": {lab: list(new) for lab, (_old, new) in x.optimized.items()}}
                              for x in self.structures if getattr(x, "optimized", None)]}
        if self.st is not None:
            data["finalize"] = {"title": self.st.title,
                                "checked": [self._hspec(h) for _i, h in sorted(self._fin_effective().items())]}
            data["comparison"] = [{"name": e["name"], "info": e["info"], "cif": e["name"].startswith("H in CIF"),
                                   "hyp": [self._hspec(h) for h in e["hs"] if not h.located]} for e in self.cmp]
        # the complete state (every analysed structure, schemes, comparison, Finalize, subschemes), so that
        # opening the file shows it as it is instead of searching again
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self._snapshot_current()
            snaps = {}
            for t, sn in self._cache.items():
                if not any(x is sn["payload"][0] for x in self.structures):
                    continue
                sn = dict(sn)
                if sn.get("subs"):                       # subscheme analyses are re-created when needed
                    subs = dict(sn["subs"])
                    subs["groups"] = [{k: v for k, v in g.items() if k not in ("eval", "member_eval")}
                                      for g in subs["groups"]]
                    sn["subs"] = subs
                snaps[t] = sn
            extra = [x for x in self.structures if getattr(x, "sub_parent", None) is not None]
            data["state"] = dump_state({"version": STATE_VERSION, "snapshots": snaps, "extra": extra,
                                        "order": [x.title for x in self.structures]})
        except Exception as e:  # noqa: BLE001
            data.pop("state", None)
            messagebox.showwarning(APP_NAME, "The results could not be stored (they will be recalculated when "
                                             "the file is opened):\n%s" % e)
        finally:
            self.root.config(cursor="")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        self.project_path = path
        self.root.title("%s  [%s]" % (APP_TITLE, os.path.basename(path)))
        self.status_var.set("Saved analysis to %s" % path)

    def open_analysis(self, path=None):
        path = path or filedialog.askopenfilename(title="Open analysis",
                                                  filetypes=[("HydroGEN analysis", "*" + PROJECT_EXT),
                                                             ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("format") != "HydroGEN analysis":
                raise ValueError("not a HydroGEN analysis file")
        except (OSError, ValueError) as e:
            messagebox.showerror(APP_NAME, "Could not open %s:\n%s" % (path, e))
            return
        structures, notes = [], []
        for f in data.get("files", []):
            p = f["path"]
            if not os.path.exists(p):
                alt = os.path.join(os.path.dirname(path), os.path.basename(p))
                if not os.path.exists(alt) and f.get("text"):
                    with open(alt, "w", encoding="utf-8") as fh:
                        fh.write(f["text"])
                    notes.append("re-created %s from the analysis file" % os.path.basename(alt))
                p = alt
            try:
                structures.extend(hc.load_structures(p))
            except Exception as e:  # noqa: BLE001
                notes.append("%s: %s" % (os.path.basename(p), e))
        if not structures:
            messagebox.showerror(APP_NAME, "No structure could be loaded.\n" + "\n".join(notes))
            return
        for o in data.get("optimized", []):         # re-create optimised structures from the stored coordinates
            base = next((x for x in structures if os.path.basename(x.source) == o["file"] and x.name == o["name"]
                         and not getattr(x, "optimized", None)), None)
            if base is None:
                notes.append("optimized %s: original structure not found" % o["name"])
                continue
            idx = {a.label: k for k, a in enumerate(base.atoms)}
            new_f = {idx[lab]: tuple(v) for lab, v in o["coords"].items() if lab in idx}
            structures.append(hc.optimized_structure(base, new_f, o.get("note", "")))
        self.clear_all()
        self._cache = {}
        state = None
        if data.get("state") and data.get("ran"):
            try:
                state = load_state(data["state"])
                if state.get("version") != STATE_VERSION:
                    raise ValueError("other version")
                snaps = state.get("snapshots", {})
                by_title = {x.title: k for k, x in enumerate(structures)}
                for t, sn in snaps.items():            # the saved structure objects carry the results
                    st_ = sn["payload"][0]
                    if t in by_title:
                        structures[by_title[t]] = st_
                for x in state.get("extra", []):
                    if x.title not in by_title:
                        structures.append(x)
                        by_title[x.title] = len(structures) - 1
                order = state.get("order") or []
                structures.sort(key=lambda x: order.index(x.title) if x.title in order else len(order))
                self._cache = {t: sn for t, sn in snaps.items()}
            except Exception as e:  # noqa: BLE001
                state = None
                self._cache = {}
                notes.append("The saved results could not be restored (%s) - the analysis is run again." % e)
        self.structures = structures
        names = [s_.title for s_ in structures]
        self.struct_box.configure(values=names, state="readonly")
        cur = data.get("current")
        self.struct_var.set(cur if cur in names else names[0])
        self.set_inputs(data.get("inputs", {}))

        def ikeys(dct):
            return {t: {int(i): v for i, v in ov.items()} for t, ov in dct.items()}

        self.overrides = {t: {(i, tuple(k)): v for i, k, v in lst} for t, lst in data.get("bond_overrides", {}).items()}
        self.class_overrides = ikeys(data.get("class_overrides", {}))
        self.valence_overrides = ikeys(data.get("valence_overrides", {}))
        self.role_overrides = ikeys(data.get("role_overrides", {}))
        self.role_reason = ikeys(data.get("role_reason", {}))
        self.project_path = path
        self.root.title("%s  [%s]" % (APP_TITLE, os.path.basename(path)))
        self.on_structure_change()
        self.run_btn.configure(state="normal" if self.params else "disabled")
        if notes:
            messagebox.showwarning(APP_NAME, "\n".join(notes))
        if state is not None and self.st is not None and self.st.title == self.struct_var.get():
            self.status_var.set("Opened %s - results as saved (%s). Run analysis to recalculate." % (
                os.path.basename(path), data.get("saved", "")))
            return
        if data.get("ran"):
            self._pending_restore = data
            self.run()
        else:
            self.status_var.set("Opened %s - press 'Run analysis'." % os.path.basename(path))

    def _restore_after_run(self, data):
        """Re-apply the saved Finalize selection and comparison entries after the analysis has run."""
        missing = 0
        fin = data.get("finalize") or {}
        if fin.get("title") == self.st.title and fin.get("checked") is not None:
            hs = [self._from_spec(sp) for sp in fin["checked"]]
            missing += sum(1 for h in hs if h is None)
            self.load_hatoms([h for h in hs if h is not None], "saved selection")
        for e in data.get("comparison", []) if fin.get("title") == self.st.title else []:
            if e.get("cif"):
                if self.loc_all:
                    ev = hc.evaluate_hset(self.st, self.bv, self.loc_all, self.s, self.params)
                    self._add_cmp(e["name"], list(self.loc_all), ev, e.get("info", ""), tag=("cif",))
                continue
            hyp = [self._from_spec(sp) for sp in e.get("hyp", [])]
            missing += sum(1 for h in hyp if h is None)
            hs = hc.resolve_flips(self.st, self.bv, list(self.loc_h) + [h for h in hyp if h is not None], self.s)
            ev = hc.evaluate_hset(self.st, self.bv, hs, self.s, self.params)
            self._add_cmp(e["name"], hs, ev, e.get("info", ""))
        msg = "Opened %s." % os.path.basename(self.project_path or "")
        if missing:
            msg += " %d saved H atom(s) could not be matched to the current candidates (inputs changed?)." % missing
        self.status_var.set(msg + "  " + self.status_var.get())

    # ================================================================ settings
    def _build_settings(self, sf):
        d = hc.Settings()
        self.vars = {}
        rows = [
            ("Bond valence", [("Sum s ≥", "sum_cutoff"), ("List s ≥", "list_cutoff"),
                              ("Env. radius Å", "env_radius"), ("Ignore non-H <Å", "min_bond_length"),
                              ("Water if w/o H <", "water_max"), ("OH if <", "hydroxyl_max")]),
            ("Candidates", [("D···A min", "da_min"), ("D···A max", "da_max"), ("Cone vertex °", "cone_angle"),
                            ("Pass if A–D–C >°", "adc_pass_angle"), ("D–H (GRG) Å", "dh_length"),
                            ("D–H (placement) Å", "geom_dh_length"), ("H–D–H °", "hdh_angle")]),
            ("Angle scoring", [("A–D–C soft °", "adc_soft_angle"), ("pen./°", "adc_soft_w"),
                               ("A–D–C hard °", "adc_hard_angle"), ("pen./°²", "adc_hard_w"),
                               ("Acc. cone ×", "acc_cone_factor"), ("A–D–C min occ.", "adc_min_occ")]),
            ("A1–D–A2", [("no pen. ≥ °", "pair_min_angle"), ("hard limit °", "pair_hard_angle"),
                         ("no pen. ≤ °", "pair_max_angle"), ("hard max °", "pair_max_hard"),
                         ("pen. at limit", "pair_soft_penalty"), ("pen. beyond", "pair_angle_penalty")]),
            ("BV scoring", [("BV scale vu", "bv_scale"), ("BV power", "bv_power"),
                            ("Donor band ±", "donor_band"), ("Group band ±", "group_band"), ("minor zone", "donor_minor"),
                            ("minor pen.", "minor_penalty"), ("BV weight", "w_bv"), ("Start score", "score_base")]),
            ("H···H", [("red <Å", "hh_red"), ("red pen.", "hh_red_penalty"),
                       ("slide from <Å", "hh_mod"), ("pen. at red", "hh_mod_penalty"),
                       ("impossible <Å", "hh_severe"), ("H flex Å", "h_flex")]),
            ("GRG & search", [("GRG D–H ±Å", "dh_range"), ("step Å", "dh_step"),
                              ("Options/donor", "max_options"), ("Exhaustive ≤", "exhaustive_limit"),
                              ("Beam width", "beam_width"), ("Keep", "keep_schemes")]),
            ("F&I", [("d0", "fi_d0"), ("n", "fi_n"), ("c", "fi_c")]),
            ("Located H", [("H···A ≤", "loc_max_ha"), ("D···A ≤", "loc_max_da"), ("D–H···A ≥", "loc_min_angle"),
                           ("D–H ≤", "loc_max_dh")]),
            ("Disorder/limits", [("Split <Å", "split_dist"), ("PART <Å", "group_dist"),
                                 ("H···cat. min Å", "h_min_cation"), ("H···an. min Å", "h_min_anion"),
                                 ("Flip pen.", "flip_penalty"), ("Dis. conflict pen.", "dis_conflict_penalty")]),
        ]
        self.var_labels = {}
        for r, (title, fields) in enumerate(rows):
            ttk.Label(sf, text=title, font=self.bold).grid(row=r, column=0, sticky="w")
            for k, (lab, key) in enumerate(fields):
                self.var_labels[key] = "%s: %s" % (title, lab)
                ttk.Label(sf, text=lab).grid(row=r, column=1 + 2 * k, sticky="e", padx=(8, 3), pady=1)
                var = tk.StringVar(value=str(getattr(d, key)))
                w = 9 if key in ("exhaustive_limit", "pair_angle_penalty", "hh_red_penalty") else 6
                ttk.Entry(sf, textvariable=var, width=w).grid(row=r, column=2 + 2 * k, sticky="w")
                self.vars[key] = var
        row_of = {title: r for r, (title, _f) in enumerate(rows)}
        self.grg_norm = tk.BooleanVar(value=d.grg_norm_located)
        ttk.Checkbutton(sf, text="GRG: located H at D–H (GRG)", variable=self.grg_norm).grid(
            row=row_of["GRG & search"], column=13, columnspan=4, sticky="w", padx=(10, 0))
        self.occ_w = tk.BooleanVar(value=d.occ_weight)
        ttk.Checkbutton(sf, text="Weight by occupancy", variable=self.occ_w).grid(
            row=row_of["BV scoring"], column=15, columnspan=3, sticky="w", padx=(10, 0))
        self.nh4_gr = tk.BooleanVar(value=d.nh4_gr)
        ttk.Checkbutton(sf, text="NH4+ bonds: García-Rodríguez 2000", variable=self.nh4_gr).grid(
            row=row_of["Bond valence"], column=13, columnspan=4, sticky="w", padx=(10, 0))
        ttk.Label(sf, text="Rank by").grid(row=row_of["F&I"], column=7, sticky="e", padx=(8, 3))
        self.rank_var = tk.StringVar(value="F&I only")
        ttk.Combobox(sf, textvariable=self.rank_var, values=list(RANK_CHOICES), state="readonly",
                     width=16).grid(row=row_of["F&I"], column=8, columnspan=3, sticky="w")
        for k, (lab, key, val) in enumerate((("Donors (tab)", "loc_donors", "N O"),
                                             ("Acceptors (tab)", "loc_acceptors", "N O F Cl S"))):
            self.var_labels[key] = "Located H: " + lab
            ttk.Label(sf, text=lab).grid(row=row_of["Located H"], column=9 + 2 * k, sticky="e", padx=(8, 3))
            var = tk.StringVar(value=val)
            ttk.Entry(sf, textvariable=var, width=10).grid(row=row_of["Located H"], column=10 + 2 * k, sticky="w")
            self.vars[key] = var

    def _show_settings(self, show):
        self._settings_open = show
        if show:
            self.settings_frame.pack(fill="x", before=self.after_settings)
            self.toggle_btn.configure(text="▼  Settings   (click to hide)")
        else:
            self.settings_frame.pack_forget()
            self.toggle_btn.configure(text="▶  Settings   (click to show all inputs)")

    def toggle_settings(self):
        self._show_settings(not self._settings_open)
        self.prefs["settings_open"] = self._settings_open
        save_prefs(self.prefs)

    def settings(self):
        s = hc.Settings()
        ints = {"max_options", "exhaustive_limit", "beam_width", "keep_schemes"}
        for key, var in self.vars.items():
            if key in ("loc_donors", "loc_acceptors"):
                continue
            try:
                setattr(s, key, int(float(var.get())) if key in ints else float(var.get()))
            except ValueError:
                raise ValueError("Setting '%s' must be a number." % key)
        s.rank_by = RANK_CHOICES.get(self.rank_var.get(), "fi")
        s.show_severe = bool(self.show_severe.get())
        s.ignore_located = bool(self.ignore_loc.get())
        s.grg_norm_located = bool(self.grg_norm.get())
        s.occ_weight = bool(self.occ_w.get())
        s.mix_balance = bool(self.mix_bal.get())
        s.nh4_gr = bool(self.nh4_gr.get())
        s.disorder_mode = DIS_MODES.get(self.dis_mode.get(), "donor")
        for key, var in (("refine_max_deg", self.ref_max), ("refine_n_worst", self.ref_n),
                         ("refine_sigma_bv", self.ref_sbv), ("refine_sigma_hh", self.ref_shh),
                         ("refine_hdh_deg", self.ref_hdh), ("refine_w_sym", self.ref_wsym),
                         ("refine_dh_range", self.ref_dhr), ("refine_dh_weight", self.ref_dhw)):
            try:
                setattr(s, key, int(float(var.get())) if key == "refine_n_worst" else float(var.get()))
            except ValueError:
                raise ValueError("Refinement setting '%s' must be a number." % key)
        s.refine_auto_pick = bool(self.ref_auto.get())
        s.refine_dh = bool(self.ref_dh.get())
        if s.refine_dh_range < 0 or s.refine_dh_weight < 0:
            raise ValueError("The D–H refinement range and weight must not be negative.")
        if s.list_cutoff > s.sum_cutoff:
            raise ValueError("The list cutoff must not be larger than the sum cutoff.")
        if s.env_radius < s.da_max:
            raise ValueError("The environment radius must be at least the D···A maximum.")
        if s.bv_scale <= 0 or s.dh_length <= 0 or s.geom_dh_length <= 0 or s.dh_step <= 0 or s.dh_range < 0:
            raise ValueError("BV scale, D–H and the GRG step must be positive; the GRG range must not be negative.")
        return s

    def located_args(self, s):
        return dict(max_ha=s.loc_max_ha, max_da=s.loc_max_da, min_angle=s.loc_min_angle, max_dh=s.loc_max_dh,
                    donors=re.split(r"[\s,;]+", self.vars["loc_donors"].get().strip()),
                    acceptors=re.split(r"[\s,;]+", self.vars["loc_acceptors"].get().strip()))

    # ==================================================================== tabs
    def _build_bv_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="Bond valence")
        self.formula_lbl = tk.Label(tab, text="", anchor="w", justify="left", padx=6, pady=3)
        self.formula_lbl.pack(fill="x", pady=(0, 4))
        bvbar = ttk.Frame(tab)
        bvbar.pack(fill="x", pady=(0, 4))
        # live charge balance of the model (cations + H from the classes + anions, per formula unit)
        self.charge_lbl = tk.Label(bvbar, text="Formula charge: -", padx=8, pady=2, relief="groove",
                                   font=("Segoe UI", 10, "bold"))
        self.charge_lbl.pack(side="left")
        self.mix_bal = tk.BooleanVar(value=hc.Settings().mix_balance)
        ttk.Checkbutton(bvbar, text="Keep charge balance (mixed valence, partial OH)", variable=self.mix_bal,
                        command=lambda: self._mark_pending("charge balance (mixed valence, partial OH) %s"
                                                           % ("on" if self.mix_bal.get() else "off"))
                        if self.bv is not None else None).pack(side="left", padx=(6, 14))
        ttk.Button(bvbar, text="Optimize structure...", command=self.optimize_dialog).pack(side="left")
        ttk.Label(bvbar, foreground="#555", text="  small bond-valence-driven shifts of the anion positions before the "
                  "H-bond analysis (optional)").pack(side="left")
        pw = ttk.PanedWindow(tab, orient="vertical")
        pw.pack(fill="both", expand=True)
        self.sites = ScrolledTree(pw, [
            ("site", "Site", 70, "w"), ("el", "El", 35, "w"), ("role", "Role", 55, "w"),
            ("val", "Valence", 75, "center"), ("occ", "Occ.", 60, "e"),
            ("bvs", "BVS (vu)", 70, "e"), ("noh", "BVS w/o H", 75, "e"), ("dev", "BVS − |V|", 75, "e"),
            ("n", "# summed", 65, "e"), ("weak", "Listed, not summed", 120, "e"), ("cls", "D/A class", 125, "w"),
            ("how", "Oxidation state from", 330, "w")], height=10, selectmode="extended")
        pw.add(self.sites, weight=3)
        self.sites.tree.bind("<<TreeviewSelect>>", lambda e: self.show_bonds())
        self.sites.tree.bind("<Button-3>", self.class_menu)
        self.sites.tree.tag_configure("manual", foreground="#b45309")
        for tag, colour in (("water", "#dbeafe"), ("hydroxyl", "#dcfce7"), ("probable", "#fef9c3"),
                            ("possible", "#ffedd5")):
            self.sites.tree.tag_configure(tag, background=colour)
        bf = ttk.Frame(pw)
        pw.add(bf, weight=3)
        self.bond_title = tk.StringVar(value="Select a site to see its bonds. Double-click a bond to include/exclude it.")
        ttk.Label(bf, textvariable=self.bond_title).pack(fill="x")
        self.bonds = ScrolledTree(bf, [
            ("nb", "Neighbour", 75, "w"), ("el", "El", 35, "w"), ("d", "d (Å)", 65, "e"),
            ("s", "s (vu)", 65, "e"), ("occ", "Occ. of nbr", 75, "e"), ("w", "s × occ", 65, "e"),
            ("status", "Status", 190, "w"), ("symop", "Symmetry op.", 160, "w"), ("code", "Code", 70, "w"),
            ("param", "Parameters", 260, "w")], height=8)
        self.bonds.pack(fill="both", expand=True)
        self.bonds.tree.bind("<Double-1>", self.toggle_bond)
        self.bonds.tree.tag_configure("off", foreground="#888")
        self.bonds.tree.tag_configure("manual", foreground="#b45309")
        nf = ttk.Frame(pw)
        pw.add(nf, weight=1)
        self.bv_notes = tk.Text(nf, height=5, wrap="word", relief="flat", background="#f6f6f6")
        self.bv_notes.pack(fill="both", expand=True)
        self.bv_notes.configure(state="disabled")

    def _build_disorder_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="Disorder")
        ttk.Label(tab, foreground="#555", wraplength=1180, justify="left", text=(
            "Exclusion groups = positions (symmetry copies included) that cannot be occupied at the same time: "
            "occupancy of one means the others are vacant. Pairs are linked when both are partly occupied, closer than "
            "'Split <' and their occupancies sum to ≤ 1, or when they are in different CIF disorder groups (PART) of "
            "one assembly. H···H contacts between such positions are ignored, and they are never donor and acceptor of "
            "each other. H-bond roles: 'acceptor only' sites donate nothing and their own BV sum is not scored; "
            "'excluded' sites are left out of the H-bond search completely. Set roles with the 'Disordered water/OH' "
            "selector, with right-click on the Bond valence tab, or below.")).pack(fill="x")
        pw = ttk.PanedWindow(tab, orient="vertical")
        pw.pack(fill="both", expand=True, pady=(4, 0))
        self.grp = ScrolledTree(pw, [
            ("site", "Site", 80, "w"), ("code", "Symmetry", 90, "w"), ("occ", "Occupancy", 80, "e"),
            ("cls", "Class", 140, "w"), ("role", "H-bond role", 170, "w")],
            tree_heading="Exclusion group / position", tree_width=560, height=8)
        pw.add(self.grp, weight=1)
        self.grp.tree.tag_configure("group", background="#eef2ff")
        self.grp.tree.tag_configure("bad", foreground="#b91c1c")
        rf = ttk.Frame(pw)
        pw.add(rf, weight=1)
        bar = ttk.Frame(rf)
        bar.pack(fill="x")
        ttk.Label(bar, text="Sites with a special H-bond role, and hard fails", font=self.bold).pack(side="left")
        ttk.Button(bar, text="Set selected: normal", command=lambda: self.set_role_sel("normal")).pack(side="right")
        ttk.Button(bar, text="Set selected: excluded", command=lambda: self.set_role_sel("exclude")).pack(
            side="right", padx=4)
        ttk.Button(bar, text="Set selected: acceptor only", command=lambda: self.set_role_sel("acceptor")).pack(
            side="right")
        ttk.Button(bar, text="Reset all roles", command=self.reset_roles).pack(side="right", padx=(0, 12))
        ttk.Button(bar, text="Lower symmetry...", command=self.lower_symmetry_dialog).pack(side="right",
                                                                                         padx=(0, 12))
        self.roles = ScrolledTree(rf, [
            ("site", "Site", 80, "w"), ("occ", "Occupancy", 80, "e"), ("cls", "Class", 140, "w"),
            ("dis", "Disordered", 80, "center"), ("role", "H-bond role", 170, "w"), ("why", "Why", 560, "w")],
            height=7, selectmode="extended")
        self.roles.pack(fill="both", expand=True)
        self.roles.tree.tag_configure("fail", background="#fecaca")
        self.dis_note = tk.StringVar()
        ttk.Label(tab, textvariable=self.dis_note, foreground="#a05000", wraplength=1180,
                  justify="left").pack(fill="x")

    def populate_disorder(self):
        st, bv, res = self.st, self.bv, self.res
        self.grp.clear()
        self.roles.clear()
        for k, g in enumerate(bv.groups, 1):
            bad = g["occ_sum"] > 1.05
            pid = self.grp.tree.insert("", "end", open=True, text="Group %d - %s" % (k, hc.group_text(st, g)),
                                       tags=("group",) + (("bad",) if bad else ()))
            for j, _vec, nb in g["members"]:
                code = "." if nb is None else st.symop_code(nb.op, nb.shift)
                self.grp.tree.insert(pid, "end", text="", values=(
                    st.atoms[j].label, code, _fmt(st.atoms[j].occ), bv.cls[j] or bv.roles[j],
                    ROLE_TEXT[bv.hrole.get(j, "normal")]))
        if not bv.groups:
            self.grp.tree.insert("", "end", text="No mutually exclusive positions found.")
        rov = self.role_overrides.get(st.title, {})
        why = self.role_reason.get(st.title, {})
        sites = set(bv.hrole) | set(res.hard_fail) | {i for i in bv.disordered if bv.cls[i] in hc.DONOR_CLASSES}
        for i in sorted(sites):
            r = bv.hrole.get(i, "normal")
            if i in res.hard_fail:
                reason = "HARD FAIL: " + res.hard_fail[i]
            elif i in rov:
                reason = why.get(i, "set by hand")
            elif r != "normal":
                reason = "'Disordered water/OH' = %s" % self.dis_mode.get()
            else:
                reason = "disordered; treated as a donor"
            self.roles.tree.insert("", "end", iid=str(i), tags=("fail",) if i in res.hard_fail else (), values=(
                st.atoms[i].label, _fmt(st.atoms[i].occ), bv.cls[i], "yes" if i in bv.disordered else "",
                ROLE_TEXT[r], reason))
        notes = []
        if res.hard_fail:
            notes.append("Hard fails: %s. 'Re-evaluate without hard-fail donors' makes them acceptor-only and "
                         "runs again." % ", ".join(st.atoms[i].label for i in res.hard_fail))
        if bv.h_excluded > 0.005:
            notes.append("%.2f H per cell belong to acceptor-only / excluded sites and are not placed." % bv.h_excluded)
        self.dis_note.set("  ".join(notes))
        self.hf_btn.configure(state="normal" if res.hard_fail else "disabled")

    def _set_roles(self, sites, role, reason, now=False):
        rov = self.role_overrides.setdefault(self.st.title, {})
        why = self.role_reason.setdefault(self.st.title, {})
        for i in sites:
            rov[i] = role
            why[i] = reason
        if now:
            self.run(keep_selection=True)
        else:
            for i in sites:
                self._show_pending_site(i, role=role)
            self._mark_pending("H-bond role of %s → %s" % (", ".join(self.st.atoms[i].label for i in sites),
                                                           ROLE_TEXT.get(role, role)))

    def set_role_sel(self, role):
        sel = [int(x) for x in self.roles.tree.selection()]
        if sel:
            self._set_roles(sel, role, "set by hand")

    def reset_roles(self):
        if self.st is not None:
            self.role_overrides[self.st.title] = {}
            self.role_reason[self.st.title] = {}
            self._mark_pending("all H-bond roles reset")

    # ---- manual changes wait for 'Run analysis' (several can be made, then one recalculation)
    def _mark_pending(self, what):
        self._pending.append(what)
        n = len(self._pending)
        self.run_btn.configure(text="  Run analysis  (%d change%s pending)  " % (n, "" if n == 1 else "s"))
        self.status_var.set("Pending (press Run analysis to apply): " + "; ".join(self._pending[-6:]) +
                            (" ..." if n > 6 else ""))

    def _clear_pending(self):
        self._pending = []
        self.run_btn.configure(text="  Run analysis  ")

    def _show_pending_site(self, i, cls=None, role=None):
        """Show a not-yet-applied class / role change in the Bond valence table."""
        iid = str(i)
        if not self.sites.tree.exists(iid):
            return
        vals = list(self.sites.tree.item(iid)["values"])
        cur = str(vals[10]).split("  →")[0]
        new = cls if cls is not None else (ROLE_TEXT.get(role, role) if role else "")
        vals[10] = "%s  → %s (pending)" % (cur, new)
        self.sites.tree.item(iid, values=vals, tags=tuple(self.sites.tree.item(iid)["tags"]) + ("manual",))

    def drop_hard_fails(self):
        if self.res is None or not self.res.hard_fail:
            return
        self._set_roles(list(self.res.hard_fail), "acceptor", now=True, reason=
                        "hard fail in an earlier run → acceptor only (re-evaluation)")

    def _build_inferred_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="Inferred H-bonds")
        ttk.Label(tab, foreground="#555", wraplength=1180, justify="left", text=(
            "Donors = O classed water/hydroxyl by the BV sum without H, or O carrying located H. Located H are "
            "kept as they are (their H-bond = shortest qualifying H···A); only missing H are inferred. Candidates: "
            "anions within the D···A shell, numbered by distance; pass if every A–D–C angle is above the pass "
            "angle and outside every forbidden cone.")).pack(fill="x")
        pw = ttk.PanedWindow(tab, orient="vertical")
        pw.pack(fill="both", expand=True)
        self.cands = ScrolledTree(pw, [
            ("acc", "Acceptor", 75, "w"), ("da", "D···A (Å)", 70, "e"), ("minadc", "min A–D–C (°)", 95, "e"),
            ("angles", "A–D–C per cation (°)", 220, "w"), ("cone", "Cone", 50, "center"),
            ("adc", "A–D–C", 55, "center"), ("adcpen", "A–D–C pen.", 75, "e"),
            ("sg", "s GRG*", 65, "e"), ("sf", "s F&I", 65, "e"), ("result", "Result", 150, "w"),
            ("symop", "Symmetry op.", 150, "w"), ("code", "Code", 70, "w")],
            tree_heading="Donor / candidate", tree_width=330, height=9)
        pw.add(self.cands, weight=1)
        self.cands.tree.tag_configure("fail", foreground="#999")
        self.cands.tree.tag_configure("donor", background="#eef2ff")
        self.cands.tree.tag_configure("located", font=self.italic, foreground="#1d4ed8")
        self.opts = ScrolledTree(pw, [
            ("hb", "H-bonds", 190, "w"), ("ang", "A1–D–A2 (°)", 85, "e"), ("hdh", "H–D–H (°)", 70, "e"),
            ("ha", "H···A (Å)", 110, "w"), ("dha", "D–H···A (°)", 110, "w"),
            ("apen", "A1–D–A2 pen.", 90, "e"), ("cpen", "A–D–C pen.", 75, "e"),
            ("df", "F&I: w/o H − donated", 135, "e"), ("status", "Status", 140, "w")],
            tree_heading="Donor / option", tree_width=330, height=9)
        pw.add(self.opts, weight=1)
        self.opts.tree.tag_configure("donor", background="#eef2ff")
        self.opts.tree.tag_configure("bad", foreground="#999")
        self.inf_note = tk.StringVar()
        ttk.Label(tab, textvariable=self.inf_note, foreground="#a05000", wraplength=1180,
                  justify="left").pack(fill="x")

    def _build_schemes_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="Schemes")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        # buttons first (right), so the explanatory text only takes the width that is left
        btns = ttk.Frame(top)
        btns.pack(side="right", anchor="n")
        b1 = ttk.Frame(btns)
        b1.pack(anchor="e")
        ttk.Button(b1, text="Compare", command=self.compare_schemes).pack(side="left", padx=(0, 4))
        ttk.Button(b1, text="Refine ▶", command=self.goto_refine).pack(side="left")
        ttk.Button(b1, text="BV table ▶", command=lambda: self.nb.select(TAB_TABLE)).pack(side="left", padx=4)
        ttk.Button(b1, text="Finalize ▶", command=self.send_scheme).pack(side="left")
        b2 = ttk.Frame(btns)
        b2.pack(anchor="e", pady=(3, 0))
        self.consol_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(b2, text="Consolidate equivalent schemes after each run",
                        variable=self.consol_var).pack(side="left")
        ttk.Button(b2, text="Consolidate now", command=self.consolidate_now).pack(side="left", padx=(6, 0))
        self.sch_info = tk.StringVar(value="Run the analysis to rank H-bond schemes.")
        info = ttk.Label(top, textvariable=self.sch_info, foreground="#555", wraplength=600, justify="left")
        info.pack(side="left", fill="x", expand=True, padx=(0, 10))
        info.bind("<Configure>", lambda e: info.configure(wraplength=max(250, e.width - 4)))
        pw = ttk.PanedWindow(tab, orient="vertical")
        pw.pack(fill="both", expand=True, pady=(4, 0))
        self.sch = ScrolledTree(pw, [
            ("rank", "Rank", 62, "e"), ("same", "≡F&I", 45, "center"), ("total", "Total score", 90, "e"),
            ("bvf", "BV score F&I", 95, "e"),
            ("bvg", "BV score GRG", 95, "e"), ("geom", "Angle pen.", 80, "e"), ("hh", "H···H pen.", 80, "e"),
            ("red", "Red", 40, "e"), ("mod", "Mod.", 45, "e"), ("sev", "<1.75", 45, "e"),
            ("cone", "Cone", 45, "e"), ("dis", "Dis.", 40, "e"),
            ("maxf", "max outside F&I", 105, "e"), ("maxg", "max |Δ| GRG", 85, "e"), ("hcell", "H/cell", 60, "e"),
            ("summary", "Inferred H-bonds (donor→acceptors)", 520, "w")], height=10, selectmode="extended")
        pw.add(self.sch, weight=2)
        self.sch.tree.tag_configure("hbad", foreground="#b91c1c")
        self.sch.tree.tag_configure("red", background="#fecaca")
        self.sch.tree.tag_configure("mod", background="#fed7aa")
        self.sch.tree.bind("<<TreeviewSelect>>", lambda e: self.show_scheme())
        self.sch.tree.bind("<Double-1>", lambda e: self.send_scheme())
        det = ttk.Frame(pw)
        pw.add(det, weight=3)
        self.sch_flag = tk.Label(det, text="", anchor="w", justify="left", padx=6, pady=3, wraplength=1180)
        self.sch_flag.pack(fill="x")
        self.sch_bv = BVTable(det, height=9)
        self.sch_bv.pack(fill="both", expand=True)

    def _build_refine_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="Refinement")
        ttk.Label(tab, foreground="#555", wraplength=1180, justify="left", text=(
            "Rotates the inferred H of each chosen water/hydroxyl rigidly about its O (at most the set angle; a water "
            "with one located H turns about that O–H) to minimise Σ((GRG Σ − ideal)/σ_bv)² + Σ((H···H limit − d)/σ_hh)² "
            "(bounded pattern search; optionally each inferred D–H length too, restrained). F&I sums depend only on "
            "D···A distances and do not change. Located H are never "
            "moved. Entries below can be compared, shown as a BV table, or sent to Finalize for CIF export.")
        ).pack(fill="x")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=(4, 0))
        # prominent action button (plain tk so the colour shows in the native light theme too)
        self.ref_btn = accent_button(bar, "  Refine  ", self.refine_selected, self.bold)
        self.ref_btn.configure(state="disabled")
        self.ref_btn.pack(side="left", padx=(0, 4))
        ttk.Button(bar, text="Adjust...", command=self.adjust_scheme).pack(side="left", padx=(0, 10))
        ttk.Label(bar, text="the scheme selected on the Schemes tab:").pack(side="left")
        self.ref_mode = tk.StringVar(value="All inferred H")
        for lab in REFINE_MODES:
            ttk.Radiobutton(bar, text=lab, variable=self.ref_mode, value=lab).pack(side="left", padx=3)
        d = hc.Settings()
        self.ref_n = tk.StringVar(value=str(d.refine_n_worst))
        self.ref_max = tk.StringVar(value=str(d.refine_max_deg))
        self.ref_sbv = tk.StringVar(value=str(d.refine_sigma_bv))
        self.ref_shh = tk.StringVar(value=str(d.refine_sigma_hh))
        self.ref_hdh = tk.StringVar(value=str(d.refine_hdh_deg))
        self.ref_wsym = tk.StringVar(value=str(d.refine_w_sym))
        self.ref_auto = tk.BooleanVar(value=d.refine_auto_pick)
        bar1b = ttk.Frame(tab)
        bar1b.pack(fill="x", pady=(2, 0))
        for parent_, lab, var in ((bar, "N worst", self.ref_n), (bar, "Max rotation °", self.ref_max),
                                  (bar, "H–D–H ±°", self.ref_hdh), (bar1b, "σ_bv vu", self.ref_sbv),
                                  (bar1b, "σ_hh Å", self.ref_shh), (bar1b, "Similar-strength weight", self.ref_wsym)):
            ttk.Label(parent_, text=lab).pack(side="left", padx=(10, 3))
            ttk.Entry(parent_, textvariable=var, width=6).pack(side="left")
        ttk.Checkbutton(bar1b, text="Auto-pick scheme (best without H···H violations among the top 20, else the "
                                    "least violating)", variable=self.ref_auto).pack(side="left", padx=(14, 0))
        self.ref_dh = tk.BooleanVar(value=d.refine_dh)
        self.ref_dhr = tk.StringVar(value=str(d.refine_dh_range))
        self.ref_dhw = tk.StringVar(value=str(d.refine_dh_weight))
        bar1c = ttk.Frame(tab)
        bar1c.pack(fill="x", pady=(2, 0))
        ttk.Checkbutton(bar1c, text="Also refine D–H lengths", variable=self.ref_dh).pack(side="left")
        for lab, var in (("± range Å (= 2σ)", self.ref_dhr), ("Restraint weight", self.ref_dhw)):
            ttk.Label(bar1c, text=lab).pack(side="left", padx=(10, 3))
            ttk.Entry(bar1c, textvariable=var, width=6).pack(side="left")
        ttk.Label(bar1c, foreground="#555", text="  each inferred D–H may change within ± range; cost = weight × "
                  "(ΔD–H/σ)², so H move only when BV misfit / H···H clashes outweigh it").pack(side="left")
        bar2 = ttk.Frame(tab)
        bar2.pack(fill="x", pady=4)
        ttk.Button(bar2, text="Add selected scheme (unrefined)", command=self.add_selected_scheme).pack(side="left")
        ttk.Button(bar2, text="Score H in CIF", command=self.score_cif_h).pack(side="left", padx=4)
        ttk.Button(bar2, text="Remove entry", command=self.remove_cmp).pack(side="left")
        ttk.Button(bar2, text="Clear", command=self.clear_cmp).pack(side="left", padx=4)
        ttk.Button(bar2, text="Finalize ▶", command=self.cmp_to_finalize).pack(side="right")
        ttk.Button(bar2, text="BV table ▶", command=self.cmp_table).pack(side="right", padx=4)
        self.ref_status = tk.StringVar()
        ttk.Label(bar2, textvariable=self.ref_status, foreground="#555").pack(side="left", padx=10)
        pw = ttk.PanedWindow(tab, orient="vertical")
        pw.pack(fill="both", expand=True)
        self.cmpt = ScrolledTree(pw, [
            ("name", "Entry", 260, "w"), ("total", "Total score", 90, "e"), ("bvf", "BV score F&I", 95, "e"),
            ("bvg", "BV score GRG", 95, "e"), ("geom", "Angle pen.", 80, "e"), ("hh", "H···H pen.", 80, "e"),
            ("red", "Red", 40, "e"), ("mod", "Mod.", 45, "e"), ("sev", "<1.75", 45, "e"),
            ("cone", "Cone", 45, "e"), ("dis", "Dis.", 40, "e"),
            ("maxf", "max outside F&I", 105, "e"), ("maxg", "max |Δ| GRG", 85, "e"), ("hcell", "H/cell", 60, "e"),
            ("info", "Refined donors (rotation °) / notes", 380, "w")], height=8)
        pw.add(self.cmpt, weight=2)
        self.cmpt.tree.tag_configure("red", background="#fecaca")
        self.cmpt.tree.tag_configure("mod", background="#fed7aa")
        self.cmpt.tree.tag_configure("cif", foreground="#1d4ed8")
        self.cmpt.tree.bind("<<TreeviewSelect>>", lambda e: self.show_cmp())
        det = ttk.Frame(pw)
        pw.add(det, weight=3)
        self.cmp_flag = tk.Label(det, text="", anchor="w", justify="left", padx=6, pady=3, wraplength=1180)
        self.cmp_flag.pack(fill="x")
        self.cmp_bv = BVTable(det, height=9)
        self.cmp_bv.pack(fill="both", expand=True)

    # ============================================================== subschemes
    def _build_subs_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="Subschemes")
        ttk.Label(tab, foreground="#555", wraplength=1180, justify="left", text=(
            "A disordered structure is an average of ordered configurations. Each subscheme is one of them: the "
            "structure in P1 (same cell) with one choice for every set of disordered positions (positions that "
            "exclude each other; partly vacant ones), and the H of the source scheme that fit it - donor and "
            "acceptor present; an H-bond to a vacant position goes to the occupied alternative nearby. Valid = every "
            "donor gets all its H, no H···H or H···atom clash; invalid ones can be shown, opened and exported for "
            "inspection. Scores are per asymmetric unit (P1 cell / number of symmetry operations), comparable with "
            "the Schemes tab. Empty for an ordered structure.")).pack(fill="x")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=(4, 0))
        ttk.Label(bar, text="Source:").pack(side="left")
        self.sub_src = tk.StringVar(value=SUB_SRC_SCHEME)
        self.sub_src_box = ttk.Combobox(bar, textvariable=self.sub_src, state="readonly", width=48,
                                        postcommand=self._sub_sources)
        self.sub_src_box.pack(side="left", padx=4)
        self.sub_btn = accent_button(bar, "  Find subschemes  ", self.find_subschemes, self.bold)
        self.sub_btn.pack(side="left", padx=(4, 10))
        self.sub_consol = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Consolidate symmetry-equivalent subschemes", variable=self.sub_consol,
                        command=self._fill_subs).pack(side="left")
        self.sub_bad = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Show invalid", variable=self.sub_bad, command=self._fill_subs).pack(
            side="left", padx=(10, 0))
        bar1 = ttk.Frame(tab)
        bar1.pack(fill="x", pady=(3, 0))
        self.sub_anew = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar1, text="Search each arrangement anew (its own best networks; otherwise the source scheme's "
                                   "H are inherited)", variable=self.sub_anew).pack(side="left")
        self.sub_nconf = tk.StringVar(value="30")
        self.sub_ntop = tk.StringVar(value="3")
        for lab, var in (("most frequent arrangements", self.sub_nconf), ("networks each", self.sub_ntop)):
            ttk.Entry(bar1, textvariable=var, width=4).pack(side="left", padx=(10, 3))
            ttk.Label(bar1, text=lab).pack(side="left")
        self.sub_link = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar1, text="Require linked layers (3D network when the candidates allow it)",
                        variable=self.sub_link).pack(side="left", padx=(14, 0))
        bar1b = ttk.Frame(tab)
        bar1b.pack(fill="x", pady=(3, 0))
        self.sub_formula = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar1b, text="Only formula-preserving subschemes: every element (H from the D/A classes) within "
                                    "±1 atom per formula unit of the Bond valence tab's model (±Z per cell)",
                        variable=self.sub_formula, command=self._refill_subs_keep_selection).pack(side="left")
        bar2 = ttk.Frame(tab)
        bar2.pack(fill="x", pady=4)
        ttk.Button(bar2, text="Open as structure", command=lambda: self.sub_open()).pack(side="left")
        ttk.Button(bar2, text="Refine ▶", command=lambda: self.sub_open("refine")).pack(side="left", padx=4)
        ttk.Button(bar2, text="Finalize ▶", command=lambda: self.sub_open("finalize")).pack(side="left")
        ttk.Button(bar2, text="Export all to CIFs", command=lambda: self.sub_export(True)).pack(side="right")
        ttk.Button(bar2, text="Export CIF", command=lambda: self.sub_export(False)).pack(side="right", padx=4)
        self.sub_status = tk.StringVar()
        ttk.Label(bar2, textvariable=self.sub_status, foreground="#555").pack(side="left", padx=10)
        pw = ttk.PanedWindow(tab, orient="vertical")
        pw.pack(fill="both", expand=True)
        self.subt = ScrolledTree(pw, [
            ("n", "#", 45, "e"), ("total", "Score / asym. unit", 115, "e"), ("bvf", "BV score F&I", 95, "e"),
            ("bvg", "BV score GRG", 95, "e"), ("geom", "Angle pen.", 80, "e"), ("hh", "H···H pen.", 80, "e"),
            ("red", "Red", 40, "e"), ("mod", "Mod.", 45, "e"), ("maxf", "max outside F&I", 105, "e"),
            ("w", "Frequency %", 85, "e"), ("eq", "×n", 40, "e"),
            ("conf", "Occupied disordered positions", 330, "w"), ("status", "Status", 420, "w"),
            ("comp", "Composition vs model (per f.u.)", 260, "w")],
            height=9, selectmode="extended")
        pw.add(self.subt, weight=2)
        self.subt.tree.tag_configure("hbad", foreground="#b91c1c")
        self.subt.tree.tag_configure("red", background="#fecaca")
        self.subt.tree.tag_configure("mod", background="#fed7aa")
        self.subt.tree.bind("<<TreeviewSelect>>", lambda e: self.show_sub())
        self.subt.tree.bind("<Double-1>", lambda e: self.sub_open())
        det = ttk.Frame(pw)
        pw.add(det, weight=3)
        self.sub_flag = tk.Label(det, text="", anchor="w", justify="left", padx=6, pady=3, wraplength=1180)
        self.sub_flag.pack(fill="x")
        self.sub_bv = BVTable(det, height=9)
        self.sub_bv.pack(fill="both", expand=True)

    def _sub_sources(self):
        vals = [SUB_SRC_SCHEME] + [SUB_SRC_CMP + e["name"] for e in self.cmp] + [SUB_SRC_FINAL]
        self.sub_src_box.configure(values=vals)
        return vals

    def _sub_source_hs(self):
        """(H list, description) of the chosen source, or (None, why)."""
        src = self.sub_src.get()
        if src.startswith(SUB_SRC_CMP):
            name = src[len(SUB_SRC_CMP):]
            e = next((e for e in self.cmp if e["name"] == name), None)
            return (list(e["hs"]), name) if e is not None else (None, "that Refinement entry no longer exists")
        if src == SUB_SRC_FINAL:
            if not self.checked and not self.fin_loc:
                return None, "nothing is ticked on the Finalize tab"
            return self.final_hatoms(), "Finalize selection (%s)" % (getattr(self, "_fin_src", "") or "by hand")
        if self.res is None or not self.res.schemes:
            return None, "no schemes - run the analysis first"
        r = self.selected_scheme()
        if r is None:
            r = 0
        return hc.scheme_hatoms(self.res, self.res.schemes[r]), "Scheme #%d" % (r + 1)

    def find_subschemes(self, note=""):
        if self.st is None:
            messagebox.showinfo(APP_NAME, "Run the analysis first.")
            return
        if self._sub_thread is not None and self._sub_thread.is_alive():
            return
        par = getattr(self.st, "sub_parent", None)
        if par is not None and not note:
            # an opened subscheme is ordered: its subschemes are the parent's - go back to that list
            parent = next((x for x in self.structures if x.title == par), None)
            if parent is not None:
                self.struct_var.set(parent.title)
                self.on_structure_change()
                self.nb.select(TAB_SUBS)
                if self.subs is None and self.res is not None:
                    self.find_subschemes()
                else:
                    self.status_var.set("Back to %s and its subschemes (the opened subscheme stays in the "
                                        "structure box)." % par)
                return
        hs, what = self._sub_source_hs()
        if hs is None:
            messagebox.showinfo(APP_NAME, "No source scheme: %s." % what)
            return
        st, bv, params = self.st, self.bv, self.params
        s = hc.replace(self.s, sub_require_link=bool(self.sub_link.get()))
        anew = bool(self.sub_anew.get())
        formula_only = bool(self.sub_formula.get())
        try:
            n_conf, n_top = max(1, int(self.sub_nconf.get())), max(1, int(self.sub_ntop.get()))
        except ValueError:
            messagebox.showerror(APP_NAME, "Arrangements and networks must be whole numbers.")
            return
        box, prog = {}, {"t": ""}

        def progress(n, tot):
            prog["t"] = "Configurations... %d / %d" % (n, tot)

        def progress2(n, tot):
            prog["t"] = "Searching arrangement %d / %d anew..." % (n, tot)

        def work():
            try:
                r = hc.find_subschemes(st, bv, hs, s, progress)
                ev0 = hc.evaluate_hset(st, bv, hs, s, params)
                r["src_total"], r["src_geom"] = ev0["total"], ev0["geom"]
                r["settings"] = s
                r["ref_cell"] = hc.cell_contents(st, bv)        # the model of the Bond valence tab
                r["z"], r["z_note"] = hc.effective_z(st, bv)
                if anew and r["groups"]:
                    n_all = len(r["groups"])
                    pre = ""
                    if formula_only:                  # search only arrangements whose atoms keep the formula
                        ref_noh = {e: v for e, v in r["ref_cell"].items() if e != "H"}
                        keep = []
                        for g in r["groups"]:
                            sub = {}
                            for c in g["members"][0]["present"]:
                                el = st.atoms[r["cell"].copies[c][0]].el
                                sub[el] = sub.get(el, 0.0) + 1.0
                            if hc.composition_check(ref_noh, sub, r["z"])[0]:
                                keep.append(g)
                        pre = " (pre-selected: %d of %d arrangements keep the formula in their non-H atoms)" % (
                            len(keep), n_all)
                        r["groups"] = keep
                    r["groups"] = hc.research_subschemes(st, bv, s, params, r, top=n_top, max_conf=n_conf,
                                                         progress=progress2) if r["groups"] else []
                    for g in r["groups"]:
                        if "eval" in g:
                            g["score"] = self._sub_score(g["eval"])
                        else:                         # searched: no network at all - never re-search it
                            g["no_network"] = True
                    nv = sum(1 for g in r["groups"] if g["valid"])
                    r["note"] = ("%s configuration(s) of the disordered positions%s, %d symmetry-distinct "
                                 "arrangements. Searched anew: the %d most frequent, up to %d distinct network(s) "
                                 "each - %d valid network(s)%s." % (
                                     "{:,}".format(r["n_conf"]), " (sampled)" if r["sampled"] else "", n_all,
                                     min(n_conf, n_all), n_top, nv, pre))
                    r["anew"] = True
                else:
                    valid = [g for g in r["groups"] if g["valid"]]
                    for k, g in enumerate(valid[:hc.SUB_MAX_EVAL]):
                        prog["t"] = "Scoring subschemes... %d / %d" % (k + 1, min(len(valid), hc.SUB_MAX_EVAL))
                        g["eval"] = self._sub_evaluate(st, bv, s, r["cell"], g["members"][0])
                        self._sub_link_check(g, s)
                        g["score"] = self._sub_score(g["eval"])
                box["r"] = r
            except Exception as e:  # noqa: BLE001
                box["err"] = "%s\n\n%s" % (e, traceback.format_exc(limit=5))

        self.sub_btn.configure(state="disabled")
        self.root.config(cursor="watch")
        self._sub_thread = threading.Thread(target=work, daemon=True)
        self._sub_thread.start()

        def poll():
            if self._sub_thread.is_alive():
                self.sub_status.set(prog["t"] or "Finding subschemes...")
                self.root.after(150, poll)
                return
            self.root.config(cursor="")
            self.sub_btn.configure(state="normal")
            if "err" in box:
                messagebox.showerror("Subschemes failed", box["err"])
                return
            if st is not self.st:                 # the structure changed meanwhile
                return
            r = box["r"]
            r["source"], r["src_key"] = what, self.sub_src.get()
            r["note"] = (note + " " if note else "") + r["note"]
            self.subs = r
            self._fill_subs()
            if not note:
                self.nb.select(TAB_SUBS)

        self.root.after(150, poll)

    def _sub_evaluate(self, st, bv, s, cell, conf, number=0):
        """Ordered P1 structure of a configuration with its network, analysed and scored."""
        st1 = hc.subscheme_structure(st, bv, cell, conf, number)
        bv1, loc, donors, res1, loc_all = hc.subscheme_analysis(st1, self.params, s)
        hs1 = hc.scheme_hatoms(res1, res1.schemes[0]) if res1.schemes else list(loc)
        ev = hc.evaluate_hset(st1, bv1, hs1, s, self.params)
        return {"st": st1, "bv": bv1, "loc": loc, "donors": donors, "res": res1, "loc_all": loc_all,
                "hs": hs1, "ev": ev}

    @staticmethod
    def _sub_link_check(g, s):
        """Inherited networks: invalid when they link fewer directions than the arrangement's candidates could
        (e.g. a water double layer whose H-bonds all point to the sheets)."""
        if not g["valid"] or not getattr(s, "sub_require_link", True) or "eval" not in g:
            return
        e = g["eval"]
        dim, pot, txt = hc.subscheme_connectivity(e["st"], e["bv"], e["hs"], e["donors"])
        g["dim"] = dim
        if dim < pot:
            g["valid"] = False
            g["reasons"] = list(g["reasons"]) + [
                "layers not linked - %s (the candidate H-bonds could link it in %dD)" % (txt, pot)]

    def _sub_score(self, e):
        """Scores of a subscheme per asymmetric unit (P1 cell / number of symmetry operations), so that they
        compare with the disordered structure's schemes; contact counts and max misfit as in the P1 cell."""
        ev, n = e["ev"], max(1, getattr(e["st"], "sub_units", 1))
        base = self.s.score_base
        out = {k: ev[k] / n for k in ("bvF", "bvG", "geom", "hh")}
        out["total"] = base + (ev["total"] - base) / n
        out.update({k: ev[k] for k in ("red", "mod", "maxF")})
        return out

    def _fill_subs(self):
        self.subt.clear()
        self.sub_bv.clear()
        self.sub_flag.configure(text="", background=self.root.cget("background"))
        self.sub_rows = []
        r = self.subs
        if r is None:
            par = getattr(self.st, "sub_parent", None) if self.st is not None else None
            self.sub_status.set(
                "This is subscheme %s of %s, opened as an ordered P1 structure - it has no subschemes of its own. "
                "The list belongs to the parent: click 'Find subschemes' (or choose the parent in the structure box "
                "above) to go back to it." % (getattr(self.st, "sub_number", "?"), par) if par else "")
            return
        groups = r["groups"]
        n_valid = sum(1 for g in groups if g["valid"])
        show_bad = self.sub_bad.get() or n_valid == 0
        order = sorted(range(len(groups)), key=lambda k: (
            (0, "score" not in groups[k], -groups[k]["score"]["total"] if "score" in groups[k] else -groups[k]["weight"])
            if groups[k]["valid"] else (1, False, -groups[k]["weight"])))
        n_comp_ok = 0
        for k in order:
            g = groups[k]
            ok_c, _t = self._sub_comp(g, 0)
            n_comp_ok += bool(ok_c)
            if self.sub_formula.get() and ok_c is False:
                continue
            if not g["valid"] and not show_bad:
                continue
            if self.sub_consol.get():
                self.sub_rows.append((k, None))
            else:
                self.sub_rows += [(k, m) for m in range(len(g["members"]))]
        for n, (k, m) in enumerate(self.sub_rows, 1):
            g = groups[k]
            sc = g.get("score")
            conf = g["members"][m if m is not None else 0]
            w = g["weight"] if m is None else conf["weight"]
            status = "valid" if g["valid"] else "invalid: " + "; ".join(g["reasons"][:4]) + (
                " ..." if len(g["reasons"]) > 4 else "")
            if g["valid"] and sc is None:
                status = "valid (not scored yet - select it)"
            elif g["valid"] and sc["geom"] < r.get("src_geom", 0.0) - 150:
                status = "valid; strained H directions (angle penalty %.0f vs %.0f in the source scheme - e.g. a " \
                         "cation present here blocks them): refine it" % (sc["geom"], r.get("src_geom", 0.0))
            if conf.get("moved"):
                status += "; H-bond(s) moved to the occupied alternative: " + ", ".join(conf["moved"][:4]) + (
                    " ..." if len(conf["moved"]) > 4 else "")
            if g.get("component"):
                status = "component %d: %s" % (g["component"], status)
            tag = ("hbad",) if not g["valid"] else (("red",) if sc and sc["red"] else (("mod",) if sc and sc["mod"] else ()))
            self.subt.tree.insert("", "end", iid=str(n - 1), tags=tag, values=(
                n, _fmt(sc["total"], 1) if sc else "", _fmt(sc["bvF"], 2) if sc else "",
                _fmt(sc["bvG"], 2) if sc else "", _fmt(sc["geom"], 1) if sc else "", _fmt(sc["hh"], 0) if sc else "",
                sc["red"] if sc else "", sc["mod"] if sc else "", _fmt(sc["maxF"]) if sc else "",
                "%.2f" % (100 * w), g.get("equiv", len(g["members"])) if m is None else "", conf["desc"] or "-",
                status, self._sub_comp(g, m or 0)[1]))
        txt = "Source: %s%s. %s" % (r.get("source", "?"), (" (score %.1f)" % r["src_total"]) if "src_total" in r
                                    else "", r["note"])
        if n_valid == 0 and groups:
            txt += " No valid network - the invalid ones are listed with the reasons (the source scheme cannot be " \
                   "realised as an ordered arrangement as it is)."
        if "ref_cell" in r:
            z = r["z"]
            txt += "  Model per formula unit (Bond valence tab, Z = %g%s): %s; %d of %d preserve it within ±1 atom%s." % (
                z, ("; " + r["z_note"]) if r.get("z_note") else "",
                hc.formula_text({e: v / z for e, v in r["ref_cell"].items()}), n_comp_ok, len(groups),
                " (only these are listed)" if self.sub_formula.get() else "")
        self.sub_status.set(txt)

    def _sub_comp(self, g, m):
        """(None / True / False, text): composition of a subscheme's P1 cell against the model of the Bond valence
        tab (every element within ±1 atom per formula unit). The H come from its network (computed when known)."""
        r = self.subs
        if r is None or "ref_cell" not in r:
            return None, ""
        e = g.get("eval") if m == 0 else g.get("member_eval", {}).get(m)
        if e is not None:
            sub = hc.subscheme_composition(e["st"], e["hs"])
        elif g.get("searched"):
            return None, "no network"
        else:                                           # inherited: atoms present + its H records
            conf = g["members"][m]
            sub = {}
            cell = r["cell"]
            for c in conf["present"]:
                el = cell.st.atoms[cell.copies[c][0]].el
                sub[el] = sub.get(el, 0.0) + 1.0
            nh = len(conf["hrecs"])
            if nh:
                sub["H"] = float(nh)
        ok, txt = hc.composition_check(r["ref_cell"], sub, r["z"])
        return ok, ("✓ " if ok else "✗ ") + txt

    def _sub_row(self, n):
        k, m = self.sub_rows[n]
        g = self.subs["groups"][k]
        return g, (m if m is not None else 0)

    def _sub_eval_row(self, n):
        """Evaluation (P1 structure, analysis, score) of displayed row n (computed when first needed)."""
        g, m = self._sub_row(n)
        s = self.subs.get("settings", self.s)
        if m == 0 and "eval" not in g and g.get("no_network"):
            return None
        if m == 0 and "eval" not in g and g.get("searched"):
            # searched anew (results not stored in the analysis file): search this arrangement again
            src = {"cell": self.subs["cell"], "groups": [{"members": g["members"], "weight": g["weight"],
                                                          "key": g["key"][0]}]}
            nets = hc.research_subschemes(self.st, self.bv, s, self.params, src, top=g["key"][1] + 1, max_conf=1)
            hit = next((x for x in nets if x["key"][1] == g["key"][1]), nets[0] if nets else None)
            if hit is None or "eval" not in hit:
                return None
            g["eval"] = hit["eval"]
            g["score"] = self._sub_score(g["eval"])
            return g["eval"]
        if m == 0:
            if "eval" not in g:
                g["eval"] = self._sub_evaluate(self.st, self.bv, s, self.subs["cell"], g["members"][0])
                self._sub_link_check(g, s)
                g["score"] = self._sub_score(g["eval"])
            return g["eval"]
        ev = g.setdefault("member_eval", {})
        if m not in ev:
            ev[m] = self._sub_evaluate(self.st, self.bv, self.s, self.subs["cell"], g["members"][m])
        return ev[m]

    def _sub_selected_rows(self):
        return sorted(int(x) for x in self.subt.tree.selection())

    def show_sub(self):
        rows = self._sub_selected_rows()
        if not rows or self.subs is None:
            return
        g, _m = self._sub_row(rows[0])
        had = "score" in g
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            e = self._sub_eval_row(rows[0])
        finally:
            self.root.config(cursor="")
        if e is None:
            self.sub_bv.clear()
            self.sub_flag.configure(text="Invalid: " + "; ".join(g["reasons"]), background="#fecaca",
                                    foreground="black")
            return
        if not had:                                 # show the new score (the row may move in the list)
            self._refill_subs_keep_selection()
        labels = hc.label_hydrogens(e["st"], e["hs"])
        self.sub_bv.fill(e["st"], e["bv"], e["hs"], self.s, self.params, labels)
        if g["valid"]:
            self._flag_box(self.sub_flag, e["ev"]["contacts"], labels, e["hs"], st=e["st"])
        else:
            self.sub_flag.configure(text="Invalid (BV sums below are for the H that fit; export or open it to inspect): "
                                         + "; ".join(g["reasons"]), background="#fecaca", foreground="black")

    def _refill_subs_keep_selection(self):
        keep = [self.sub_rows[int(x)] for x in self.subt.tree.selection() if int(x) < len(self.sub_rows)]
        self._fill_subs()
        sel = [str(n) for n, km in enumerate(self.sub_rows) if km in keep]
        if sel:
            self.subt.tree.selection_set(sel)
            self.subt.tree.see(sel[0])

    def sub_open(self, then=None):
        """Open the selected subscheme as a structure of its own ('<title> [subscheme n]'), with its network as
        the only scheme: refine it, finalize it, export it like any structure."""
        rows = self._sub_selected_rows()
        if not rows or self.subs is None:
            messagebox.showinfo(APP_NAME, "Select a subscheme first.")
            return
        n = rows[0]
        key = self.sub_rows[n]
        self.subt.tree.selection_set(str(n))
        e = self._sub_eval_row(n)
        if e is None:
            messagebox.showinfo(APP_NAME, "No H-bond network was found for this arrangement: "
                                + "; ".join(self._sub_row(n)[0]["reasons"]))
            return
        self._refill_subs_keep_selection()        # scored now: its place (number) in the list is final
        n = self.sub_rows.index(key)
        g, _m = self._sub_row(n)
        st1 = e["st"]
        st1.sub_reasons = [] if g["valid"] else list(g["reasons"])
        st1.sub_number = n + 1
        base = " [subscheme %d]" % (n + 1)
        titles = {x.title for x in self.structures if x is not st1}
        st1.tag, k = base, 2
        while st1.title in titles:
            st1.tag = "%s (%d)" % (base, k)
            k += 1
        st1.sub_source = source = self.subs.get("source", "")
        self._snapshot_current()
        if st1 not in self.structures:
            self.structures.append(st1)
        self.struct_box.configure(values=[x.title for x in self.structures], state="readonly")
        self.struct_var.set(st1.title)
        s = self.s
        located = hc.located_hbonds(st1, env_radius=s.env_radius, **self.located_args(s))
        self.clear_all()
        self._clear_pending()
        self._set_info(st1)
        if getattr(st1, "forced_roles", None):              # kept on a re-run of the subscheme
            self.role_overrides[st1.title] = dict(st1.forced_roles)
        self._apply_payload((st1, s, e["bv"], e["loc"], e["donors"], e["res"], located, e["loc_all"]))
        self._fill_subs()                                   # says where the parent's list is
        r = e.get("scheme_index", 0)
        if self.res.schemes and r >= len(self.res.schemes):
            r = 0
        self.status_var.set("%sOpened subscheme %d of %s (%s) as an ordered P1 structure - its network is scheme #%d%s. "
                            "The parent structure is kept: choose it above to go back." % (
                                "" if g["valid"] else "⚠ INVALID network, opened for inspection (see the Bond "
                                                      "valence notes). ", n + 1, st1.sub_parent, source, r + 1,
                                " (this arrangement was searched anew: its other networks are listed too)"
                                if g.get("searched") else ""))
        if self.res.schemes:
            self.sch.tree.selection_set(str(r))
            self.sch.tree.see(str(r))
        if then == "refine":
            self.goto_refine()
        elif then == "finalize":
            self.load_scheme(r)
        else:
            self.nb.select(TAB_SCHEMES)

    def sub_export(self, all_rows):
        """P1 CIF(s) of the selected (or all listed - invalid ones too when shown) subschemes: one ->
        <name>_hypothetical_subscheme<n>.cif next to the CIF; several -> a new folder
        <name>_hypothetical_subschemes. An invalid one is written with the H that fit and its reasons."""
        if self.subs is None or not self.sub_rows:
            messagebox.showinfo(APP_NAME, "Find subschemes first.")
            return
        rows = list(range(len(self.sub_rows))) if all_rows else self._sub_selected_rows()
        if not rows:
            messagebox.showinfo(APP_NAME, "Select one or more subschemes first.")
            return
        self.root.config(cursor="watch")
        try:
            # score them first (a newly scored row can move in the list), then number them as listed
            keys = [self.sub_rows[n] for n in rows]
            for k, n in enumerate(rows, 1):
                self.sub_status.set("Scoring subscheme %d / %d..." % (k, len(rows)))
                self.root.update_idletasks()
                if self._sub_eval_row(n) is None:              # no network at all for this arrangement
                    keys.remove(self.sub_rows[n])
            if not keys:
                self.root.config(cursor="")
                messagebox.showinfo(APP_NAME, "No H-bond network to write for the selected arrangement(s).")
                return
            if not all_rows:
                self.subt.tree.selection_set([str(n) for n in rows])
            self._refill_subs_keep_selection()
            rows = sorted(self.sub_rows.index(k) for k in keys)
            folder, paths = hc.subscheme_paths(self.st.source, [n + 1 for n in rows],
                                               optimized=bool(getattr(self.st, "optimized", None)))
            if folder:
                os.makedirs(folder)
            for k, n in enumerate(rows, 1):
                self.sub_status.set("Writing subscheme %d (%d / %d)..." % (n + 1, k, len(rows)))
                self.root.update_idletasks()
                e = self._sub_eval_row(n)
                e["st"].sub_number = n + 1
                g_n = self._sub_row(n)[0]
                e["st"].sub_reasons = [] if g_n["valid"] else list(g_n["reasons"])
                txt = hc.subscheme_cif_text(e["st"], e["bv"], e["hs"],
                                            "source: %s" % self.subs.get("source", ""))
                with open(paths[n + 1], "w", encoding="utf-8") as fh:
                    fh.write(txt)
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror(APP_NAME, "Could not write the CIF(s):\n%s" % ex)
            return
        finally:
            self.root.config(cursor="")
        where = folder if folder else paths[rows[0] + 1]
        self.sub_status.set("Wrote %d subscheme CIF(s): %s" % (len(rows), where))
        self.status_var.set("Wrote %s" % where)

    def write_cif(self, st, bv, hs):
        """Write the CIF for H set hs: a subscheme structure -> its own P1 CIF
        (<name>_hypothetical_subscheme<n>.cif); otherwise the source CIF with the H added. Returns the path."""
        if getattr(st, "sub_parent", None) is not None:
            _f, paths = hc.subscheme_paths(st.source, [getattr(st, "sub_number", 0) or 0],
                                           optimized=bool(getattr(st, "optimized", None)))
            path = next(iter(paths.values()))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(hc.subscheme_cif_text(st, bv, hs, "source: %s" % getattr(st, "sub_source", "")))
            return path
        path, _labels = hc.write_hypothetical_cif(st, hs, self.s, drop_cif_h=self.s.ignore_located, bv=bv)
        return path

    def _subs_follow_refine(self, r, new_name):
        """After 'Scheme #r' was refined: subschemes found from it are re-derived from the refined scheme."""
        if self.subs is None:
            return
        key = self.subs.get("src_key", "")
        src = self.subs.get("source", "")
        if src == "Scheme #%d" % (r + 1) and key in (SUB_SRC_SCHEME, SUB_SRC_CMP + "Scheme #%d" % (r + 1)):
            self.sub_src.set(SUB_SRC_CMP + new_name)
            self.find_subschemes(note="Updated from the refined scheme.")

    def _build_details_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="H-bond details")
        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Label(bar, text="Show:").pack(side="left")
        self.det_src = tk.StringVar(value="scheme")
        for lab, val in (("Selected scheme", "scheme"), ("Refinement entry", "refine"),
                         ("Finalize selection", "final")):
            ttk.Radiobutton(bar, text=lab, variable=self.det_src, value=val,
                            command=self.refresh_details).pack(side="left", padx=4)
        ttk.Button(bar, text="Copy", command=self.copy_details).pack(side="right")
        ttk.Button(bar, text="Save .csv...", command=lambda: self.save_details("csv")).pack(side="right", padx=4)
        ttk.Button(bar, text="Save .txt...", command=lambda: self.save_details("txt")).pack(side="right")
        tf = ttk.Frame(tab)
        tf.pack(fill="both", expand=True, pady=(4, 0))
        self.det_text = tk.Text(tf, wrap="none", font=("Consolas", 10), relief="flat", background="#fcfcfc")
        ys = ttk.Scrollbar(tf, orient="vertical", command=self.det_text.yview)
        xs = ttk.Scrollbar(tf, orient="horizontal", command=self.det_text.xview)
        self.det_text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.det_text.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        tf.rowconfigure(0, weight=1)
        tf.columnconfigure(0, weight=1)
        self.det_data = None

    def refresh_details(self):
        src = self.det_src.get()
        hs, what = None, ""
        if self.st is not None:
            if src == "final":
                hs, what = self.final_hatoms(), "Finalize selection"
            elif src == "refine":
                e = self.selected_cmp()
                hs, what = (e["hs"], e["name"]) if e else (None, "")
            else:
                r = self.selected_scheme()
                if r is not None:
                    hs, what = hc.scheme_hatoms(self.res, self.res.schemes[r]), "scheme #%d" % (r + 1)
                elif self.loc_h:
                    hs, what = list(self.loc_h), "located H only"
        self.det_text.configure(state="normal")
        self.det_text.delete("1.0", "end")
        if hs is None:
            self.det_data = None
            self.det_text.insert("1.0", "Run the analysis and select a scheme, a refinement entry or Finalize H atoms.")
        else:
            title, header, rows, notes = hc.hbond_details(self.st, self.bv, hs, self.s)
            title += "  [%s]" % what
            self.det_data = (title, header, rows, notes)
            self.det_text.insert("1.0", hc.table_text(title, header, rows, notes, left=(0, 5, 6)))
        self.det_text.configure(state="disabled")

    def copy_details(self):
        if self.det_data:
            self.root.clipboard_clear()
            self.root.clipboard_append(hc.table_text(*self.det_data, left=(0, 5, 6)))
            self.status_var.set("H-bond details copied to the clipboard.")

    def save_details(self, kind):
        if not self.det_data:
            return
        base = os.path.splitext(os.path.basename(self.st.source))[0] + "_hbond_details"
        path = filedialog.asksaveasfilename(
            title="Save H-bond details", defaultextension="." + kind, initialfile=base + "." + kind,
            initialdir=os.path.dirname(self.st.source),
            filetypes=[("Text", "*.txt")] if kind == "txt" else [("CSV", "*.csv")])
        if not path:
            return
        title, header, rows, notes = self.det_data
        if kind == "txt":
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(hc.table_text(title, header, rows, notes, left=(0, 5, 6)) + "\n")
        else:
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow([title])
                w.writerow(header)
                for row in rows:
                    w.writerow(["\n".join(c) for c in row])
                for n in notes:
                    w.writerow([n])
        self.status_var.set("Saved %s" % path)

    # ======================================================== SHELXL output
    def shelx_dialog(self):
        if self.st is None:
            return
        st = self.st
        win = tk.Toplevel(self.root)
        win.title("%s - SHELXL output for %s" % (APP_NAME, st.name))
        win.geometry("980x640")
        ico = _resource("HydroGEN.ico")
        if ico:
            try:
                win.iconbitmap(ico)
            except tk.TclError:
                pass
        state = {"res": st.res_text, "src": "the .res embedded in the CIF" if st.res_text else None, "out": None}
        top = ttk.Frame(win, padding=6)
        top.pack(fill="x")
        src_var = tk.StringVar()
        ttk.Label(top, textvariable=src_var, wraplength=940, justify="left").pack(fill="x")
        bar = ttk.Frame(win, padding=(6, 0))
        bar.pack(fill="x")
        txt = tk.Text(win, wrap="none", font=("Consolas", 10))
        dark = self.theme.get() == "dark"
        txt.configure(background="#2a2f36" if dark else "#fcfcfc", foreground="#e5e7eb" if dark else "#111111",
                      insertbackground="#e5e7eb" if dark else "#111111")
        ys = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y", pady=6)
        txt.pack(fill="both", expand=True, padx=6, pady=6)

        def regenerate():
            hs = self.final_hatoms()
            out = hc.shelx_export(st, hs, self.s, state["res"], drop_cif_h=self.s.ignore_located)
            state["out"] = out
            n = sum(1 for h in hs if not h.located)
            if state["res"]:
                src_var.set("Using %s. 'Write .ins' saves a copy of it with the %d hypothetical H inserted after "
                            "their parents and the DFIX restraints added (and the .hkl from the CIF, if embedded), "
                            "ready for SHELXL. H names are shortened to <= 4 characters." % (state["src"], n))
            else:
                src_var.set("No .res embedded in this CIF and none loaded - showing the SHELXL lines for the %d "
                            "hypothetical H only. Load a .res to write a complete .ins." % n)
            body = out["text"]
            if out["warnings"]:
                body = "REM " + "\nREM ".join(out["warnings"]) + "\n\n" + body
            if not n:
                body = "REM No hypothetical H selected on the Finalize tab.\n\n" + body
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            txt.insert("1.0", body)
            ins_btn.configure(state="normal" if (state["res"] and n) else "disabled")

        def load_res():
            p = filedialog.askopenfilename(parent=win, title="Load SHELXL .res", filetypes=[
                ("SHELXL .res / .ins", "*.res *.ins"), ("All files", "*.*")])
            if p:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    state["res"] = fh.read()
                state["src"] = os.path.basename(p)
                regenerate()

        def use_embedded():
            state["res"], state["src"] = st.res_text, "the .res embedded in the CIF"
            regenerate()

        def write_ins():
            out = state["out"]
            if not out or not out["ins"]:
                return
            ins_path, hkl_path = hc.shelx_output_paths(st.source, optimized=bool(getattr(st, "optimized", None)))
            with open(ins_path, "w", encoding="utf-8", newline="\r\n") as fh:
                fh.write(out["ins"])
            msg = "Wrote %s" % ins_path
            if st.hkl_text:
                with open(hkl_path, "w", encoding="utf-8", newline="\r\n") as fh:
                    fh.write(st.hkl_text.strip("\n") + "\n")
                msg += " and %s" % os.path.basename(hkl_path)
            else:
                msg += " (no .hkl embedded in the CIF: copy your .hkl to %s)" % os.path.basename(hkl_path)
            src_var.set(msg + ".")
            self.status_var.set(msg)

        ttk.Button(bar, text="Load .res...", command=load_res).pack(side="left")
        ttk.Button(bar, text="Use .res embedded in CIF", command=use_embedded,
                   state="normal" if st.res_text else "disabled").pack(side="left", padx=4)
        ins_btn = ttk.Button(bar, text="Write .ins (+ .hkl)", command=write_ins)
        ins_btn.pack(side="left", padx=(12, 4))

        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(txt.get("1.0", "end"))
        ttk.Button(bar, text="Copy lines", command=copy).pack(side="left")
        ttk.Button(bar, text="Close", command=win.destroy).pack(side="right")
        regenerate()
        self._shelx_win = (win, txt, regenerate, write_ins, load_res, state, src_var)

    def _build_table_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="BV table")
        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Label(bar, text="Show:").pack(side="left")
        self.table_src = tk.StringVar(value="scheme")
        ttk.Radiobutton(bar, text="Selected scheme", variable=self.table_src, value="scheme",
                        command=self.refresh_table).pack(side="left", padx=4)
        ttk.Radiobutton(bar, text="Refinement entry", variable=self.table_src, value="refine",
                        command=self.refresh_table).pack(side="left", padx=4)
        ttk.Radiobutton(bar, text="Finalize selection", variable=self.table_src, value="final",
                        command=self.refresh_table).pack(side="left", padx=4)
        ttk.Label(bar, text="H-bond valences:").pack(side="left", padx=(16, 3))
        self.table_method = tk.StringVar(value="Ferraris & Ivaldi")
        cb = ttk.Combobox(bar, textvariable=self.table_method, values=["Ferraris & Ivaldi", "GRG (G&H 2015)"],
                          state="readonly", width=18)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_table())
        ttk.Button(bar, text="Copy", command=self.copy_table).pack(side="right")
        ttk.Button(bar, text="Save .csv...", command=lambda: self.save_table("csv")).pack(side="right", padx=4)
        ttk.Button(bar, text="Save .txt...", command=lambda: self.save_table("txt")).pack(side="right")
        tf = ttk.Frame(tab)
        tf.pack(fill="both", expand=True, pady=(4, 0))
        self.table_text = tk.Text(tf, wrap="none", font=("Consolas", 10), relief="flat", background="#fcfcfc")
        ys = ttk.Scrollbar(tf, orient="vertical", command=self.table_text.yview)
        xs = ttk.Scrollbar(tf, orient="horizontal", command=self.table_text.xview)
        self.table_text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.table_text.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        tf.rowconfigure(0, weight=1)
        tf.columnconfigure(0, weight=1)

    def _build_located_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="Located H-bonds")
        self.loc = ScrolledTree(tab, [
            ("bond", "D–H···A", 190, "w"), ("d_DH", "D–H (Å)", 75, "e"), ("d_HA", "H···A (Å)", 80, "e"),
            ("d_DA", "D···A (Å)", 80, "e"), ("angle", "D–H···A (°)", 90, "e"),
            ("symop", "Symmetry op. for A", 190, "w"), ("code", "Code", 70, "w")])
        self.loc.pack(fill="both", expand=True)
        self.loc_note = tk.StringVar()
        ttk.Label(tab, textvariable=self.loc_note, foreground="#a05000").pack(fill="x")

    def _build_env_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="Environment")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Label(top, text="Site:").pack(side="left")
        self.env_site = tk.StringVar()
        self.env_box = ttk.Combobox(top, textvariable=self.env_site, state="disabled", width=12)
        self.env_box.pack(side="left", padx=4)
        self.env_box.bind("<<ComboboxSelected>>", lambda e: self.show_env())
        ttk.Label(top, text="Angles between neighbours within (Å):").pack(side="left", padx=(12, 3))
        self.angle_radius = tk.StringVar(value="3.0")
        ttk.Entry(top, textvariable=self.angle_radius, width=6).pack(side="left")
        ttk.Button(top, text="Update", command=self.show_env).pack(side="left", padx=6)
        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="both", expand=True, pady=(4, 0))
        self.env = ScrolledTree(pw, [
            ("sum", "Sum", 40, "center"),
            ("nb", "Neighbour", 75, "w"), ("el", "El", 35, "w"), ("d", "d (Å)", 65, "e"),
            ("s", "s (vu)", 60, "e"), ("note", "Note", 230, "w"), ("symop", "Symmetry op.", 150, "w"),
            ("code", "Code", 70, "w")])
        pw.add(self.env, weight=3)
        self.env.tree.tag_configure("manual", foreground="#b45309")
        self.env.tree.tag_configure("off", foreground="#9ca3af")
        # click the Sum box (or double-click a row) to include / exclude that bond - queued for Run analysis
        self.env.tree.bind("<Button-1>", lambda e: self.env_toggle(e) if self.env.tree.identify_column(e.x) == "#1"
                           else None)
        self.env.tree.bind("<Double-1>", self.env_toggle)
        ttk.Label(tab, foreground="#555", text=(
            "Sum ☑ / ☐: click to include or exclude a bond in BOTH atoms' sums (applied at the next Run analysis). "
            "Bonds between two configurations of a disorder, and bonds of two partly occupied atoms that would "
            "exceed a valence, are unticked by default.")).pack(fill="x", pady=(4, 0))
        self.angles = ScrolledTree(pw, [
            ("x", "X", 70, "w"), ("y", "Y", 70, "w"), ("ang", "X–site–Y (°)", 90, "e"),
            ("dx", "d(X) (Å)", 65, "e"), ("dy", "d(Y) (Å)", 65, "e")])
        pw.add(self.angles, weight=2)

    def _build_finalize_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text="Finalize & CIF")
        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Button(bar, text="Load best scheme", command=lambda: self.load_scheme(0)).pack(side="left")
        ttk.Button(bar, text="Check all passing (bold)", command=self.check_passing).pack(side="left", padx=4)
        ttk.Button(bar, text="Uncheck all", command=self.uncheck_all).pack(side="left")
        self.cif_btn = accent_button(bar, "  Export hypothetical CIF  ", self.export_cif, self.bold)
        self.cif_btn.configure(state="disabled")
        self.cif_btn.pack(side="left", padx=(16, 4))
        ttk.Button(bar, text="SHELXL (.res / DFIX)...", command=self.shelx_dialog).pack(side="left", padx=(0, 4))
        ttk.Button(bar, text="BV table ▶", command=self.show_final_table).pack(side="left")
        self.fin_status = tk.StringVar(value="Click ☐ (or double-click a row, or select rows and press Space) "
                                             "to include an H atom. Bold = passes all tests; ■ = located in the CIF.")
        ttk.Label(bar, textvariable=self.fin_status, foreground="#555").pack(side="left", padx=8)
        self.fin_source = tk.StringVar(value="")
        ttk.Label(tab, textvariable=self.fin_source, font=self.bold).pack(fill="x", pady=(4, 0))
        self._fin_loaded_cmp = None
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")
        self.fin_flag = tk.Label(tab, text="", anchor="w", justify="left", padx=6, pady=4, wraplength=1180)
        self.fin_flag.pack(fill="x", pady=(4, 2))
        self.fin_hc = tk.Label(tab, text="", anchor="w", justify="left", padx=6)
        self.fin_hc.pack(fill="x")
        pw = ttk.PanedWindow(tab, orient="vertical")
        pw.pack(fill="both", expand=True)
        self.fin = ScrolledTree(pw, [
            ("chk", "Use", 40, "center"), ("label", "Label", 80, "w"), ("acc", "Acceptor", 95, "w"),
            ("da", "D···A (Å)", 70, "e"), ("ha", "H···A (Å)", 70, "e"), ("dha", "D–H···A (°)", 80, "e"),
            ("opt", "In option(s)", 120, "w"), ("tests", "Tests", 170, "w"), ("sym", "Occupancy", 110, "w"),
            ("flag", "H···H flag (vs. all H in use)", 260, "w")],
            tree_heading="Donor / H", tree_width=230, height=12, selectmode="extended")
        pw.add(self.fin, weight=3)
        t = self.fin.tree
        t.tag_configure("pass", font=self.bold)
        t.tag_configure("located", font=self.italic, foreground="#1d4ed8")
        t.tag_configure("red", background="#fecaca")
        t.tag_configure("mod", background="#fed7aa")
        t.tag_configure("donor", background="#eef2ff")
        t.bind("<Button-1>", self.fin_click)
        t.bind("<Double-1>", self.fin_double)
        t.bind("<space>", self.fin_space)
        self.fin_bv = BVTable(pw, height=8)
        pw.add(self.fin_bv, weight=2)

    # ============================================================== file input
    def browse(self):
        paths = filedialog.askopenfilenames(
            title="Select CIF file(s)", filetypes=[("CIF files", "*.cif *.CIF"), ("All files", "*.*")])
        if not paths:
            return
        structures, errors = [], []
        for p in paths:
            try:
                structures.extend(hc.load_structures(p))
            except Exception as e:  # noqa: BLE001
                errors.append("%s: %s" % (os.path.basename(p), e))
        if errors:
            messagebox.showwarning("Some files could not be read", "\n\n".join(errors))
        if not structures:
            return
        self.clear_all()
        self._cache = {}
        self.structures = structures
        self.project_path = None
        self.root.title(APP_TITLE)
        names = [s.title for s in structures]
        self.struct_box.configure(values=names, state="readonly")
        self.struct_var.set(names[0])
        self.on_structure_change()
        self.run_btn.configure(state="normal" if self.params else "disabled")

    # ========================================================== lower symmetry
    def lower_symmetry_dialog(self):
        """Describe the current structure in an index-2 subgroup (written as a new CIF and added to the
        structure list): H disordered across the lost symmetry element can then be inferred ordered."""
        st = self.current_structure()
        if st is None:
            messagebox.showinfo(APP_NAME, "Open a CIF first.")
            return
        try:
            s = self.settings()
        except ValueError as e:
            messagebox.showerror("Invalid settings", str(e))
            return
        rows, pairs = hc.subgroup_report(st, s)
        if not rows:
            messagebox.showinfo(APP_NAME, "%s has no subgroup of index 2 to offer (%d operations)."
                                % (st.spacegroup, len(st.symops)))
            return
        win = tk.Toplevel(self.root)
        win.title("%s - lower symmetry (%s, %s)" % (APP_NAME, st.name, st.spacegroup))
        win.geometry("900x520")
        f = ttk.Frame(win, padding=8)
        f.pack(fill="both", expand=True)
        hp = [(i, op, d) for i, op, d in pairs if st.atoms[i].el == "H"]
        txt = ("Describes the structure in a subgroup of index 2 of %s (same cell). Sites whose symmetry copies "
               "are no longer equivalent become two independent sites (label + a / b). An H disordered across a "
               "lost symmetry element (e.g. an H-bond ring running either way round) can then be inferred as "
               "ORDERED - each half of the disorder in its own site. The new CIF is written next to the "
               "original and added to the structure list; then press Run analysis." % st.spacegroup)
        ttk.Label(f, text=txt, wraplength=860, justify="left").pack(fill="x")
        ttk.Label(f, text=("H disordered across a symmetry element: " + "; ".join(
            "%s ↔ %s %s (%.2f Å)" % (st.atoms[i].label, st.atoms[i].label, st.symop_text(op, (0, 0, 0)), d)
            for i, op, d in hp)) if hp else "No half-occupied H next to its own symmetry copy was found "
                                            "(no recommendation).", foreground="#a05000", wraplength=860,
                  justify="left").pack(fill="x", pady=(4, 4))
        t = ScrolledTree(f, [("name", "Subgroup", 90, "w"), ("sep", "Disordered H pairs separated", 170, "e"),
                             ("split", "Sites split", 80, "e"), ("rec", "", 110, "w"),
                             ("ops", "Operations kept", 420, "w")], height=8)
        t.pack(fill="both", expand=True)
        t.tree.tag_configure("rec", background="#bbf7d0", foreground="black")
        for k, (idx, name, sep, nsplit, rec) in enumerate(rows):
            t.tree.insert("", "end", iid=str(k), tags=("rec",) if rec else (), values=(
                name, sep, nsplit, "recommended" if rec else "",
                "; ".join(st.symop_text(q, (0, 0, 0)) for q in idx)))
        t.tree.selection_set("0")
        hmode = tk.StringVar(value="partial")
        hb = ttk.Frame(f)
        hb.pack(fill="x", pady=(6, 0))
        ttk.Label(hb, text="H atoms in the CIF:").pack(side="left")
        for lab, val in (("remove the disordered (partly occupied) H - re-inferred ordered", "partial"),
                         ("remove all H", "all"), ("keep all", "none")):
            ttk.Radiobutton(hb, text=lab, variable=hmode, value=val).pack(side="left", padx=6)
        bb = ttk.Frame(f)
        bb.pack(fill="x", pady=(8, 0))

        def create():
            sel = t.tree.selection()
            if not sel:
                return
            idx, name = rows[int(sel[0])][0], rows[int(sel[0])][1]
            try:
                text, summary = hc.lower_symmetry_cif(st, idx, drop_h=hmode.get(), s=s)
                path = hc.lower_symmetry_path(st.source or os.path.join(os.getcwd(), st.name + ".cif"), name)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                new = hc.load_structures(path)
            except Exception as e:  # noqa: BLE001
                messagebox.showerror(APP_NAME, "Could not create the lower-symmetry structure:\n%s" % e)
                return
            win.destroy()
            self.structures.extend(new)
            names = [x.title for x in self.structures]
            self.struct_box.configure(values=names, state="readonly")
            self.struct_var.set(new[0].title)
            self.on_structure_change()
            self.status_var.set("Created %s - press 'Run analysis'.  %s" % (os.path.basename(path), " ".join(summary)))
        ttk.Button(bb, text="Create and open", command=create).pack(side="left")
        ttk.Button(bb, text="Cancel", command=win.destroy).pack(side="right")
        if self.theme.get() == "dark":
            self.apply_theme()

    def current_structure(self):
        for s in self.structures:
            if s.title == self.struct_var.get():
                return s
        return self.structures[0] if self.structures else None

    def _set_info(self, st):
        n_h = sum(1 for a in st.atoms if a.el == "H")
        self.info_var.set("%s | %s | a=%.4f b=%.4f c=%.4f  α=%.3f β=%.3f γ=%.3f | %d sym. ops | %d sites (%d H) | "
                          "formula: %s" % (st.name, st.spacegroup, st.a, st.b, st.c, st.alpha, st.beta, st.gamma,
                                           len(st.symops), len(st.atoms), n_h, st.formula_text))

    def on_structure_change(self):
        st = self.current_structure()
        if st is None:
            return
        if self.st is not None and self.st is not st:     # keep the analysis of the structure we leave
            self._snapshot_current()
        self._set_info(st)
        self._clear_pending()
        snap = self._cache.get(st.title)
        if snap is not None and snap["payload"][0] is st:
            try:
                self._restore_snapshot(snap)
                self.status_var.set("Showing the earlier analysis of this structure (Run analysis to redo it). "
                                    + (snap.get("status") or ""))
                return
            except Exception:  # noqa: BLE001
                self._cache.pop(st.title, None)
        self.clear_all()
        self.status_var.set("Ready - click 'Run analysis'.")

    def clear_all(self):
        for t in (self.sites, self.bonds, self.cands, self.opts, self.sch, self.sch_bv, self.loc, self.env,
                  self.angles, self.fin, self.fin_bv, self.cmpt, self.cmp_bv, self.grp, self.roles):
            t.clear()
        self.dis_note.set("")
        self.hf_btn.configure(state="disabled")
        self.cmp = []
        self.subs = None
        self.sub_rows = []
        self.subt.clear()
        self.sub_bv.clear()
        self.sub_status.set("")
        self.sub_flag.configure(text="", background=self.root.cget("background"))
        self.loc_all = []
        self.ref_btn.configure(state="disabled")
        self.cmp_flag.configure(text="", background=self.root.cget("background"))
        self.ref_status.set("")
        self._set_notes("")
        self.inf_note.set("")
        self.loc_note.set("")
        self.sch_info.set("Run the analysis to rank H-bond schemes.")
        self.sch_flag.configure(text="", background=self.root.cget("background"))
        self.fin_flag.configure(text="", background=self.root.cget("background"))
        self.fin_hc.configure(text="")
        self.st, self.bv, self.res, self.donors, self.loc_h, self.located = None, None, None, [], [], ([], [])
        self.fin_h, self.fin_pass, self.fin_info, self.checked, self.fin_loc = {}, set(), {}, set(), {}
        self.fin_parent, self._fin_n = {}, 0
        self.fin_rowkey, self.fin_by_key, self.fin_opts_of = {}, {}, {}
        self._fin_loaded_cmp, self._fin_src = None, ""
        if hasattr(self, "fin_source"):
            self.fin_source.set("")
        self.table_data = None
        self._set_table("")
        self.det_data = None
        self.det_text.configure(state="normal")
        self.det_text.delete("1.0", "end")
        self.det_text.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.cif_btn.configure(state="disabled")
        self.env_box.configure(state="disabled", values=[])

    # ================================================================= running
    def run(self, keep_selection=False):
        st = self.current_structure()
        if st is None or self.params is None:
            return
        try:
            s = self.settings()
            la = self.located_args(s)
        except ValueError as e:
            messagebox.showerror("Invalid settings", str(e))
            return
        if self._pending and self.st is not None and st is self.st:
            keep_selection = True              # applying hand-made changes: keep the user's selections
        self._clear_pending()
        sel = self.sites.tree.selection() if keep_selection else ()
        checked_keys = {self.fin_rowkey.get(i) for i in self.checked} if keep_selection else set()
        donor_set = {d["i"] for d in self.res.donors} if (keep_selection and self.res) else None
        self.clear_all()
        self.run_btn.configure(state="disabled")
        self.status_var.set("Running...")
        self.root.config(cursor="watch")
        ov = self.overrides.setdefault(st.title, {})
        cov = dict(self.class_overrides.setdefault(st.title, {}))
        rov = dict(self.role_overrides.setdefault(st.title, {}))
        vov = dict(self.valence_overrides.setdefault(st.title, {}))

        def progress(done, total):
            self._progress = ("Searching schemes... %d / %d" % (done, total)) if total else \
                "Evaluating the best schemes (GRG D–H optimisation)..."

        def work():
            try:
                self._progress = "Bond valence..."
                if getattr(st, "sub_spec", None) is not None:     # a subscheme: its own network only
                    bv, loc, donors, res, loc_all = hc.subscheme_analysis(st, self.params, s, ov, cov, vov, rov)
                else:                                             # (charge-balanced rounds when ticked)
                    bv, loc, donors, res, loc_all = hc.analyse_and_search(st, self.params, s, ov, cov, vov, rov,
                                                                          progress)
                located = hc.located_hbonds(st, env_radius=s.env_radius, **la)
                self._result = ("ok", (st, s, bv, loc, donors, res, located, loc_all))
            except Exception as e:  # noqa: BLE001
                self._result = ("error", "%s\n\n%s" % (e, traceback.format_exc(limit=4)))

        self._result = None
        self._keep = (sel, checked_keys, donor_set)
        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self.root.after(100, self._poll)

    def _poll(self):
        if self._worker is not None and self._worker.is_alive():
            self.status_var.set(self._progress)
            self.root.after(150, self._poll)
            return
        self.root.config(cursor="")
        self.run_btn.configure(state="normal")
        if self._result is None:
            return
        status, payload = self._result
        self._result = None
        if status == "error":
            self.status_var.set("Analysis failed.")
            messagebox.showerror("Analysis failed", payload)
            return
        self._apply_payload(payload)
        sel, checked_keys, donor_set = self._keep
        if sel:
            self.sites.tree.selection_set(sel)
            self.sites.tree.see(sel[0])
        # keep the user's Finalize selection only if the same donors still need H
        if checked_keys and donor_set == {d["i"] for d in self.res.donors}:
            self.checked = {iid for iid in self.fin_h if self.fin_rowkey.get(iid) in checked_keys}
            self.refresh_finalize()
        self.status_var.set("Done: %d sites, %d located H%s, %d donor(s) with missing H, %d scheme(s) kept of %s (%s)."
                            % (len(self.st.atoms), sum(1 for h in self.loc_all if h.primary),
                               " (ignored)" if self.s.ignore_located and self.loc_all else "",
                               len(self.res.donors), len(self.res.schemes),
                               "{:,}".format(self.res.n_total), self.res.method))
        if self.res.hard_fail:
            self.status_var.set(self.status_var.get() + "  ⚠ Hard fail: %s - see the Disorder tab or press "
                                "'Re-evaluate without hard-fail donors'."
                                % ", ".join(self.st.atoms[i].label for i in self.res.hard_fail))
        if self._pending_restore is not None:
            data, self._pending_restore = self._pending_restore, None
            self._restore_after_run(data)

    def _apply_payload(self, payload):
        """Show a finished analysis (st, settings, bv, located H used, donors, schemes, located table, all
        located H) - after a run, or restored from a snapshot / saved analysis."""
        self.st, self.s, self.bv, self.loc_h, self.donors, self.res, self.located, self.loc_all = payload
        self.populate_all()
        self.ref_btn.configure(state="normal" if self.res.schemes else "disabled")
        self.export_btn.configure(state="normal")
        self.run_btn.configure(state="normal" if self.params else "disabled")

    def _snapshot(self):
        """Everything needed to show the current analysis again without re-running it."""
        if self.st is None or self.res is None:
            return None
        try:
            fin = [h for _i, h in sorted(self._fin_effective().items())]
        except Exception:  # noqa: BLE001
            fin = []
        return {"payload": (self.st, self.s, self.bv, self.loc_h, self.donors, self.res, self.located, self.loc_all),
                "cmp": [{"name": e["name"], "hs": e["hs"], "ev": e["ev"], "info": e["info"]} for e in self.cmp],
                "fin": fin, "fin_src": getattr(self, "_fin_src", ""), "subs": self.subs,
                "sub_src": self.sub_src.get(), "scheme_sel": self.selected_scheme(),
                "status": self.status_var.get()}

    def _snapshot_current(self):
        snap = self._snapshot()
        if snap is not None:
            self._cache[self.st.title] = snap

    def _restore_snapshot(self, snap):
        self.clear_all()
        self._apply_payload(snap["payload"])
        for e in snap.get("cmp", []):
            self._add_cmp(e["name"], e["hs"], e["ev"], e["info"],
                          tag=("cif",) if e["name"].startswith("H in CIF") else ())
        if snap.get("fin"):
            self.load_hatoms(snap["fin"], snap.get("fin_src") or "saved selection")
            if snap.get("fin_src"):
                self._fin_src = snap["fin_src"]
                self._set_fin_source()
        self.subs = snap.get("subs")
        if snap.get("sub_src"):
            self.sub_src.set(snap["sub_src"])
        self._fill_subs()
        r = snap.get("scheme_sel")
        if r is not None and self.res.schemes and r < len(self.res.schemes):
            self.sch.tree.selection_set(str(r))
            self.sch.tree.see(str(r))
        self.status_var.set(snap.get("status") or "Restored.")

    def rerun_if_done(self):
        if self.st is not None and (self._worker is None or not self._worker.is_alive()):
            self.run(keep_selection=True)

    # ============================================================== populating
    def populate_all(self):
        self.populate_bv()
        self.populate_disorder()
        self.populate_inferred()
        self.populate_schemes()
        self.populate_located()
        self.build_finalize()
        labels = ["%d: %s" % (i + 1, a.label) for i, a in enumerate(self.st.atoms)]
        self.env_box.configure(values=labels, state="readonly")
        if not self.env_site.get() or self.env_site.get() not in labels:
            self.env_site.set(labels[0])
        self.show_env()
        self.refresh_table()
        self.refresh_details()

    def populate_bv(self):
        st, bv = self.st, self.bv
        self.sites.clear()
        for i, a in enumerate(st.atoms):
            role = bv.roles[i]
            dev = None if bv.ideal[i] is None else bv.bvs[i] - bv.ideal[i]
            cls = bv.cls[i]
            tag = ()
            if cls.startswith("water"):
                tag = ("water",)
            elif cls in (hc.CLASS_OH, hc.CLASS_POH):
                tag = ("hydroxyl",)
            elif cls in hc.DONOR_CLASSES:          # NH4+, NH3, NH2, H3O+
                tag = ("water",)
            elif cls == hc.CLASS_PROBABLE:
                tag = ("probable",)
            elif cls == hc.CLASS_POSSIBLE:
                tag = ("possible",)
            if i in bv.cls_manual or i in getattr(bv, "val_manual", ()):
                tag = tag + ("manual",)
                cls = "%s (set; BV: %s)" % (cls, bv.cls_auto[i] or "-")
            if bv.hrole.get(i):
                cls = "%s [%s]" % (cls, ROLE_TEXT[bv.hrole[i]])
            noh = bv.bvs_noH[i]
            gr = i in getattr(bv, "nh4_gr", ())
            occ_txt = _fmt(a.occ, 3)
            if abs(getattr(a, "occ_cif", a.occ) - a.occ) > 1e-6:
                occ_txt += " (CIF %.3f)" % a.occ_cif
            self.sites.tree.insert("", "end", iid=str(i), tags=tag, values=(
                a.label, a.el, role, "1+ (NH4+)" if gr else _val_text(bv.valence.get(i), role), occ_txt,
                _fmt(bv.bvs[i]), _fmt(noh) if abs(noh - bv.bvs[i]) > 1e-9 else "", _fmt(dev), bv.nsum[i],
                _fmt(bv.weak[i]) if bv.weak[i] else "", cls,
                "NH4+ as a whole: García-Rodríguez et al. 2000" if gr else
                (bv.how.get(i, "") if role == "cation" else "")))
        lines = list(bv.notes)
        if bv.vanadium:
            lines.append("V sites (Cooper et al. 2019):  " + ";  ".join(
                "%s Vc=%.3f → Vp=%.3f (≈%.0f%% V4+)" % (d["label"], d["vc"], d["vp"], 100 * d["frac_v4"])
                for d in bv.vanadium))
        lines.append("Bonds with %.2f ≤ s < %.2f vu are listed but not summed; contacts < %.2f Å to non-H atoms "
                     "and cation–cation / anion–anion pairs are ignored. Neighbour contributions are weighted by the "
                     "neighbour's occupancy, never by the site's own. D/A (donor / acceptor) classes (O: water < "
                     "%.2f ≤ hydroxyl < %.2f vu; N: NH4+ / NH3 / NH2) use the sum without H." % (self.s.list_cutoff, self.s.sum_cutoff, self.s.min_bond_length,
                                                 self.s.water_max, self.s.hydroxyl_max))
        n_ov = len(self.overrides.get(st.title, {}))
        if n_ov:
            lines.append("%d manual bond override(s) active for this structure." % n_ov)
        self._set_notes("\n".join(lines))
        self.bonds.clear()
        self.update_formula_bar()

    def update_charge_label(self):
        """'Formula charge: +0.00' - green when the model is neutral, orange-red otherwise; the tooltip-like
        breakdown follows in the label."""
        st, bv = self.st, self.bv
        if st is None or bv is None:
            self.charge_lbl.configure(text="Formula charge: -", background=self.root.cget("background"),
                                      foreground=getattr(self, "_plain_fg", "#1f2937"))
            return
        q = hc.formula_charge(st, bv)
        t = q["total"]
        ok = abs(t) < 0.05
        self.charge_lbl.configure(
            text="Formula charge: %s   (cations %+.2f, H %+.2f, anions %+.2f)" % (
                "0" if abs(t) < 0.005 else "%+.2f" % t, q["cations"], q["H"], q["anions"]),
            background="#bbf7d0" if ok else ("#fde68a" if abs(t) < 0.5 else "#fdba74"), foreground="black")

    def update_formula_bar(self):
        """Given formula vs the formula modelled from the sites (occupancy x multiplicity / Z) with H from the
        D/A classes. Green = same composition; blue = same apart from H, which the given formula lacks;
        orange-red = different composition or H count (check the classes)."""
        st, bv = self.st, self.bv
        self.update_charge_label()
        if st is None or bv is None:
            self.formula_lbl.configure(text="", background=self.root.cget("background"))
            return
        cmp_ = hc.formula_comparison(st, bv)
        if cmp_ is None:
            self.formula_lbl.configure(text="Formula: %s  (no _chemical_formula_sum or Z - nothing to compare)" %
                                       hc.formula_text(hc.model_formula(st, bv)[0]),
                                       background=self.root.cget("background"), foreground=getattr(self, "_plain_fg", "#1f2937"))
            return
        given, model, status, diffs = cmp_
        colour = {"agree": "#bbf7d0", "noH": "#bfdbfe", "differ": "#fdba74"}[status]
        meaning = {"agree": "agree",
                   "noH": "agree apart from H - the given formula has no H (H not located / inferred elsewhere)",
                   "differ": "DIFFER" + ((" (%s)" % ", ".join(diffs)) if diffs else "") +
                             " - check the D/A classes (H) or the occupancies"}[status]
        self.formula_lbl.configure(text="Given formula:  %s        Modelled (sites + H from classes):  %s        → %s" % (
            hc.formula_text(given, order=list(given)), hc.formula_text(model, order=list(given)), meaning),
            background=colour, foreground="black")

    def class_menu(self, event):
        """Right-click on a site: choose its class by hand (or go back to the BV-based class)."""
        if self.bv is None:
            return
        row = self.sites.tree.identify_row(event.y)
        if not row:
            return
        # several sites selected (Ctrl / Shift + click) and the click is on one of them: act on all of them
        sel = list(self.sites.tree.selection())
        if row not in sel:
            self.sites.tree.selection_set(row)
            sel = [row]
        i = int(row)
        st, bv = self.st, self.bv
        if bv.roles[i] == "cation" and st.atoms[i].el != "H":
            cats = [int(r) for r in sel if bv.roles[int(r)] == "cation" and st.atoms[int(r)].el != "H"]
            self.valence_menu(cats, event)
            return
        if bv.roles[i] != "anion":
            return
        ans = [int(r) for r in sel if bv.roles[int(r)] == "anion"]
        names = self._names(ans)
        cov = self.class_overrides.setdefault(st.title, {})

        def mark(cond):
            return "● " if cond else "   "

        def all_set(c):                          # every selected site hand-set to class c
            return all(k in cov and bv.cls[k] == c for k in ans)
        m = tk.Menu(self.root, tearoff=0)
        m.add_command(label="Class of %s:" % names, state="disabled")
        rare = (hc.CLASS_H3O, hc.CLASS_NH4, hc.CLASS_NH3, hc.CLASS_NH2)   # uncommon: in an "Other" submenu
        for c in hc.ALL_CLASSES:
            if c in rare:
                continue
            m.add_command(label=mark(all_set(c)) + c, command=lambda c=c: self.set_class(ans, c))
        sub = tk.Menu(m, tearoff=0)
        for c in rare:
            sub.add_command(label=mark(all_set(c)) + c, command=lambda c=c: self.set_class(ans, c))
        m.add_cascade(label=mark(any(all_set(c) for c in rare)) + "other", menu=sub)
        m.add_separator()
        auto = (bv.cls_auto[i] or "none") if len(ans) == 1 else "each from its BV sum"
        m.add_command(label=mark(all(k not in cov for k in ans)) + "automatic (from BV: %s)" % auto,
                      command=lambda: self.set_class(ans, None))
        m.add_separator()
        m.add_command(label="H-bond role of %s:" % names, state="disabled")
        for r in ("normal", "acceptor", "exclude"):
            m.add_command(label=mark(all(bv.hrole.get(k, "normal") == r for k in ans)) + ROLE_TEXT[r],
                          command=lambda r=r: self._set_roles(ans, r, "set by hand"))
        try:
            m.tk_popup(event.x_root, event.y_root)
        finally:
            m.grab_release()

    def _names(self, sites):
        labs = [self.st.atoms[k].label for k in sites]
        return labs[0] if len(labs) == 1 else ("%s (%d sites)" % (", ".join(labs[:6]) + (" ..." if len(labs) > 6
                                                                                         else ""), len(labs)))

    def _states(self, i):
        st, bv = self.st, self.bv
        an_els = [st.atoms[k].el for k in range(len(st.atoms)) if bv.roles[k] == "anion"]
        main_an = "O" if "O" in an_els else (max(set(an_els), key=an_els.count) if an_els else "O")
        return sorted(self.params.states(st.atoms[i].el, main_an))

    def valence_menu(self, sites, event):
        """Right-click on one or several cations: oxidation state - one state, a mixture with the best BV
        fit, an average site charge typed in, or automatic. Applied at the next Run analysis. With several
        sites, a state is applied to every selected site that has it."""
        if isinstance(sites, int):
            sites = [sites]
        st, bv = self.st, self.bv
        vov = self.valence_overrides.setdefault(st.title, {})
        states_of = {k: self._states(k) for k in sites}
        els = sorted({st.atoms[k].el for k in sites})
        all_states = sorted({v for lst in states_of.values() for v in lst})

        def mark(spec):
            return "● " if all(vov.get(k) == spec for k in sites) else "   "
        m = tk.Menu(self.root, tearoff=0)
        now = _val_text(bv.valence.get(sites[0]), "cation") if len(sites) == 1 else "-"
        m.add_command(label="Valence of %s%s:" % (self._names(sites), (" (now %s)" % now) if len(sites) == 1 else ""),
                      state="disabled")
        for v in all_states:
            have = [k for k in sites if v in states_of[k]]
            name = ("%s%d+" % (els[0], v)) if len(els) == 1 else \
                ("%d+  (%s)" % (v, ", ".join(sorted({st.atoms[k].el for k in have}))))
            m.add_command(label=mark({"v": v}) + name +
                          ("" if len(have) == len(sites) else "   - %d of %d sites" % (len(have), len(sites))),
                          command=lambda v=v, have=have: self.set_valence(have, {"v": v}))
        multi = [k for k in sites if len(states_of[k]) >= 2]
        if multi:
            sub = tk.Menu(m, tearoff=0)
            sub.add_command(label=mark({"mix": None}) + (
                "best BV fit (Cooper et al. 2019 for V)" if els == ["V"] else "best BV fit (any two neighbouring states)"),
                command=lambda: self.set_valence(multi, {"mix": None}))
            for v1, v2 in zip(all_states, all_states[1:]):
                have = [k for k in multi if v1 in states_of[k] and v2 in states_of[k]]
                if have:
                    sub.add_command(label=mark({"mix": [v1, v2]}) + "%d+ / %d+  best BV fit" % (v1, v2),
                                    command=lambda v1=v1, v2=v2, have=have: self.set_valence(have, {"mix": [v1, v2]}))
            m.add_cascade(label=("● " if all(vov.get(k) and "mix" in vov[k] for k in sites) else "   ") + "mixed",
                          menu=sub)
        m.add_command(label=("● " if all(vov.get(k) and "avg" in vov[k] for k in sites) else "   ") +
                      "average site charge (type it in)...",
                      command=lambda: self._ask_avg_valence(sites, all_states))
        m.add_separator()
        auto = bv.how.get(sites[0], "") if len(sites) == 1 else "each from the formula / BV fit"
        m.add_command(label=("● " if all(k not in vov for k in sites) else "   ") + "automatic (%s)" % auto,
                      command=lambda: self.set_valence(sites, None))
        try:
            m.tk_popup(event.x_root, event.y_root)
        finally:
            m.grab_release()

    def _ask_avg_valence(self, sites, states):
        from tkinter import simpledialog
        if isinstance(sites, int):
            sites = [sites]
        el = "/".join(sorted({self.st.atoms[k].el for k in sites}))
        q = simpledialog.askfloat(APP_NAME, "Average charge of %s (e.g. 3.5 = half %s3+ / half %s4+).\n"
                                            "Available states: %s" % (self._names(sites), el, el,
                                                                      ", ".join("%d+" % v for v in states)),
                                  parent=self.root, minvalue=min(states) if states else 0.0,
                                  maxvalue=max(states) if states else 10.0)
        if q is not None:
            self.set_valence(sites, {"avg": round(q, 4)})

    def set_valence(self, sites, spec):
        """Hand-set the valence of one site or a list of sites (one pending change)."""
        if isinstance(sites, int):
            sites = [sites]
        if not sites:
            return
        vov = self.valence_overrides.setdefault(self.st.title, {})
        if spec is None:
            shown = "automatic"
        else:
            shown = ("%d+" % spec["v"]) if "v" in spec else ("average %+.2f" % spec["avg"]) if "avg" in spec else \
                ("mixed %d+/%d+ (best fit)" % tuple(spec["mix"]) if spec.get("mix") else "mixed (best fit)")
        for i in sites:
            if spec is None:
                vov.pop(i, None)
            else:
                vov[i] = dict(spec)
            iid = str(i)
            if self.sites.tree.exists(iid):
                vals = list(self.sites.tree.item(iid)["values"])
                vals[3] = "%s → %s (pending)" % (str(vals[3]).split(" →")[0], shown)
                self.sites.tree.item(iid, values=vals, tags=tuple(self.sites.tree.item(iid)["tags"]) + ("manual",))
        self._mark_pending("valence of %s → %s" % (self._names(sites), shown))

    def set_class(self, sites, cls):
        """Hand-set the D/A class of one site or a list of sites (one pending change)."""
        if isinstance(sites, int):
            sites = [sites]
        if not sites:
            return
        cov = self.class_overrides.setdefault(self.st.title, {})
        for i in sites:
            if cls is None or cls == self.bv.cls_auto[i]:
                cov.pop(i, None)
            else:
                cov[i] = cls
            self._show_pending_site(i, cls=cls or "automatic (%s)" % (self.bv.cls_auto[i] or "none"))
        self._mark_pending("%s → %s" % (self._names(sites), cls or "automatic"))

    def _set_notes(self, text):
        self.bv_notes.configure(state="normal")
        self.bv_notes.delete("1.0", "end")
        self.bv_notes.insert("1.0", text)
        self.bv_notes.configure(state="disabled")

    def _bond_status(self, i, bd):
        manual = (i, bd.nb.key) in self.overrides.get(self.st.title, {})
        if bd.included:
            txt = "summed" + (" (H)" if bd.to_h else "")
        elif bd.too_short:
            txt = "< %.2f Å - ignored" % self.s.min_bond_length
        elif getattr(bd, "blocked", False):
            txt = getattr(bd, "why", "") or "NH4+ – cation: not a bond"
        else:
            txt = "listed, not summed"
        return txt + (" (manual)" if manual else ""), manual

    def _bond_default(self, bd):
        return (not bd.too_short) and bd.s >= self.s.sum_cutoff and not getattr(bd, "blocked", False)

    def _toggle_bond_pair(self, i, bd, new_state):
        """Include / exclude a bond in BOTH atoms' sums (queued for the next Run analysis)."""
        ov = self.overrides.setdefault(self.st.title, {})
        j = bd.nb.j
        back = next((b for b in self.bv.bonds[j] if b.nb.j == i and abs(b.nb.d - bd.nb.d) < 1e-3), None)
        for site, b in ((i, bd), (j, back)):
            if b is None:
                continue
            key = (site, b.nb.key)
            if new_state == self._bond_default(b):
                ov.pop(key, None)
            else:
                ov[key] = new_state
        self._mark_pending("%s–%s %.3f Å %s" % (self.st.atoms[i].label, self.st.atoms[j].label, bd.nb.d,
                                                "included" if new_state else "excluded"))

    def show_bonds(self):
        sel = self.sites.tree.selection()
        self.bonds.clear()
        if not sel or self.bv is None:
            return
        i = int(sel[0])
        st, bv = self.st, self.bv
        self.bond_title.set("Bonds of %s (%s, BVS %.3f vu, w/o H %.3f) - double-click a bond to include/exclude it."
                            % (st.atoms[i].label, _val_text(bv.valence.get(i), bv.roles[i]), bv.bvs[i],
                               bv.bvs_noH[i]))
        for k, bd in enumerate(bv.bonds[i]):
            nb = bd.nb
            status, manual = self._bond_status(i, bd)
            tags = ("manual",) if manual else (() if bd.included else ("off",))
            self.bonds.tree.insert("", "end", iid=str(k), tags=tags, values=(
                st.atoms[nb.j].label, st.atoms[nb.j].el, _fmt(nb.d, 4), _fmt(bd.s, 4), _fmt(bd.occ, 3),
                _fmt(bd.weighted, 4), status, st.symop_text(nb.op, nb.shift), st.symop_code(nb.op, nb.shift),
                "%s (%s)" % (bd.param, bd.param.source)))

    def toggle_bond(self, event):
        row = self.bonds.tree.identify_row(event.y)
        sel = self.sites.tree.selection()
        if not row or not sel or self.bv is None:
            return
        i, k = int(sel[0]), int(row)
        bd = self.bv.bonds[i][k]
        key = (i, bd.nb.key)
        cur = self.overrides.get(self.st.title, {}).get(key, bd.included)
        new_state = not cur
        vals = list(self.bonds.tree.item(row)["values"])
        vals[6] = ("summed" if new_state else "not summed") + " (pending)"
        self.bonds.tree.item(row, values=vals, tags=("manual",))
        self._toggle_bond_pair(i, bd, new_state)

    def populate_inferred(self):
        st = self.st
        self.cands.clear()
        self.opts.clear()
        if not self.donors:
            self.inf_note.set("No O site is classed water/hydroxyl and none carries located H.")
            return
        for d in self.donors:
            cat_txt = ", ".join("%s %.3f Å" % (st.atoms[c.j].label, c.d) for c in d["cations"]) or "no cation bonds"
            head = "%s - %s, BVS w/o H %.3f - %d H (%d located, %d to infer) - bonded to: %s" % (
                d["label"], d["cls"], d["bvs"], d["n_h"], len(d["located"]), d["n_missing"], cat_txt)
            pid = self.cands.tree.insert("", "end", text=head, open=True, tags=("donor",))
            for h in d["located"]:
                c = h.cand
                self.cands.tree.insert(pid, "end", text="located %s" % h.located, tags=("located",), values=(
                    st.atoms[c.acceptor].label if c else "none", _fmt(c.nb.d) if c else "", "", "", "", "", "",
                    _fmt(h.s_grg) if c else "", _fmt(h.s_fi) if c else "",
                    "H···A %.3f Å, D–H···A %.1f°" % (h.d_ha, h.angle) if c else "no acceptor",
                    st.symop_text(c.nb.op, c.nb.shift) if c else "", st.symop_code(c.nb.op, c.nb.shift) if c else ""))
            for c in d["cands"]:
                nb = c.nb
                angs = ", ".join("%s %.1f" % (st.atoms[cn.j].label, ang) for cn, ang in c.cation_angles)
                self.cands.tree.insert(pid, "end", text=hc.ordinal(c.ordinal), tags=() if c.passes else ("fail",),
                                       values=(st.atoms[nb.j].label, _fmt(nb.d), _fmt(c.min_adc, 1), angs,
                                               "ok" if c.cone_ok else "IN", "ok" if c.adc_ok else "no",
                                               _fmt(c.adc_pen, 1) if c.adc_pen else "", _fmt(c.s_grg), _fmt(c.s_fi),
                                               "pass" if c.passes else c.reason,
                                               st.symop_text(nb.op, nb.shift), st.symop_code(nb.op, nb.shift)))
            if not d["n_missing"]:
                continue
            imp = d.get("n_impossible", 0)
            oid = self.opts.tree.insert("", "end", open=False, tags=("donor",),
                                        text="%s - %s (%d option(s) for %d missing H%s)" % (
                                            d["label"], d["cls"], len(d["options"]), d["n_missing"],
                                            ("; %d impossible %s" % (imp, "shown" if self.s.show_severe
                                                                     else "removed")) if imp else ""))
            for o in d["options"]:
                hb = " + ".join("%s %s" % (hc.ordinal(c.ordinal), st.atoms[c.acceptor].label) for c in o.cands)
                status = "ok" if o.passes else (
                    "incomplete" if o.incomplete else
                    ("A1–D–A2 > %.0f°" % self.s.pair_max_hard) if (o.angle or 0) > self.s.pair_max_hard
                    else "A1–D–A2 < %.0f°" % self.s.pair_hard_angle)
                if getattr(o, "impossible", ""):
                    status = "IMPOSSIBLE: " + o.impossible
                self.opts.tree.insert(oid, "end", text="#%d" % o.number, tags=() if o.passes else ("bad",), values=(
                    hb, _fmt(o.angle, 1), _fmt(o.hdh, 1), ", ".join("%.3f" % h.d_ha for h in o.hs),
                    ", ".join("%.1f" % h.angle for h in o.hs), _fmt(o.angle_pen, 0) if o.angle_pen else "",
                    _fmt(o.adc_pen, 1) if o.adc_pen else "", _fmt(o.donorF), status))
        notes = ["*s GRG for a candidate assumes H on the D···A line (H···A = D···A − %.3f Å). The F&I column of the "
                 "options is the BV sum w/o H minus all H-bonds this donor donates (located + option); the H-bonds it "
                 "accepts are added on the Schemes tab, where water aims at 0 ± %.2f and hydroxyl at 1 ± %.2f vu."
                 % (self.s.dh_length, self.s.donor_band, self.s.donor_band)]
        self.inf_note.set("  ".join(notes))

    def consolidate_now(self):
        if self.res is None or not self.res.schemes:
            return
        if getattr(self.res, "consolidated", False):
            self.status_var.set("Equivalent schemes are already consolidated.")
            return
        n = hc.consolidate_schemes(self.st, self.res)
        self.populate_schemes()
        self.status_var.set("Consolidated: %d scheme(s) with the same acceptor atoms merged; %d remain (×n in the "
                            "Rank column = number merged, 'var.' = some used other symmetry copies of the acceptors "
                            "with another score - the best is kept)." % (n, len(self.res.schemes)))

    def populate_schemes(self):
        st, res = self.st, self.res
        if self.consol_var.get() and res is not None and res.schemes and not getattr(res, "consolidated", False):
            hc.consolidate_schemes(st, res)
        self.sch.clear()
        hidden = ""
        if not self.s.show_severe:
            hidden = ("  %s scheme(s) with an inferred H···H < %.2f Å are hidden (tick 'Show schemes with H···H "
                      "< %.2f Å' to see them)." % ("{:,}".format(res.n_hidden) if res.n_hidden is not None else "Any",
                                                  self.s.hh_severe, self.s.hh_severe))
        if not res.schemes:
            msg = "No viable scheme found."
            if res.hard_fail:
                msg += " Hard fails: %s. Use 'Re-evaluate without hard-fail donors', set disordered water/OH to " \
                       "'acceptors only' / 'excluded', or tick 'Show impossible configurations'." % \
                       ", ".join(st.atoms[i].label for i in res.hard_fail)
            self.sch_info.set(msg + hidden)
            return
        skipped = ", ".join(d["label"] for d in res.skipped)
        fixed = ", ".join(d["label"] for d in res.fixed)
        loc_txt = ("Located H (%d) on %s are part of every scheme." % (len(res.located), fixed or "-")
                   if not self.s.ignore_located else "Located H in the CIF are IGNORED (every H is predicted).")
        self.sch_info.set(
            "%s combinations (%s); top %d shown, ranked by %s. %s Score = BV score + angle penalties + "
            "H···H penalties.%s%s"
            % ("{:,}".format(res.n_total), res.method, len(res.schemes),
               {"both": "the mean of F&I and GRG", "grg": "GRG", "fi": "F&I"}[self.s.rank_by], loc_txt,
               ("  No passing candidates for: %s." % skipped) if skipped else "", hidden))
        if getattr(res, "consolidated", False) and any(sc.get("equiv", 1) > 1 for sc in res.schemes):
            self.sch_info.set(self.sch_info.get() + "  Schemes with the same acceptor atoms are merged: (×n) = number "
                              "merged, 'var.' = some used other symmetry copies of the acceptors (another geometry and "
                              "score) - the best is shown.")
        hf = self.bv.h_formula
        best_f = res.schemes[0]["bvF"]
        n_same = 0
        for r, sc in enumerate(res.schemes, 1):
            tag = ("red",) if sc["red"] else (("mod",) if sc["mod"] else ())
            if hf is not None and abs(sc["h_cell"] - hf) > 0.05:
                tag = tag + ("hbad",)
            same = abs(sc["bvF"] - best_f) <= 0.05
            n_same += same
            eq = sc.get("equiv", 1)
            rank = ("%d (×%d%s)" % (r, eq, " var." if sc.get("variants") else "")) if eq > 1 else r
            self.sch.tree.insert("", "end", iid=str(r - 1), tags=tag, values=(
                rank, "≡" if same else "", _fmt(sc["total"], 2), _fmt(sc["bvF"], 2), _fmt(sc["bvG"], 2),
                _fmt(sc["geom"], 1), _fmt(sc["hh"], 0), int(sc["red"]), int(sc["mod"]), int(sc.get("sev", 0)),
                int(sc.get("cone", 0)), int(sc.get("dis", 0)), _fmt(sc["maxF"]),
                _fmt(sc["maxG"]), _fmt(sc["h_cell"], 2), hc.scheme_summary(st, res, sc)))
        if n_same > 1:
            self.sch_info.set(self.sch_info.get() + "  ≡ %d schemes have the same F&I balance as #1 (e.g. H-bonds "
                              "running the other way round a ring of waters): F&I cannot tell them apart and only "
                              "the angle / H···H terms rank them - compare them (GRG, refinement) before choosing."
                              % n_same)
        if hf is not None:
            self.sch_info.set(self.sch_info.get() + "  H per cell from the formula: %.2f%s." % (
                hf, "" if abs(res.schemes[0]["h_cell"] - hf) <= 0.05 else
                " - ⚠ the top scheme has %.2f (check classes / missing H)" % res.schemes[0]["h_cell"]))
        self.sch.tree.selection_set("0")
        self.show_scheme()

    def _flag_box(self, widget, contacts, labels, hs=None, st=None):
        st = st if st is not None else self.st
        geo = [c for c in contacts if hc.contact_is_geometry(st, c)]
        dis = [c for c in contacts if not hc.contact_is_geometry(st, c)]
        red = [c for c in geo if c[2] < self.s.hh_red]
        mod = [c for c in geo if self.s.hh_red <= c[2] < self.s.hh_mod]
        cones = hc.cone_violations(st, hs, self.s) if hs else []

        def name(h):
            lab = (labels or {}).get(id(h))
            if lab:
                return lab + ("" if h.located else "*")
            return "H(%s→%s)" % (self.st.atoms[h.donor].label,
                                 self.st.atoms[h.cand.acceptor].label if h.cand else "?")

        def pair(c):
            a, b, d, op, shift = c
            return "%s–%s [%s] %.2f Å" % (name(a), name(b), self.st.symop_code(op, shift), d)

        extra = ""
        if cones:
            extra += "   |   ⚠ H-bond inside a water's acceptor cone: " + "; ".join(
                "%s→%s (%.0f° < %.0f°)" % (name(h), self.st.atoms[w].label, a, half) for h, w, a, half in cones)
        if dis:
            extra += "   |   disorder-only donor conflicts (donor occupancies sum ≤ 1, not geometry): " + "; ".join(
                pair(c) for c in dis)
        if red or cones:
            sev = sum(1 for c in red if c[2] < self.s.hh_severe)
            txt = ("⚠ RED FLAG - H···H < %.2f Å between different donors%s: " % (
                self.s.hh_red, (" (%d below %.2f Å)" % (sev, self.s.hh_severe)) if sev else "") + "; ".join(
                pair(c) for c in red)) if red else "⚠ RED FLAG"
            if mod:
                txt += "   |   sliding scale (%.2f–%.2f Å): " % (self.s.hh_red, self.s.hh_mod) + "; ".join(
                    pair(c) for c in mod)
            widget.configure(text=txt + extra, background="#dc2626", foreground="white")
        elif mod:
            widget.configure(text="H···H %.2f–%.2f Å (sliding-scale penalty): " % (self.s.hh_red, self.s.hh_mod)
                             + "; ".join(pair(c) for c in mod) + extra, background="#fdba74", foreground="black")
        else:
            widget.configure(text="✓ No H···H contacts below %.2f Å and no acceptor-cone violations%s." % (
                self.s.hh_mod, extra), background="#bbf7d0", foreground="black")

    def selected_scheme(self):
        sel = self.sch.tree.selection()
        if not sel or self.res is None or not self.res.schemes:
            return None
        return int(sel[0])

    def show_scheme(self):
        r = self.selected_scheme()
        if r is None:
            return
        hs = hc.scheme_hatoms(self.res, self.res.schemes[r])
        labels = hc.label_hydrogens(self.st, hs)
        self.sch_bv.fill(self.st, self.bv, hs, self.s, self.params, labels)
        self._flag_box(self.sch_flag, hc.h_contacts(self.st, hs, cutoff=self.s.hh_mod), labels, hs)
        if self.det_src.get() == "scheme":
            self.refresh_details()
        if self.table_src.get() == "scheme":
            self.refresh_table()

    def send_scheme(self):
        r = self.selected_scheme()
        if r is not None:
            self.load_scheme(r)

    # ============================================================== compare schemes
    def compare_schemes(self):
        """Ctrl/Shift-click several schemes, then Compare: side-by-side scores, H-bonds per donor,
        BV sums and H···H / cone flags. The H located in the CIF are added as a column when present."""
        if self.res is None or not self.res.schemes:
            messagebox.showinfo(APP_NAME, "Run the analysis first.")
            return
        rows = sorted(int(x) for x in self.sch.tree.selection())
        entries = []
        for r in rows[:12]:
            hs = hc.scheme_hatoms(self.res, self.res.schemes[r])
            entries.append(("#%d" % (r + 1), hs))
        if self.loc_all and any(h.primary for h in self.loc_all):
            entries.append(("Located (CIF)", list(self.loc_all)))
        if len(entries) < 2:
            messagebox.showinfo(APP_NAME, "Select two or more schemes (Ctrl+click / Shift+click) to compare"
                                          "%s." % ("" if not self.loc_all else ", or one scheme to compare with "
                                                   "the H located in the CIF"))
            return
        if len(rows) > 12:
            messagebox.showinfo(APP_NAME, "Comparing the first 12 selected schemes.")
        self._compare_window(entries)

    def _donor_view(self, hs):
        """{donor: (acceptor keys, text, A1-D-A2 angle, angle penalty)} for a set of H atoms."""
        st, s = self.st, self.s
        by = {}
        for h in hs:
            if h.cand is not None and not getattr(h, "flipped", False):
                by.setdefault(h.donor, {})[tuple(h.cand.nb.key)] = h
        out = {}
        for i, dct in by.items():
            cands = [h.cand for h in dct.values()]
            cands.sort(key=lambda c: (st.atoms[c.acceptor].label, c.nb.d))
            ang = hc.angle_between(cands[0].nb.vec, cands[1].nb.vec) if len(cands) == 2 else None
            pen = hc.pair_angle_penalty(ang, s) + sum(c.adc_pen or 0.0 for c in cands if hasattr(c, "adc_pen"))
            txt = ", ".join("%s%s" % (st.atoms[c.acceptor].label,
                                      "" if st.symop_code(c.nb.op, c.nb.shift) == "." else
                                      " " + st.symop_code(c.nb.op, c.nb.shift)) for c in cands)
            if ang is not None:
                txt += "  (%.0f°)" % ang
            if pen < -0.05:
                txt += "  ⚠%.0f" % pen
            out[i] = (frozenset(dct), txt, ang, pen)
        return out

    def _compare_window(self, entries):
        st, bv, s, params = self.st, self.bv, self.s, self.params
        data = []
        for name, hs in entries:
            ev = hc.evaluate_hset(st, bv, hs, s, params)
            data.append({"name": name, "hs": hs, "ev": ev, "don": self._donor_view(hs),
                         "labels": hc.label_hydrogens(st, hs)})
        win = tk.Toplevel(self.root)
        win.title("%s - compare schemes (%s)" % (APP_NAME, st.name))
        win.geometry("1150x700")
        if self.theme.get() == "dark":
            win.configure(background="#1f2329")
        outer = ttk.Frame(win, padding=6)
        outer.pack(fill="both", expand=True)
        names = [d["name"] for d in data]
        ref = data[0]
        donors = sorted({i for d in data for i in d["don"]} | {d_["i"] for d_ in self.donors})

        # ---- what differs (text)
        lines = []
        for d in data[1:]:
            diff = []
            for i in donors:
                a, b = ref["don"].get(i), d["don"].get(i)
                if (a[0] if a else None) != (b[0] if b else None):
                    diff.append("%s: %s → %s" % (st.atoms[i].label, a[1].split("  ")[0] if a else "—",
                                                 b[1].split("  ")[0] if b else "—"))
            dt = d["ev"]["total"] - ref["ev"]["total"]
            lines.append("%s vs %s (total %+.1f): %s" % (d["name"], ref["name"], dt,
                                                         "; ".join(diff) if diff else "same H-bonds"))
        ttk.Label(outer, text="\n".join(lines), wraplength=1100, justify="left").pack(fill="x")
        nb = ttk.Notebook(outer)
        nb.pack(fill="both", expand=True, pady=(6, 0))
        cols = [("what", "", 190, "w")] + [("e%d" % k, n, 150, "e") for k, n in enumerate(names)]

        # ---- scores
        t1 = ScrolledTree(nb, cols, height=14)
        nb.add(t1, text="Scores")
        t1.tree.tag_configure("diff", background="#fef3c7", foreground="black")
        metrics = [("Total score (higher = better)", "total", 2), ("BV score F&I", "bvF", 2),
                   ("BV score GRG", "bvG", 2), ("Angle penalties", "geom", 1),
                   ("H···H + cone penalties", "hh", 0), ("Red H···H (< %.2f Å)" % s.hh_red, "red", None),
                   ("Moderate H···H (%.2f–%.2f Å)" % (s.hh_red, s.hh_mod), "mod", None),
                   ("H···H < %.2f Å" % s.hh_severe, "sev", None), ("Cone violations", "cone", None),
                   ("Disorder conflicts", "dis", None), ("Max outside F&I band (vu)", "maxF", 3),
                   ("Max |Δ| GRG (vu)", "maxG", 3), ("H per cell", "h_cell", 2)]
        for lab, key, nd in metrics:
            vals = [d["ev"].get(key, 0) for d in data]
            txt = [(_fmt(v, nd) if nd is not None else str(int(v))) for v in vals]
            tag = ("diff",) if len(set(txt)) > 1 else ()
            t1.tree.insert("", "end", values=[lab] + txt, tags=tag)
        nd_ = [sum(1 for i in donors if (ref["don"].get(i) or (None,))[0] != (d["don"].get(i) or (None,))[0])
               for d in data]
        t1.tree.insert("", "end", values=["Donors with other H-bonds than %s" % ref["name"]] + [str(x) for x in nd_])

        # ---- H-bonds per donor
        f2 = ttk.Frame(nb)
        nb.add(f2, text="H-bonds by donor")
        only = tk.BooleanVar(value=False)
        t2 = ScrolledTree(f2, [("donor", "Donor", 90, "w")] + [("e%d" % k, n, 190, "w") for k, n in enumerate(names)],
                          height=18)
        t2.tree.tag_configure("diff", background="#fef3c7", foreground="black")
        t2.tree.tag_configure("pen", foreground="#b91c1c")

        def fill2():
            t2.clear()
            for i in donors:
                cells = [d["don"].get(i) for d in data]
                keys = {c[0] if c else None for c in cells}
                if only.get() and len(keys) <= 1:
                    continue
                tags = (("diff",) if len(keys) > 1 else ()) + (("pen",) if any(c and c[3] < -0.05 for c in cells)
                                                                else ())
                t2.tree.insert("", "end", tags=tags, values=[st.atoms[i].label] + [c[1] if c else "—" for c in cells])
        ttk.Checkbutton(f2, text="Only donors that differ", variable=only, command=fill2).pack(anchor="w")
        t2.pack(fill="both", expand=True)
        ttk.Label(f2, text="Acceptor (symmetry code if not x,y,z), A1–D–A2 angle, ⚠ = angle penalty "
                           "(A1–D–A2 and A–D–C). Yellow rows differ between the columns.").pack(anchor="w")
        fill2()

        # ---- BV sums
        f3 = ttk.Frame(nb)
        nb.add(f3, text="BV sums")
        c3 = [("site", "Site", 70, "w"), ("cls", "Class", 110, "w")]
        for k, n in enumerate(names):
            c3 += [("f%d" % k, "%s F&I Σ" % n, 105, "e"), ("g%d" % k, "%s GRG Δ" % n, 90, "e")]
        t3 = ScrolledTree(f3, c3, height=18)
        t3.tree.tag_configure("diff", background="#fef3c7", foreground="black")
        t3.tree.tag_configure("bad", background="#fecaca", foreground="black")
        for i, a in enumerate(st.atoms):
            if not ref["ev"]["r"]["anion"][i]:
                continue
            vals, outs = [a.label, bv.cls[i] or bv.roles[i]], []
            for d in data:
                r = d["ev"]["r"]
                o = float(r["outF"][i])
                outs.append(o)
                vals += ["%.3f%s" % (r["totF"][i], (" ▲%.2f" % o) if o > 0.0005 else ""), "%+.3f" % r["devG"][i]]
            tags = ("bad",) if max(outs) >= 0.15 else (("diff",) if len({round(x, 3) for x in outs}) > 1 else ())
            t3.tree.insert("", "end", tags=tags, values=vals)
        t3.pack(fill="both", expand=True)
        ttk.Label(f3, text="F&I Σ (▲ = amount outside the target band), GRG Δ = GRG sum − ideal. Red rows: "
                           "≥ 0.15 vu outside the F&I band in some column; yellow: the columns differ.").pack(anchor="w")

        # ---- H...H and cone
        t4 = ScrolledTree(nb, [("entry", "Scheme", 110, "w"), ("pair", "H···H / cone", 260, "w"),
                               ("d", "Distance / angle", 120, "e"), ("kind", "Flag", 200, "w")], height=18)
        nb.add(t4, text="H···H and cone flags")
        t4.tree.tag_configure("red", background="#fecaca", foreground="black")
        t4.tree.tag_configure("mod", background="#fed7aa", foreground="black")
        for d in data:
            lb = d["labels"]
            geo = [c for c in d["ev"]["contacts"] if hc.contact_is_geometry(st, c)]
            for c in sorted(geo, key=lambda c: c[2]):
                kind = "red" if c[2] < s.hh_red else "mod"
                t4.tree.insert("", "end", tags=(kind,), values=(
                    d["name"], "%s%s – %s%s" % (lb.get(id(c[0]), "H"), "" if c[0].located else "*",
                                                lb.get(id(c[1]), "H"), "" if c[1].located else "*"),
                    "%.2f Å" % c[2], "red" if kind == "red" else "moderate (sliding penalty)"))
            for h, w, an_, half in d["ev"].get("cones", []):
                t4.tree.insert("", "end", tags=("red",), values=(
                    d["name"], "%s → cone of %s" % (lb.get(id(h), "H"), st.atoms[w].label),
                    "%.1f° < %.1f°" % (an_, half), "acceptor-cone violation"))
            if not geo and not d["ev"].get("cones"):
                t4.tree.insert("", "end", values=(d["name"], "none", "", ""))
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(6, 0))

        def to_cmp():
            for d in data:
                nm = "H in CIF (located)" if d["name"].startswith("Located") else "Scheme %s" % d["name"]
                if not any(c["name"] == nm for c in self.cmp):
                    self._add_cmp(nm, d["hs"], d["ev"], "" if nm.startswith("H in CIF") else
                                  hc.scheme_summary(st, self.res, self.res.schemes[int(d["name"][1:]) - 1]),
                                  tag=("cif",) if nm.startswith("H in CIF") else ())
            self.nb.select(TAB_REFINE)
        ttk.Button(bar, text="Add all to Refinement comparison", command=to_cmp).pack(side="left")
        schemes_only = [d for d in data if d["name"].startswith("#")]
        if len(schemes_only) >= 2:
            accent_button(bar, "  Combine into disordered scheme...  ",
                          lambda: self._combine_dialog(win, schemes_only), self.bold).pack(side="left", padx=8)
        ttk.Button(bar, text="Close", command=win.destroy).pack(side="right")
        self._compare_win = (win, t1, t2, t3, t4)
        if self.theme.get() == "dark":
            self.apply_theme()

    def _combine_dialog(self, parent, data):
        """Weights for a combined (disordered) scheme made of the compared schemes."""
        dlg = tk.Toplevel(parent)
        dlg.title("%s - combine into a disordered scheme" % APP_NAME)
        dlg.transient(parent)
        f = ttk.Frame(dlg, padding=10)
        f.pack(fill="both", expand=True)
        ttk.Label(f, wraplength=470, justify="left", text=(
            "Each scheme becomes one component of a disordered model, present in a fraction of the cells given by "
            "its weight (weights are normalised to sum 1; fractions such as 1/3 are accepted). H at the same "
            "position in several components are merged; H of different components never clash with each other.")
                  ).grid(row=0, column=0, columnspan=3, sticky="w")
        n = len(data)
        vars_ = []
        for k, d in enumerate(data):
            ttk.Label(f, text="Scheme %s" % d["name"]).grid(row=1 + k, column=0, sticky="w", pady=2)
            v = tk.StringVar(value="1/%d" % n)
            ttk.Entry(f, textvariable=v, width=8).grid(row=1 + k, column=1, sticky="w", padx=6)
            ttk.Label(f, text="total %.1f" % d["ev"]["total"], foreground="#555").grid(row=1 + k, column=2, sticky="w")
            vars_.append(v)

        def weight(txt):
            txt = txt.strip()
            if "/" in txt:
                a, b = txt.split("/", 1)
                return float(a) / float(b)
            return float(txt)

        def create():
            try:
                ws = [weight(v.get()) for v in vars_]
                if any(w < 0 for w in ws) or sum(ws) <= 0:
                    raise ValueError
            except (ValueError, ZeroDivisionError):
                messagebox.showerror(APP_NAME, "Weights must be non-negative numbers or fractions (e.g. 0.5 or 1/3).",
                                     parent=dlg)
                return
            tot = sum(ws)
            ws = [w / tot for w in ws]
            used = [(d, w) for d, w in zip(data, ws) if w > 0]
            if len(used) < 2:
                messagebox.showerror(APP_NAME, "Give at least two schemes a weight above 0.", parent=dlg)
                return
            hs = hc.combine_hsets([(d["hs"], w) for d, w in used])
            ev = hc.evaluate_hset(self.st, self.bv, hs, self.s, self.params)
            name = "Disordered: " + " + ".join("%s ×%s" % (d["name"], hc.occ_text(w)) for d, w in used)
            info = "; ".join("%s (×%s): total %.1f, angle %.1f, H···H %.0f" % (
                d["name"], hc.occ_text(w), ev["components"][k][1]["total"], ev["components"][k][1]["geom"],
                ev["components"][k][1]["hh"]) for k, (d, w) in enumerate(used))
            self._add_cmp(name, hs, ev, info, tag=("red",) if ev["red"] else (("mod",) if ev["mod"] else ()))
            dlg.destroy()
            self.ref_status.set("Added '%s' - select it for its BV sums, 'BV table ▶', or 'Finalize ▶' to export it "
                                "as a CIF / SHELXL file." % name)
            self.nb.select(TAB_REFINE)
        b = ttk.Frame(f)
        b.grid(row=2 + n, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        accent_button(b, "  Create  ", create, self.bold).pack(side="left")
        ttk.Button(b, text="Cancel", command=dlg.destroy).pack(side="right")

    def goto_refine(self):
        """Schemes tab 'Refine ▶': put the selected scheme (or #1) into the Refinement comparison
        (unrefined) and select it there, so it can be refined straight away."""
        if self.res is not None and self.res.schemes:
            r = self.selected_scheme()
            if r is None:
                r = 0
                self.sch.tree.selection_set("0")
            name = "Scheme #%d" % (r + 1)
            k = next((n for n, c in enumerate(self.cmp) if c["name"] == name), None)
            if k is None:
                hs = hc.scheme_hatoms(self.res, self.res.schemes[r])
                ev = hc.evaluate_hset(self.st, self.bv, hs, self.s, self.params)
                self._add_cmp(name, hs, ev, hc.scheme_summary(self.st, self.res, self.res.schemes[r]))
            else:
                self.cmpt.tree.selection_set(str(k))
                self.cmpt.tree.see(str(k))
        self.nb.select(TAB_REFINE)

    # ============================================================== refinement
    def _add_cmp(self, name, hs, ev, info="", tag=()):
        self.cmp.append({"name": name, "hs": hs, "ev": ev, "info": info})
        k = len(self.cmp) - 1
        if not tag:
            tag = ("red",) if ev["red"] else (("mod",) if ev["mod"] else ())
        self.cmpt.tree.insert("", "end", iid=str(k), tags=tag, values=(
            name, _fmt(ev["total"], 2), _fmt(ev["bvF"], 2), _fmt(ev["bvG"], 2), _fmt(ev["geom"], 1),
            _fmt(ev["hh"], 0), ev["red"], ev["mod"], ev["sev"], ev.get("cone", 0), ev.get("dis", 0),
            _fmt(ev["maxF"]), _fmt(ev["maxG"]),
            _fmt(ev["h_cell"], 2), info))
        self.cmpt.tree.selection_set(str(k))
        self.cmpt.tree.see(str(k))

    def add_selected_scheme(self):
        r = self.selected_scheme()
        if r is None:
            messagebox.showinfo(APP_NAME, "Select a scheme on the Schemes tab first.")
            return
        hs = hc.scheme_hatoms(self.res, self.res.schemes[r])
        ev = hc.evaluate_hset(self.st, self.bv, hs, self.s, self.params)
        self._add_cmp("Scheme #%d" % (r + 1), hs, ev, hc.scheme_summary(self.st, self.res, self.res.schemes[r]))

    def score_cif_h(self):
        if self.st is None:
            return
        if not self.loc_all:
            messagebox.showinfo(APP_NAME, "This CIF contains no H atoms attached to an anion.")
            return
        ev = hc.evaluate_hset(self.st, self.bv, self.loc_all, self.s, self.params)
        donors = {}
        for h in self.loc_all:
            donors.setdefault(self.st.atoms[h.donor].label, []).append(h)
        missing = [d["label"] for d in self.donors if d["n_h"] > len([h for h in self.loc_all if h.donor == d["i"]])]
        info = "H as in the CIF (%d H on %s)" % (sum(1 for h in self.loc_all if h.primary), ", ".join(sorted(donors)))
        if missing:
            info += "; no H located on %s" % ", ".join(missing)
        self._add_cmp("H in CIF (located)", list(self.loc_all), ev, info, tag=("cif",))
        self.nb.select(TAB_REFINE)

    def refine_selected(self):
        try:
            s = self.settings()
        except ValueError as e:
            messagebox.showerror("Invalid settings", str(e))
            return
        if self.res is None or not self.res.schemes:
            return
        if s.refine_auto_pick:
            r = hc.pick_for_refinement(self.res)
            self.sch.tree.selection_set(str(r))
            self.sch.tree.see(str(r))
        else:
            r = self.selected_scheme()
        if r is None:
            messagebox.showinfo(APP_NAME, "Select a scheme on the Schemes tab first.")
            return
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        mode_lab = self.ref_mode.get()
        mode = REFINE_MODES[mode_lab]
        hs = hc.scheme_hatoms(self.res, self.res.schemes[r])
        st, bv, params = self.st, self.bv, self.params
        ev0 = hc.evaluate_hset(st, bv, hs, self.s, params)
        if not any(c["name"] == "Scheme #%d" % (r + 1) for c in self.cmp):
            self._add_cmp("Scheme #%d" % (r + 1), hs, ev0, hc.scheme_summary(st, self.res, self.res.schemes[r]))
        box = {}
        self._ref_evals = 0

        def prog(n):
            self._ref_evals = n

        def work():
            try:
                box["rs"] = hc.refine_scheme(st, bv, hs, s, params, mode=mode, progress=prog)
            except Exception as e:  # noqa: BLE001
                box["err"] = "%s\n\n%s" % (e, traceback.format_exc(limit=4))

        self.ref_btn.configure(state="disabled")
        self.root.config(cursor="watch")
        self._bg_thread = threading.Thread(target=work, daemon=True)
        self._bg_thread.start()

        def poll():
            if self._bg_thread.is_alive():
                self.ref_status.set("Refining scheme #%d (%s)... %d evaluations" % (r + 1, mode_lab, self._ref_evals))
                self.root.after(150, poll)
                return
            self.root.config(cursor="")
            self.ref_btn.configure(state="normal")
            if "err" in box:
                messagebox.showerror("Refinement failed", box["err"])
                return
            rs = box["rs"]
            if not rs.donors:
                self.ref_status.set("Nothing to refine in scheme #%d for mode '%s'." % (r + 1, mode_lab))
                return
            ev = hc.evaluate_hset(st, bv, rs.hs, self.s, params)
            dhc = getattr(rs, "dh_changes", {}) or {}

            def _dh(i):
                ch = [c for c in dhc.get(i, []) if abs(c) >= 0.0005]
                return (" (D–H %s Å)" % "/".join("%+.3f" % c for c in dhc[i])) if ch else ""
            info = ", ".join("%s %.1f°%s%s" % (st.atoms[i].label, rs.rotations[i],
                                               (" (H–D–H %+.1f°)" % rs.scissors[i]) if abs(rs.scissors.get(i, 0)) > 0.05
                                               else "", _dh(i)) for i in rs.donors)
            new_name = "Scheme #%d refined (%s, ≤%g°%s%s)" % (r + 1, mode, rs.max_deg,
                                                             ", D–H ±%g Å" % s.refine_dh_range if s.refine_dh else "",
                                                             ", auto-picked" if s.refine_auto_pick else "")
            self._add_cmp(new_name, rs.hs, ev, info + "  | LSQ %.1f → %.1f" % (rs.e_start, rs.e_end))
            self.ref_status.set("Refined scheme #%d: %d donor(s), least-squares sum %.1f → %.1f (%d evaluations)."
                                % (r + 1, len(rs.donors), rs.e_start, rs.e_end, rs.evals))
            self.nb.select(TAB_REFINE)
            self._subs_follow_refine(r, new_name)

        self.root.after(150, poll)

    def selected_cmp(self):
        sel = self.cmpt.tree.selection()
        return self.cmp[int(sel[0])] if sel else None

    def show_cmp(self):
        e = self.selected_cmp()
        if e is None:
            return
        labels = hc.label_hydrogens(self.st, e["hs"])
        self.cmp_bv.fill(self.st, self.bv, e["hs"], self.s, self.params, labels)
        self._flag_box(self.cmp_flag, e["ev"]["contacts"], labels, e["hs"])
        if self.det_src.get() == "refine":
            self.refresh_details()
        if self.table_src.get() == "refine":
            self.refresh_table()

    def remove_cmp(self):
        sel = self.cmpt.tree.selection()
        if not sel:
            return
        del self.cmp[int(sel[0])]
        entries, self.cmp = self.cmp, []
        self.cmpt.clear()
        for e in entries:
            self._add_cmp(e["name"], e["hs"], e["ev"], e["info"],
                          tag=("cif",) if e["name"].startswith("H in CIF") else ())
        self.cmp_bv.clear()

    def clear_cmp(self):
        self.cmp = []
        self.cmpt.clear()
        self.cmp_bv.clear()
        self.cmp_flag.configure(text="", background=self.root.cget("background"))

    def cmp_table(self):
        self.table_src.set("refine")
        self.refresh_table()
        self.nb.select(TAB_TABLE)

    # ========================================================== scheme builder
    def bv_step(self):
        """The first step of the analysis only (bond valence, classes, roles, H-bond candidates and
        options - no scheme search), for the scheme builder. Returns True when it is ready."""
        st = self.current_structure()
        if st is None or self.params is None:
            messagebox.showinfo(APP_NAME, "Open a CIF first.")
            return False
        try:
            s = self.settings()
            la = self.located_args(s)
        except ValueError as e:
            messagebox.showerror("Invalid settings", str(e))
            return False
        self.clear_all()
        self._clear_pending()
        ov = self.overrides.setdefault(st.title, {})
        cov = dict(self.class_overrides.setdefault(st.title, {}))
        rov = dict(self.role_overrides.setdefault(st.title, {}))
        vov = dict(self.valence_overrides.setdefault(st.title, {}))
        bv = hc.analyse_bv(st, self.params, s, ov, cov, vov)
        hc.apply_site_roles(st, bv, s, rov)
        loc_all = hc.located_hatoms(st, bv, self.params, s)
        loc = [] if s.ignore_located else loc_all
        donors = hc.infer_hbonds(st, bv, self.params, s, loc)
        located = hc.located_hbonds(st, env_radius=s.env_radius, **la)
        self.st, self.s, self.bv, self.loc_h, self.donors, self.res, self.located, self.loc_all = \
            st, s, bv, loc, donors, None, located, loc_all
        self.populate_bv()
        self.populate_inferred()
        self.populate_located()
        self.build_finalize()
        self.sch_info.set("Only the bond-valence step has been run (for the scheme builder) - press Run analysis "
                          "to rank schemes.")
        self.status_var.set("Bond-valence step done (classes, roles, candidates) - no scheme search yet.")
        return True

    # ======================================================= structure optimisation
    def optimize_dialog(self):
        st = self.current_structure()
        if st is None:
            messagebox.showinfo(APP_NAME, "Open a CIF first.")
            return
        if self.st is None or self.st is not st or self._pending:
            if not self.bv_step():
                return
        st, bv, s = self.st, self.bv, self.s
        win = tk.Toplevel(self.root)
        win.title("%s - optimize structure (%s)" % (APP_NAME, st.name))
        win.geometry("1060x720")
        f = ttk.Frame(win, padding=8)
        f.pack(fill="both", expand=True)
        intro = ttk.Label(f, justify="left", text=(
            "Moves the ticked atoms by small amounts (within the radius, along directions their site symmetry allows) "
            "to bring the CATION bond-valence sums closer to their valences before the H-bond analysis (anion sums "
            "are not targeted - their H and H-bonds settle them later; they are shown for information). Moving costs "
            "weight × (exp(steepness × (shift / radius)²) − 1): cheap for small shifts, steep near the radius - so a "
            "cation's deficit is shared among its anions. Fully occupied cations and H stay by default (H follow "
            "their parent). The result is a NEW structure entry '[optimized]'; its CIF output is named "
            "<name>_optimized_hypothetical.cif."))
        intro.pack(fill="x")
        intro.bind("<Configure>", lambda e: intro.configure(wraplength=max(300, e.width - 4)))
        sbar = ttk.Frame(f)
        sbar.pack(fill="x", pady=6)
        vals = {}
        for lab, key, dv in (("Radius Å", "radius", "0.03"), ("σ vu", "sigma", "0.05"),
                             ("Move weight", "w_move", "1.0"), ("Steepness", "steep", "3.0")):
            ttk.Label(sbar, text=lab).pack(side="left", padx=(10, 3))
            vals[key] = tk.StringVar(value=dv)
            ttk.Entry(sbar, textvariable=vals[key], width=6).pack(side="left")
        t = ScrolledTree(f, [("mv", "Move", 45, "center"), ("site", "Site", 70, "w"), ("el", "El", 35, "w"),
                             ("role", "Role", 55, "w"), ("occ", "Occ.", 60, "e"), ("cls", "D/A class", 130, "w"),
                             ("bvs", "BVS (w/o H)", 85, "e"), ("tar", "Target", 95, "center"),
                             ("after", "After", 70, "e"), ("dshift", "Shift Å", 70, "e")], height=16)
        t.pack(fill="both", expand=True)
        t.tree.tag_configure("off", foreground="#9ca3af")
        t.tree.tag_configure("bad", background="#fecaca", foreground="black")
        t.tree.tag_configure("good", background="#dcfce7", foreground="black")
        mov = set(hc.default_movable(st, bv, s))
        heavy = [i for i, a in enumerate(st.atoms) if a.el != "H"]

        def target(i):
            if bv.roles[i] == "cation":
                return "%g" % bv.ideal[i] if bv.ideal[i] else ""
            return "(not fitted)"

        def fill(res=None):
            t.clear()
            for i in heavy:
                a = st.atoms[i]
                after, shift, tag = "", "", ()
                if res is not None:
                    after = "%.3f" % res["bvs1"][i]
                    sh = res["shift"].get(i, 0.0)
                    shift = "%.4f" % sh if sh > 1e-6 else ""
                    if res["cat"][i]:
                        d0, d1 = abs(res["bvs0"][i] - res["V"][i]), abs(res["bvs1"][i] - res["V"][i])
                        tag = ("bad",) if d1 > 0.25 else (("good",) if d1 < d0 - 0.005 else ())
                t.tree.insert("", "end", iid=str(i), tags=tag + (() if i in mov else ("off",)), values=(
                    "☑" if i in mov else "☐", a.label, a.el, bv.roles[i], _fmt(a.occ, 3), bv.cls[i] or "",
                    _fmt(bv.bvs_noH[i]), target(i), after, shift))

        def toggle(event):
            row = t.tree.identify_row(event.y)
            if row:
                mov.symmetric_difference_update({int(row)})
                fill(state.get("res"))
        t.tree.bind("<Button-1>", lambda e: toggle(e) if t.tree.identify_column(e.x) == "#1" else None)
        t.tree.bind("<Double-1>", toggle)
        state = {}
        fill()
        summ = tk.StringVar(value="Tick / untick atoms in the Move column (click it), set the radius, then Optimize.")
        sl = ttk.Label(f, textvariable=summ, justify="left")
        sl.pack(fill="x", pady=(6, 0))
        sl.bind("<Configure>", lambda e: sl.configure(wraplength=max(300, e.width - 4)))
        bb = ttk.Frame(f)
        bb.pack(fill="x", pady=(6, 0))

        def sel(kind):
            mov.clear()
            if kind == "default":
                mov.update(hc.default_movable(st, bv, s))
            elif kind == "anions":
                mov.update(i for i in heavy if bv.roles[i] == "anion")
            fill(state.get("res"))

        def run():
            try:
                kw = {k: float(v.get()) for k, v in vals.items()}
                if kw["radius"] <= 0 or kw["sigma"] <= 0 or kw["steep"] <= 0 or kw["w_move"] < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(APP_NAME, "Radius, σ and steepness must be positive numbers.", parent=win)
                return
            box, prog = {}, {"n": 0}

            def work():
                try:
                    box["res"] = hc.optimize_structure(st, bv, s, set(mov), progress=lambda n: prog.update(n=n), **kw)
                except Exception as e:  # noqa: BLE001
                    box["err"] = "%s\n\n%s" % (e, traceback.format_exc(limit=4))
            th = threading.Thread(target=work, daemon=True)
            th.start()
            opt_btn.configure(state="disabled")

            def poll():
                if th.is_alive():
                    summ.set("Optimizing ... %d evaluations" % prog["n"])
                    win.after(150, poll)
                    return
                opt_btn.configure(state="normal")
                if "err" in box:
                    messagebox.showerror(APP_NAME, box["err"], parent=win)
                    return
                res = state["res"] = box["res"]
                state["kw"] = kw
                fill(res)
                cat, V = res["cat"], res["V"]
                rms = lambda S: float(np.sqrt(np.mean((S[cat] - V[cat]) ** 2))) if cat.any() else 0.0
                mx = lambda S: float(np.max(np.abs(S[cat] - V[cat]))) if cat.any() else 0.0
                sh = [v for i, v in res["shift"].items() if v > 1e-6 and st.atoms[i].el != "H"]
                stuck = [st.atoms[i].label for i in np.nonzero(cat)[0] if abs(res["bvs1"][i] - V[i]) > 0.25]
                summ.set("Cations: rms |BVS − V| %.3f → %.3f vu, largest %.3f → %.3f.   Moved %d atom(s): mean %.4f Å, "
                         "largest %.4f Å (radius %.3f).   Cost %.1f → %.1f (BV %.1f → %.1f, moving %.1f).%s" % (
                             rms(res["bvs0"]), rms(res["bvs1"]), mx(res["bvs0"]), mx(res["bvs1"]), len(sh),
                             float(np.mean(sh)) if sh else 0.0, max(sh) if sh else 0.0, kw["radius"],
                             res["cost0"][0], res["cost1"][0], res["cost0"][1] + res["cost0"][2],
                             res["cost1"][1] + res["cost1"][2], res["cost1"][3],
                             ("   ⚠ Still > 0.25 vu off (small shifts cannot fix - check valence / occupancy / "
                              "model): " + ", ".join(stuck)) if stuck else ""))
                use_btn.configure(state="normal" if res["new_f"] else "disabled")
            poll()

        def use():
            res = state.get("res")
            if not res or not res["new_f"]:
                return
            kw = state["kw"]
            n_h = sum(1 for i in res["new_f"] if st.atoms[i].el == "H")
            note = "radius %.3f A, sigma %.3f vu, move weight %g, steepness %g; %d atom(s) moved%s, largest %.4f A" % (
                kw["radius"], kw["sigma"], kw["w_move"], kw["steep"], len(res["new_f"]) - n_h,
                (" (+%d H with their parent)" % n_h) if n_h else "", max(res["shift"].values()))
            base = st
            st2 = hc.optimized_structure(base, res["new_f"], note)
            self._copy_overrides(base, st2)      # keep the hand-set classes, valences, roles and bonds
            self.structures = [x for x in self.structures if x.title != st2.title] + [st2]
            names = [x.title for x in self.structures]
            self.struct_box.configure(values=names, state="readonly")
            self.struct_var.set(st2.title)
            self.on_structure_change()
            win.destroy()
            self.run_btn.configure(state="normal")
            self.run()                               # analyse the optimized structure straight away
            self.status_var.set("Analysing the optimized structure (%s) ... its CIF / SHELXL output is named "
                                "<name>_optimized_hypothetical / _optimized_HydroGEN." % note)

        ttk.Button(bb, text="Default selection", command=lambda: sel("default")).pack(side="left")
        ttk.Button(bb, text="All anions", command=lambda: sel("anions")).pack(side="left", padx=4)
        ttk.Button(bb, text="None", command=lambda: sel("none")).pack(side="left")
        opt_btn = accent_button(bb, "  Optimize  ", run, self.bold)
        opt_btn.pack(side="left", padx=(14, 4))
        use_btn = accent_button(bb, "  Use optimized structure  ", use, self.bold)
        use_btn.configure(state="disabled")
        use_btn.pack(side="left", padx=4)
        ttk.Button(bb, text="Close", command=win.destroy).pack(side="right")
        if self.theme.get() == "dark":
            self.apply_theme()

    def _copy_overrides(self, src, dst):
        """Copy the hand-made choices of structure src (classes, valences / site mixing, H-bond roles, bond
        inclusions) to structure dst with the same sites (e.g. its optimized copy). Bond overrides are
        matched to the same neighbour in dst (positions may have moved slightly)."""
        for store in (self.class_overrides, self.valence_overrides, self.role_overrides, self.role_reason):
            if store.get(src.title):
                store[dst.title] = dict(store[src.title])
        ov = self.overrides.get(src.title)
        if not ov:
            return
        env_s = src.environment(self.s.env_radius if self.s else 6.0)
        env_d = dst.environment(self.s.env_radius if self.s else 6.0)
        new = {}
        for (i, key), state in ov.items():
            nb0 = next((nb for nb in env_s[i] if nb.key == key), None)
            if nb0 is None:
                continue
            cand = [nb for nb in env_d[i] if nb.j == nb0.j and nb.op == nb0.op and nb.shift == nb0.shift]
            if cand:
                nb1 = min(cand, key=lambda nb: sum((nb.vec[k] - nb0.vec[k]) ** 2 for k in range(3)))
                new[(i, nb1.key)] = state
        if new:
            self.overrides[dst.title] = new

    def open_builder(self):
        st = self.current_structure()
        if st is None:
            messagebox.showinfo(APP_NAME, "Open a CIF first.")
            return
        if self.st is None or self.st is not st or self._pending:
            if not self.bv_step():
                return
        self._builder = SchemeBuilder(self)

    def adjust_scheme(self):
        """Refinement tab 'Adjust...': open the scheme builder with the selected comparison entry (or else
        the scheme selected on the Schemes tab) loaded, to change some of its H-bonds by hand."""
        if self.st is None:
            messagebox.showinfo(APP_NAME, "Run the analysis first.")
            return
        e = self.selected_cmp()
        if e is not None:
            hs, name = e["hs"], e["name"]
        else:
            r = self.selected_scheme()
            if r is None:
                messagebox.showinfo(APP_NAME, "Select an entry in the comparison table (or a scheme on the Schemes "
                                              "tab) to adjust.")
                return
            hs, name = hc.scheme_hatoms(self.res, self.res.schemes[r]), "Scheme #%d" % (r + 1)
        self._builder = SchemeBuilder(self)
        self._builder.load_hs(hs, name)

    def cmp_to_finalize(self):
        e = self.selected_cmp()
        if e is None:
            return
        self._fin_loaded_cmp = id(e)
        self.load_hatoms(e["hs"], e["name"])
        self.nb.select(TAB_FINAL)

    # ================================================================ BV table
    def _set_table(self, text):
        self.table_text.configure(state="normal")
        self.table_text.delete("1.0", "end")
        self.table_text.insert("1.0", text)
        self.table_text.configure(state="disabled")

    def table_hatoms(self):
        if self.st is None:
            return None, ""
        if self.table_src.get() == "final":
            return self.final_hatoms(), "Finalize selection"
        if self.table_src.get() == "refine":
            e = self.selected_cmp()
            if e:
                return e["hs"], e["name"]
        r = self.selected_scheme()
        if r is None:                            # no scheme (e.g. a structure without H): the BV analysis itself
            return (list(self.loc_h), "located H only") if self.loc_h else ([], "bond valence only - no H")
        return hc.scheme_hatoms(self.res, self.res.schemes[r]), "scheme #%d" % (r + 1)

    def refresh_table(self):
        hs, what = self.table_hatoms()
        if hs is None:
            self.table_data = None
            self._set_table("Run the analysis and select a scheme (Schemes tab) or H atoms (Finalize tab).")
            return
        method = "grg" if self.table_method.get().startswith("GRG") else "fi"
        title, header, rows, notes = hc.bv_table(self.st, self.bv, hs, self.s, self.params, method)
        title += "  [%s]" % what
        self.table_data = (title, header, rows, notes)
        self._set_table(hc.bv_table_text(title, header, rows, notes))

    def copy_table(self):
        if self.table_data:
            self.root.clipboard_clear()
            self.root.clipboard_append(hc.bv_table_text(*self.table_data))
            self.status_var.set("BV table copied to the clipboard.")

    def save_table(self, kind):
        if not self.table_data:
            return
        base = os.path.splitext(os.path.basename(self.st.source))[0] + "_bv_table"
        path = filedialog.asksaveasfilename(
            title="Save BV table", defaultextension="." + kind, initialfile=base + "." + kind,
            initialdir=os.path.dirname(self.st.source),
            filetypes=[("Text", "*.txt")] if kind == "txt" else [("CSV", "*.csv")])
        if not path:
            return
        title, header, rows, notes = self.table_data
        if kind == "txt":
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(hc.bv_table_text(title, header, rows, notes) + "\n")
        else:
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow([title])
                w.writerow(header)
                for row in rows:
                    w.writerow(["\n".join(cell) for cell in row])
                for n in notes:
                    w.writerow([n])
        self.status_var.set("Saved %s" % path)

    def show_final_table(self):
        self.table_src.set("final")
        self.refresh_table()
        self.nb.select(TAB_TABLE)

    # =========================================================== other tabs
    def populate_located(self):
        rows, notes = self.located
        self.loc.clear()
        for r in rows:
            self.loc.tree.insert("", "end", values=(
                "%s–%s···%s" % (r["donor"], r["hydrogen"], r["acceptor"]), _fmt(r["d_DH"]), _fmt(r["d_HA"]),
                _fmt(r["d_DA"]), _fmt(r["angle"], 1), r["symop"], r["code"]))
        self.loc_note.set("  ".join(notes))

    def show_env(self):
        if self.st is None or not self.env_site.get():
            return
        i = int(self.env_site.get().split(":")[0]) - 1
        st, bv = self.st, self.bv
        env = st.environment(self.s.env_radius)[i]
        bmap = {bd.nb.key: bd for bd in bv.bonds[i]}
        ov = self.overrides.get(st.title, {})
        self.env.clear()
        self._env_rows = {}
        for k, nb in enumerate(env):
            j = nb.j
            bd = bmap.get(nb.key)
            box, tags = "", ()
            if bv.roles[i] == bv.roles[j]:
                note = "%s–%s (not a bond)" % (bv.roles[i], bv.roles[j])
            elif bd is None:
                note = "s < %.2f vu" % self.s.list_cutoff
            else:
                note, manual = self._bond_status(i, bd)
                state = ov.get((i, bd.nb.key), bd.included)
                box = "☑" if state else "☐"
                if state != bd.included:
                    note = note.replace(" (manual)", "") + " → %s (pending)" % ("summed" if state else "not summed")
                tags = ("manual",) if manual or state != bd.included else (() if state else ("off",))
                self._env_rows[str(k)] = (i, bd)
            self.env.tree.insert("", "end", iid=str(k), tags=tags, values=(
                box, st.atoms[j].label, st.atoms[j].el, _fmt(nb.d, 4), _fmt(bd.s, 4) if bd else "", note,
                st.symop_text(nb.op, nb.shift), st.symop_code(nb.op, nb.shift)))
        self.angles.clear()
        try:
            rad = float(self.angle_radius.get())
        except ValueError:
            rad = 3.0
        for a, b, ang in hc.neighbor_angles(st, i, min(rad, self.s.env_radius), self.s.env_radius):
            self.angles.tree.insert("", "end", values=(
                "%s %s" % (st.atoms[a.j].label, st.symop_code(a.op, a.shift)),
                "%s %s" % (st.atoms[b.j].label, st.symop_code(b.op, b.shift)), _fmt(ang, 2), _fmt(a.d), _fmt(b.d)))

    def env_toggle(self, event):
        row = self.env.tree.identify_row(event.y)
        hit = getattr(self, "_env_rows", {}).get(row)
        if not hit or self.bv is None:
            return
        i, bd = hit
        cur = self.overrides.get(self.st.title, {}).get((i, bd.nb.key), bd.included)
        self._toggle_bond_pair(i, bd, not cur)
        self.show_env()

    # ================================================================ finalize
    def build_finalize(self):
        st, res = self.st, self.res
        self.fin.clear()
        self.fin_h, self.fin_pass, self.fin_info, self.checked, self.fin_loc = {}, set(), {}, set(), {}
        by_key = {}
        if res is not None:
            opts_of = {d["i"]: ol for d, ol in zip(res.donors, res.opts)}
        else:                                   # bond-valence step only (scheme builder): every option
            opts_of = {d["i"]: d["options"] for d in self.donors if d.get("options")}
        self.fin_opts_of = opts_of
        self.fin_rowkey = {}
        self._opt_hkeys = {h.key for ol in opts_of.values() for o in ol for h in o.hs}
        n = 0
        self.fin_parent = {}
        for d in self.donors:
            pid = self.fin.tree.insert("", "end", text="%s - %s" % (d["label"], d["cls"]), open=True,
                                       tags=("donor",))
            self.fin_parent[d["i"]] = pid
            for h in d["located"]:
                iid = "L%d" % len(self.fin_loc)
                self.fin_loc[iid] = h
                self.fin.tree.insert(pid, "end", iid=iid, tags=("located",), text=(
                    "located %s" % h.located if h.primary else "   ↳ %s also bonds to %s" % (
                        h.located, st.atoms[h.cand.acceptor].label)))
            ol = opts_of.get(d["i"], [])
            if d["n_missing"] and not ol:
                self.fin.tree.insert(pid, "end", text="(%d missing H: no passing candidates)" % d["n_missing"])
            # one row per H-bond (donor -> acceptor position); the exact H position of a water depends
            # on its other H-bond and is taken from the matching option when the rows are ticked
            for o in ol:
                for h in o.hs:
                    rk = self._rowkey(h)
                    iid = by_key.get(rk)
                    if iid is None:
                        n += 1
                        iid = "h%d" % n
                        by_key[rk] = iid
                        self.fin_rowkey[iid] = rk
                        self.fin_h[iid] = h
                        self.fin_info[iid] = {"opts": []}
                        self.fin.tree.insert(pid, "end", iid=iid, text="H → %s %s" % (
                            hc.ordinal(h.cand.ordinal), st.atoms[h.cand.acceptor].label))
                    if o.number not in self.fin_info[iid]["opts"]:
                        self.fin_info[iid]["opts"].append(o.number)
                    if o.passes:
                        self.fin_pass.add(iid)
        self.fin_by_key = by_key
        self._fin_n = n
        if res is not None and res.schemes:
            self.load_scheme(0, switch=False)
        else:
            self.refresh_finalize()

    def _rowkey(self, h):
        """Finalize row of an inferred H: its H-bond (donor, acceptor position) for H placed by the
        search; its exact position for refined / otherwise moved H."""
        if h.key in getattr(self, "_opt_hkeys", set()) and h.cand is not None:
            return ("hb", h.donor, tuple(h.cand.nb.key))
        return ("pos",) + tuple(h.key)

    def _fin_effective(self):
        """{row: H atom} for the ticked rows. H-bond rows of one donor take their positions from the
        option with exactly those acceptors (e.g. a water's two H placed for that pair)."""
        out, by_donor = {}, {}
        for iid in self.checked:
            if self.fin_rowkey.get(iid, ("pos",))[0] == "hb":
                by_donor.setdefault(self.fin_h[iid].donor, []).append(iid)
            else:
                out[iid] = self.fin_h[iid]
        for i, iids in by_donor.items():
            want = sorted(self.fin_rowkey[x][2] for x in iids)
            o = next((o for o in getattr(self, "fin_opts_of", {}).get(i, [])
                      if sorted(tuple(c.nb.key) for c in o.cands) == want), None)
            if o is not None:
                pos = {tuple(h.cand.nb.key): h for h in o.hs}
                for x in iids:
                    out[x] = pos.get(self.fin_rowkey[x][2], self.fin_h[x])
            else:
                for x in iids:
                    out[x] = self.fin_h[x]
        return out

    def final_hatoms(self):
        """Located H (with branches) + ticked inferred H; located-H branches that must flip are flipped."""
        eff = self._fin_effective()
        hs = list(self.fin_loc.values()) + [eff[i] for i in sorted(self.checked, key=lambda x: int(x[1:]))]
        return hc.resolve_flips(self.st, self.bv, hs, self.s) if self.st is not None else hs

    # ---- which scheme is in Finalize
    def _describe(self, what):
        """'Scheme #1 refined (all, ≤10°)' -> 'scheme #1, refined (all, ≤10°)'; prefixed with 'optimized
        structure, ' for an optimized structure."""
        d = str(what).strip()
        d = re.sub(r"^Scheme (#\d+)", r"scheme \1", d)
        d = re.sub(r"(scheme #\d+) refined", r"\1, refined", d)
        if d[:1].isupper() and not d.startswith("H in CIF"):
            d = d[0].lower() + d[1:]
        if self.st is not None and getattr(self.st, "optimized", None):
            d = "optimized structure, " + d
        if self.st is not None and getattr(self.st, "sub_number", None):
            d = "subscheme %d (ordered, P1), %s" % (self.st.sub_number, d)
        return d

    def _set_fin_source(self, what=None, edited=False):
        if what is not None:
            self._fin_src = self._describe(what)
        src = getattr(self, "_fin_src", "")
        self.fin_source.set(("Finalizing: %s%s" % (src, "  (edited by hand)" if edited else "")) if src else "")

    def _on_tab_changed(self, _e=None):
        """Opening Finalize & CIF with an entry selected in the Refinement comparison (e.g. a refined scheme)
        loads that entry - once per selection, so later hand edits are kept."""
        try:
            if self.nb.index(self.nb.select()) != TAB_FINAL or self.st is None:
                return
        except tk.TclError:
            return
        e = self.selected_cmp()
        if e is not None and id(e) != self._fin_loaded_cmp:
            self._fin_loaded_cmp = id(e)
            self.load_hatoms(e["hs"], e["name"])

    def load_scheme(self, r, switch=True):
        if self.res is None or not self.res.schemes:
            return
        hs = hc.scheme_hatoms(self.res, self.res.schemes[r], include_located=False)
        self.checked = {self.fin_by_key[self._rowkey(h)] for h in hs if self._rowkey(h) in self.fin_by_key}
        self._set_fin_source("scheme #%d" % (r + 1))
        self.refresh_finalize()
        self.fin_status.set("Loaded scheme #%d (%d hypothetical + %d located H). Adjust, then export."
                            % (r + 1, len(hs), sum(1 for h in self.fin_loc.values() if h.primary)))
        if switch:
            self.nb.select(TAB_FINAL)

    def load_hatoms(self, hs, what):
        """Check exactly the inferred H of `hs` (e.g. a refined scheme); H positions not yet listed
        (rotated by refinement) are added as new rows."""
        st = self.st
        checked = set()
        for h in hs:
            if h.located:
                continue
            rk = self._rowkey(h)
            iid = self.fin_by_key.get(rk)
            if iid is None:
                self._fin_n += 1
                iid = "h%d" % self._fin_n
                self.fin_by_key[rk] = iid
                self.fin_rowkey[iid] = rk
                self.fin_h[iid] = h
                self.fin_info[iid] = {"opts": [what]}
                pid = self.fin_parent.get(h.donor)
                if pid is None:
                    pid = self.fin.tree.insert("", "end", text=st.atoms[h.donor].label, open=True, tags=("donor",))
                    self.fin_parent[h.donor] = pid
                kind = ("disordered H (×%.2f)" % h.occf) if getattr(h, "compw", None) else "refined H"
                self.fin.tree.insert(pid, "end", iid=iid, text="%s → %s %s" % (
                    kind, hc.ordinal(h.cand.ordinal), st.atoms[h.cand.acceptor].label))
                if h.cand.passes:
                    self.fin_pass.add(iid)
            checked.add(iid)
        self.checked = checked
        self._set_fin_source(what)
        self.refresh_finalize()
        self.fin_status.set("Loaded %s (%d inferred H). Adjust, then export." % (what, len(checked)))

    def refresh_finalize(self):
        if self.st is None:
            return
        st, s = self.st, self.s
        hs = self.final_hatoms()
        labels = hc.label_hydrogens(st, hs)
        contacts = hc.h_contacts(st, hs, cutoff=s.hh_mod) if hs else []
        worst = {}
        for a, b, dist, _, _ in contacts:
            for h, other in ((a, b), (b, a)):
                if id(h) not in worst or dist < worst[id(h)][0]:
                    worst[id(h)] = (dist, other)

        def flag_for(h):
            if id(h) not in worst:
                return "", None
            dist, other = worst[id(h)]
            return ("%.2f Å to %s" % (dist, labels.get(id(other), "H") + ("" if other.located else "*")),
                    "red" if dist < s.hh_red else "mod")

        flipped = {tuple(x.key[1:]) for x in hs if getattr(x, "flipped", False)}
        sym_by_key = {}                 # H ordered on a special position (images of each other through it)
        for k, (r, _) in hc.h_site_images(st, hs).items():
            img = next(x for x in hs if id(x) == k)
            sym_by_key[tuple(img.key)] = "symmetry image of %s (not written)" % labels.get(id(r), "H")
            sym_by_key.setdefault(tuple(r.key), "ordered on the %s site" % st.atoms[r.donor].label)
        for iid, h in self.fin_loc.items():
            c = h.cand
            flag, tag = flag_for(h)
            tests = "in CIF" if h.primary else "in CIF (H-bond branch)"
            if h.conditional and c is not None:
                tests += "; p %.2f" % h.p
            if tuple(h.key) in flipped:
                tests += "; FLIPS away here"
            self.fin.tree.item(iid, tags=(tag,) if tag else ("located",), values=(
                LOCKED if h.primary else "↳", h.located,
                "%s %s" % (st.atoms[c.acceptor].label, st.symop_code(c.nb.op, c.nb.shift)) if c else "none",
                _fmt(c.nb.d) if c else "", _fmt(h.d_ha), _fmt(h.angle, 1), "located", tests, "", flag))
        eff = self._fin_effective()
        for iid, h in self.fin_h.items():
            h = eff.get(iid, h)                 # ticked: the position actually used
            c = st.atoms[h.cand.acceptor]
            nb = h.cand.nb
            tags = ["pass"] if iid in self.fin_pass else []
            flag = ""
            if iid in self.checked:
                flag, tag = flag_for(h)
                if tag:
                    tags.append(tag)
            copies = hc.h_symmetry_copies(st, h)
            sym = sym_by_key.get(tuple(h.key)) if iid in self.checked else None
            if sym is not None:
                copies = 1                  # ordered on the donor's site: a whole H (or generated by symmetry)
            tests = "pass" if iid in self.fin_pass else "only in options failing A1–D–A2"
            if h.cand.adc_pen:
                tests += "; A–D–C pen. %.1f" % h.cand.adc_pen
            self.fin.tree.item(iid, tags=tuple(tags), values=(
                CHECK if iid in self.checked else UNCHECK, labels.get(id(h), ""),
                "%s %s" % (c.label, st.symop_code(nb.op, nb.shift)), _fmt(nb.d), _fmt(h.d_ha), _fmt(h.angle, 1),
                ", ".join(("#%d" % k) if isinstance(k, int) else str(k) for k in self.fin_info[iid]["opts"]),
                tests,
                "; ".join(x for x in (("occ %.3f" % (st.atoms[h.donor].occ / copies * getattr(h, "occf", 1.0)))
                                      if (copies > 1 or st.atoms[h.donor].occ < 0.999 or
                                          getattr(h, "occf", 1.0) < 0.999) else "",
                                      (("split ×%.2f" if getattr(h, "alt", None) else "H fraction ×%.2f") % h.occf)
                                      if getattr(h, "occf", 1.0) < 0.999 else "",
                                      ("×%d copies" % copies) if copies > 1 else "",
                                      sym or "") if x), flag))
        self._flag_box(self.fin_flag, contacts, labels, hs)
        if self.det_src.get() == "final":
            self.refresh_details()
        txt = hc.h_count_text(st, self.bv, hc.h_per_cell(st, hs))
        ok_fg, bad_fg = getattr(self, "_ok_fg", "#166534"), getattr(self, "_bad_fg", "#b91c1c")
        self.fin_hc.configure(text=txt + "  (each H counts with its parent's occupancy; exported H get that "
                                         "occupancy)", foreground=bad_fg if txt.startswith("⚠") else ok_fg)
        self.fin_bv.fill(st, self.bv, hs, s, self.params, labels)
        self.cif_btn.configure(state="normal" if self.checked else "disabled")
        if self.table_src.get() == "final":
            self.refresh_table()

    def _toggle(self, iids):
        for iid in iids:
            if iid in self.fin_h:
                self.checked.symmetric_difference_update({iid})
        self._set_fin_source(edited=True)
        self.refresh_finalize()

    def fin_click(self, event):
        t = self.fin.tree
        if t.identify_region(event.x, event.y) != "cell" or t.identify_column(event.x) != "#1":
            return None
        row = t.identify_row(event.y)
        if row in self.fin_h:
            self._toggle([row])
            return "break"
        return None

    def fin_double(self, event):
        t = self.fin.tree
        if t.identify_column(event.x) == "#1":
            return "break"
        row = t.identify_row(event.y)
        if row in self.fin_h:
            self._toggle([row])
            return "break"
        return None

    def fin_space(self, event):
        self._toggle(self.fin.tree.selection())
        return "break"

    def check_passing(self):
        self.checked = set(self.fin_pass)
        self._set_fin_source("all passing H (every alternative)")
        self.refresh_finalize()
        self.fin_status.set("Checked all %d hypothetical H that pass the tests (alternatives for the same donor "
                            "are all included - uncheck the ones you don't want)." % len(self.checked))

    def uncheck_all(self):
        self.checked = set()
        self._set_fin_source("located H only (no hypothetical H)")
        self.refresh_finalize()

    def export_cif(self):
        if self.st is None or not self.checked:
            return
        hs = self.final_hatoms()
        contacts = hc.h_contacts(self.st, hs, cutoff=self.s.hh_mod)
        red = [c for c in contacts if c[2] < self.s.hh_red and hc.contact_is_geometry(self.st, c)]
        if red and not messagebox.askyesno(
                "H···H red flags", "%d H···H contact(s) below %.2f Å between different donor O atoms are in the "
                                   "selection (located H included).\n\nExport anyway?" % (len(red), self.s.hh_red)):
            return
        try:
            path = self.write_cif(self.st, self.bv, hs)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("CIF export failed", str(e))
            return
        self.fin_status.set("Wrote %s (%d hypothetical H)." % (path, len(self.checked)))
        self.status_var.set("Wrote %s" % os.path.basename(path))

    # ================================================================== export
    def export(self):
        if self.st is None:
            return
        folder = filedialog.askdirectory(title="Folder for the CSV files")
        if not folder:
            return
        st, bv, res = self.st, self.bv, self.res
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", "%s_%s" % (os.path.splitext(os.path.basename(st.source))[0], st.name))

        def write(name, header, rows):
            path = os.path.join(folder, "%s_%s.csv" % (base, name))
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                w.writerows(rows)

        write("bv_sites", ["Site", "Element", "Role", "Valence", "Occupancy", "BVS", "BVS w/o H", "BVS-|V|",
                           "N summed", "Listed not summed", "D/A class", "Oxidation state from"],
              [[a.label, a.el, bv.roles[i], bv.valence.get(i), a.occ, "%.4f" % bv.bvs[i], "%.4f" % bv.bvs_noH[i],
                "%.4f" % (bv.bvs[i] - bv.ideal[i]) if bv.ideal[i] is not None else "", bv.nsum[i],
                "%.4f" % bv.weak[i], bv.cls[i], bv.how.get(i, "")] for i, a in enumerate(st.atoms)])
        rows = []
        for i, a in enumerate(st.atoms):
            for bd in bv.bonds[i]:
                nb = bd.nb
                rows.append([a.label, st.atoms[nb.j].label, "%.4f" % nb.d, "%.4f" % bd.s, bd.occ,
                             "%.4f" % bd.weighted, self._bond_status(i, bd)[0], st.symop_text(nb.op, nb.shift),
                             st.symop_code(nb.op, nb.shift), bd.param.ro, bd.param.b,
                             bd.param.source])
        write("bv_bonds", ["Site", "Neighbour", "d", "s", "Neighbour occupancy", "s*occ", "Status", "Symmetry op",
                           "Code", "Ro", "B", "Parameter source"], rows)
        crows, orows = [], []
        for d in self.donors:
            for h in d["located"]:
                c = h.cand
                crows.append([d["label"], d["cls"], "%.4f" % d["bvs"], "located " + h.located,
                              st.atoms[c.acceptor].label if c else "", "%.4f" % c.nb.d if c else "", "", "", "",
                              "", "", _fmt(h.s_grg, 4), _fmt(h.s_fi, 4), "located",
                              st.symop_text(c.nb.op, c.nb.shift) if c else "",
                              st.symop_code(c.nb.op, c.nb.shift) if c else ""])
            for c in d["cands"]:
                nb = c.nb
                crows.append([d["label"], d["cls"], "%.4f" % d["bvs"], hc.ordinal(c.ordinal), st.atoms[nb.j].label,
                              "%.4f" % nb.d, _fmt(c.min_adc, 2),
                              "; ".join("%s %.2f" % (st.atoms[cn.j].label, ang) for cn, ang in c.cation_angles),
                              c.cone_ok, c.adc_ok, "%.3f" % c.adc_pen, _fmt(c.s_grg, 4), "%.4f" % c.s_fi,
                              "pass" if c.passes else c.reason, st.symop_text(nb.op, nb.shift),
                              st.symop_code(nb.op, nb.shift)])
            for o in d["options"]:
                orows.append([d["label"], o.number,
                              " + ".join("%s %s" % (hc.ordinal(c.ordinal), st.atoms[c.acceptor].label)
                                         for c in o.cands),
                              _fmt(o.angle, 2), _fmt(o.hdh, 2), "; ".join("%.4f" % h.d_ha for h in o.hs),
                              "; ".join("%.2f" % h.angle for h in o.hs), o.angle_pen, "%.3f" % o.adc_pen,
                              "%.4f" % o.donorF, o.passes])
        write("hbond_candidates", ["Donor", "Class", "Donor BVS w/o H", "Ordinal / located H", "Acceptor", "D...A",
                                   "min A-D-C", "A-D-C per cation", "Outside cone", "A-D-C pass", "A-D-C penalty",
                                   "s GRG", "s F&I", "Result", "Symmetry op", "Code"], crows)
        write("hbond_options", ["Donor", "Option", "H-bonds", "A1-D-A2", "H-D-H", "H...A", "D-H...A",
                                "A1-D-A2 penalty", "A-D-C penalty", "F&I: BVS w/o H - donated", "Passes"], orows)
        write("hbond_schemes", ["Rank", "Total", "BV score F&I", "BV score GRG", "Angle penalty", "H...H penalty",
                                "Red", "Moderate", "max outside band F&I", "max |dev| GRG", "Inferred H-bonds"],
              [[r, "%.3f" % sc["total"], "%.3f" % sc["bvF"], "%.3f" % sc["bvG"], "%.2f" % sc["geom"],
                "%.0f" % sc["hh"], int(sc["red"]), int(sc["mod"]), "%.4f" % sc["maxF"], "%.4f" % sc["maxG"],
                hc.scheme_summary(st, res, sc)] for r, sc in enumerate(res.schemes, 1)])
        write("located_hbonds", ["Donor", "H", "Acceptor", "D-H", "H...A", "D...A", "D-H...A", "Symmetry op",
                                 "Code"],
              [[r["donor"], r["hydrogen"], r["acceptor"], "%.4f" % r["d_DH"], "%.4f" % r["d_HA"],
                "%.4f" % r["d_DA"], "%.2f" % r["angle"], r["symop"], r["code"]] for r in self.located[0]])
        self.status_var.set("Exported 6 CSV files to %s" % folder)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    HBondApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
