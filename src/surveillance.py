"""Structured surveillance record store backing two of the MCP tools.

Deliberately not a database. The dataset is a few hundred rows of
(organism, region, year, drug class, %) and a JSON file loaded into memory is
the honest engineering choice at this size; §6 of REPORT.md states the row
count at which that stops being true.

Organism aliasing lives here because it is a data concern, not a tool concern:
the corpus says "Klebsiella pneumoniae", clinicians say "K. pneumoniae", and
the model will produce either.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from config import ROOT

DATA_FILE = ROOT / "data" / "surveillance.json"


class UnknownOrganism(KeyError):
    pass


class UnknownRegion(KeyError):
    pass


def _canon(name: str) -> str:
    """Normalise an organism string for matching."""
    return re.sub(r"[^a-z]", "", name.lower())


# Common clinical shorthand → binomial. Extend as the corpus grows.
ALIASES = {
    "mrsa": "Staphylococcus aureus",
    "staphaureus": "Staphylococcus aureus",
    "saureus": "Staphylococcus aureus",
    "kpneumoniae": "Klebsiella pneumoniae",
    "klebsiella": "Klebsiella pneumoniae",
    "ecoli": "Escherichia coli",
    "escherichiacoli": "Escherichia coli",
    "abaumannii": "Acinetobacter baumannii",
    "acinetobacter": "Acinetobacter baumannii",
    "paeruginosa": "Pseudomonas aeruginosa",
    "pseudomonas": "Pseudomonas aeruginosa",
    "vre": "Enterococcus faecium",
    "efaecium": "Enterococcus faecium",
}


class SurveillanceDB:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or DATA_FILE)
        self._records: Optional[List[dict]] = None

    # -- loading ------------------------------------------------------------

    @property
    def records(self) -> List[dict]:
        if self._records is None:
            if not self.path.exists():
                raise FileNotFoundError(
                    f"{self.path} not found. Run `python scripts/build_dataset.py` "
                    "or see data/README.md."
                )
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._records = payload["records"]
        return self._records

    def organisms(self) -> List[str]:
        return sorted({r["organism"] for r in self.records})

    def regions(self) -> List[str]:
        return sorted({r["region"] for r in self.records})

    # -- resolution ---------------------------------------------------------

    def _resolve_organism(self, name: str) -> str:
        canon = _canon(name)
        known = {_canon(o): o for o in self.organisms()}

        if canon in known:
            return known[canon]
        if canon in ALIASES and _canon(ALIASES[canon]) in known:
            return known[_canon(ALIASES[canon])]
        # Last resort: unique prefix match ("klebsiellapneu" → the full name).
        partial = [full for c, full in known.items() if c.startswith(canon) or canon in c]
        if len(partial) == 1:
            return partial[0]
        raise UnknownOrganism(name)

    def _resolve_region(self, name: str) -> str:
        known = {r.lower(): r for r in self.regions()}
        if name.lower() in known:
            return known[name.lower()]
        partial = [full for low, full in known.items() if name.lower() in low]
        if len(partial) == 1:
            return partial[0]
        raise UnknownRegion(f"Unknown region '{name}'.")

    # -- queries ------------------------------------------------------------

    def profile(
        self,
        organism: str,
        region: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[dict]:
        target = self._resolve_organism(organism)
        rows = [r for r in self.records if r["organism"] == target]

        if region:
            target_region = self._resolve_region(region)
            rows = [r for r in rows if r["region"] == target_region]
        if year:
            rows = [r for r in rows if r["year"] == int(year)]

        return sorted(rows, key=lambda r: (r["region"], r["antibiotic_class"], r["year"]))

    def compare(
        self,
        organism: str,
        regions: List[str],
        antibiotic_class: Optional[str] = None,
    ) -> dict:
        target = self._resolve_organism(organism)
        resolved = [self._resolve_region(r) for r in regions]

        rows = [r for r in self.records if r["organism"] == target]
        if antibiotic_class:
            needle = antibiotic_class.lower()
            rows = [r for r in rows if needle in r["antibiotic_class"].lower()]

        by_region: Dict[str, List[dict]] = {r: [] for r in resolved}
        for row in rows:
            if row["region"] in by_region:
                by_region[row["region"]].append(row)

        # Align on years present in EVERY region, so a difference is never an
        # artefact of comparing different periods.
        year_sets = [
            {row["year"] for row in rs} for rs in by_region.values() if rs
        ]
        aligned = sorted(set.intersection(*year_sets)) if year_sets else []

        comparison: Dict[str, dict] = {}
        for region, rs in by_region.items():
            usable = [r for r in rs if not aligned or r["year"] in aligned]
            if not usable:
                comparison[region] = {"note": "no records for this filter"}
                continue
            by_year = sorted(usable, key=lambda r: r["year"])
            latest = by_year[-1]
            earliest = by_year[0]
            comparison[region] = {
                "years": sorted({r["year"] for r in usable}),
                "latest_year": latest["year"],
                "latest_pct": latest["resistance_pct"],
                "mean_pct": round(
                    sum(r["resistance_pct"] for r in usable) / len(usable), 2
                ),
                "trend_pct_points": round(
                    latest["resistance_pct"] - earliest["resistance_pct"], 2
                ),
                "n_records": len(usable),
            }

        # Widest gap, computed only on the aligned latest year.
        scored = [
            (region, data["latest_pct"])
            for region, data in comparison.items()
            if "latest_pct" in data
        ]
        widest = None
        if len(scored) >= 2:
            scored.sort(key=lambda kv: kv[1])
            low, high = scored[0], scored[-1]
            widest = {
                "between": [high[0], low[0]],
                "pct_points": round(high[1] - low[1], 2),
                "higher": high[0],
            }

        return {
            "organism": target,
            "antibiotic_class": antibiotic_class or "all recorded classes",
            "aligned_years": aligned,
            "comparison": comparison,
            "widest_gap": widest,
            "caveat": "Regions use different reporting networks; absolute "
            "levels are not strictly commensurable. Trends within a region are "
            "more reliable than differences between regions.",
        }
