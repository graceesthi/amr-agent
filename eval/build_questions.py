#!/usr/bin/env python3
"""Generate eval/questions.json — the RAGAS evaluation set.

    python eval/build_questions.py

Numeric questions and their reference answers are DERIVED FROM
data/surveillance.json rather than typed by hand. This matters: a hand-written
reference that drifts from the data silently poisons context_recall, and the
resulting table would be measuring the question set rather than the retriever.

Narrative questions cite the thematic corpus documents, whose text is fixed in
scripts/build_dataset.py, so their references are stable.

12 questions — above the 10 the rubric asks for, and spread across the two
tool paths (narrative search vs structured lookup) so the evaluation exercises
the whole pipeline rather than one arm of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from surveillance import SurveillanceDB  # noqa: E402

OUT = ROOT / "eval" / "questions.json"


def _series(db, organism, drug, region):
    rows = [
        r for r in db.profile(organism, region=region)
        if r["antibiotic_class"] == drug
    ]
    return sorted(rows, key=lambda r: r["year"])


def build() -> list[dict]:
    db = SurveillanceDB()
    items: list[dict] = []

    # --- numeric / trend questions, references computed from the data ------
    numeric_specs = [
        ("Klebsiella pneumoniae", "carbapenems", "Southern Europe"),
        ("Klebsiella pneumoniae", "carbapenems", "Northern Europe"),
        ("Escherichia coli", "fluoroquinolones", "South-East Asia"),
        ("Staphylococcus aureus", "meticillin", "Southern Europe"),
        ("Enterococcus faecium", "vancomycin", "Eastern Europe"),
        ("Acinetobacter baumannii", "carbapenems", "Southern Europe"),
    ]

    for organism, drug, region in numeric_specs:
        rows = _series(db, organism, drug, region)
        first, last = rows[0], rows[-1]
        delta = round(last["resistance_pct"] - first["resistance_pct"], 1)
        direction = "increased" if delta > 0 else "decreased"
        items.append(
            {
                "question": (
                    f"How has {drug} resistance in {organism} changed in "
                    f"{region} between {first['year']} and {last['year']}?"
                ),
                "reference": (
                    f"{drug.capitalize()} non-susceptibility in {organism} in "
                    f"{region} {direction} from {first['resistance_pct']}% in "
                    f"{first['year']} to {last['resistance_pct']}% in "
                    f"{last['year']}, a change of {delta:+.1f} percentage "
                    f"points. The {last['year']} figure is based on "
                    f"{last['isolates_tested']} tested invasive isolates."
                ),
                "kind": "numeric_trend",
            }
        )

    # --- comparison question ----------------------------------------------
    south = _series(db, "Klebsiella pneumoniae", "carbapenems", "Southern Europe")[-1]
    north = _series(db, "Klebsiella pneumoniae", "carbapenems", "Northern Europe")[-1]
    items.append(
        {
            "question": (
                "Is carbapenem resistance in Klebsiella pneumoniae higher in "
                "Southern Europe or Northern Europe, and by how much?"
            ),
            "reference": (
                f"It is substantially higher in Southern Europe. In "
                f"{south['year']} Southern Europe reported "
                f"{south['resistance_pct']}% carbapenem non-susceptibility "
                f"versus {north['resistance_pct']}% in Northern Europe, a gap "
                f"of about "
                f"{round(south['resistance_pct'] - north['resistance_pct'], 1)} "
                f"percentage points. Between-region comparisons should be "
                f"treated cautiously because the networks differ in breakpoint "
                f"standard and hospital mix."
            ),
            "kind": "comparison",
        }
    )

    # --- narrative questions, references from the thematic documents -------
    items += [
        {
            "question": (
                "Which antimicrobial stewardship interventions had a measurable "
                "effect on carbapenem consumption, and which did not?"
            ),
            "reference": (
                "Prospective audit with direct prescriber feedback reduced "
                "carbapenem defined daily doses by a median of 18% within "
                "twelve months, with the effect attenuating when audit "
                "frequency fell below fortnightly. Preauthorisation produced a "
                "larger median reduction of 27% but caused delayed first doses "
                "in sepsis pathways and substitution toward unrestricted "
                "broad-spectrum agents. Rapid diagnostics shortened time to "
                "identification by a median of 26 hours but only changed "
                "prescribing where paired with an active stewardship response. "
                "Passive education, guideline circulation without enforcement, "
                "and one-off audits showed no measurable effect."
            ),
            "kind": "narrative",
        },
        {
            "question": (
                "Why did reduced carbapenem consumption not translate into "
                "reduced resistance within a year?"
            ),
            "reference": (
                "Resistance responds on a multi-year lag. No reporting site saw "
                "a resistance reduction within the same twelve-month window as "
                "the consumption reduction, which means stewardship programmes "
                "evaluated on one-year resistance endpoints appear to fail even "
                "when they are working."
            ),
            "kind": "narrative",
        },
        {
            "question": (
                "Why are between-region comparisons of resistance percentages "
                "unreliable, and what is more reliable?"
            ),
            "reference": (
                "Networks differ in breakpoint standard (EUCAST versus CLSI), "
                "in the mix of participating hospital types, and in whether "
                "repeat isolates from the same patient are deduplicated. "
                "Testing propensity also varies with health-system capacity, so "
                "the denominator is not the population but the set of isolates "
                "someone chose to culture and test. Differences of a few "
                "percentage points between regions should not be treated as "
                "real. Within-region trends over time are considerably more "
                "reliable because the methodological differences are roughly "
                "constant across years within one network."
            ),
            "kind": "narrative",
        },
        {
            "question": (
                "What is the difference between carbapenemase-mediated and "
                "non-carbapenemase carbapenem resistance, and why does it "
                "matter for the response?"
            ),
            "reference": (
                "Carbapenemase production (KPC, OXA-48-like, NDM, VIM) is "
                "plasmid-borne and therefore horizontally transmissible between "
                "strains and species. Non-carbapenemase mechanisms — chiefly "
                "ESBL or AmpC production combined with porin loss — are not "
                "transmissible in the same way. The two look identical in a "
                "percentage-based surveillance series, but a carbapenemase-"
                "driven rise is an infection-control problem while a porin-loss-"
                "driven rise is a prescribing problem. Percentage-only "
                "surveillance cannot distinguish them."
            ),
            "kind": "narrative",
        },
        {
            "question": (
                "At what local resistance level should an antibiotic stop being "
                "used empirically, and what caveats apply to reading regional "
                "surveillance against that threshold?"
            ),
            "reference": (
                "The usual threshold is roughly 10-20% local non-susceptibility "
                "for the likely pathogen, lower for severe syndromes such as "
                "septic shock and higher for low-stakes syndromes such as "
                "uncomplicated urinary infection. Two caveats: regional "
                "aggregates conceal hospital-level variation, so local "
                "antibiogram data should be preferred where it exists; and "
                "invasive-isolate surveillance oversamples severe disease, so "
                "applying those percentages to an outpatient syndrome "
                "overestimates risk."
            ),
            "kind": "narrative",
        },
        {
            "question": (
                "What does the corpus say about colistin resistance rates in "
                "Acinetobacter baumannii?"
            ),
            "reference": (
                "The corpus contains no colistin resistance data. It records "
                "carbapenem resistance for Acinetobacter baumannii and places "
                "the organism in the critical priority tier, but colistin is "
                "not covered. The correct response is to state the gap rather "
                "than to answer from general knowledge."
            ),
            "kind": "unanswerable",
        },
    ]

    return items


def main() -> None:
    items = build()
    OUT.write_text(
        json.dumps(
            {
                "_note": "Generated by eval/build_questions.py from "
                "data/surveillance.json and the thematic corpus documents. "
                "Regenerate whenever the dataset is rebuilt.",
                "n": len(items),
                "questions": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    kinds: dict[str, int] = {}
    for item in items:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    print(f"Wrote {len(items)} questions → {OUT.relative_to(ROOT)}")
    print(f"Breakdown: {kinds}")


if __name__ == "__main__":
    main()
