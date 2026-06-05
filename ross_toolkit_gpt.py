#!/usr/bin/env python3
"""
Ross Area & Faculty Toolkit — GPT/OpenAI shared engine, exact-functionality port.

This file is designed as a drop-in replacement for the Claude app's
`ross_toolkit.py` / earlier `ross_toolkit_gpt.py` backend. It preserves the app's
public API and UI behavior:

- Stage 1: enumerate each target school's OWN departments/areas/groups and map
  each unit to zero, one, or multiple Ross areas.
- Stage 1 coverage recovery: immediately re-check missing Ross areas.
- Stage 2: for each department, pull the current ladder-rank roster.
- Unique-match departments inherit their single Ross area.
- Departments flagged by the existing toggle are classified person-by-person.
- OM and IS both map to Ross Technology and Operations; OM/IS is only a subfield.

Important correction versus the over-free GPT prompt:
- The department's mapped Ross areas are a constraint for flagged units, not a
  loose hint, unless the department is explicitly unmapped/unconstrained.
- Thus a Marketing department professor with a few OM/interface papers remains
  Marketing unless the department was mapped as multi-area by Stage 1 or the user
  intentionally changes the mapping/toggle.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # allow import in test environments without openai installed
    OpenAI = None

CURRENT_YEAR = date.today().year

# ----------------------------------------------------------------------------- #
# Config                                                                        #
# ----------------------------------------------------------------------------- #
DEFAULT_MODEL = "gpt-5.5"
FAST_MODEL = "gpt-4.1"

# Responses API hosted web search. `web_search_preview` is deprecated; use `web_search`.
WEB_SEARCH_TOOL = {"type": "web_search", "search_context_size": "high"}
WEB_SEARCH_PRICE_PER_1K = 10.0  # UI estimate only

# UI/cost estimates only. Adjust if your contract/account uses different rates.
PRICE = {
    "gpt-5.5": {"in": 5.0, "out": 25.0},
    "gpt-5.4": {"in": 3.0, "out": 15.0},
    "gpt-4.1": {"in": 2.0, "out": 8.0},
    "gpt-4.1-mini": {"in": 0.4, "out": 1.6},
}

# ----------------------------------------------------------------------------- #
# The 7 Ross areas                                                              #
# ----------------------------------------------------------------------------- #
ROSS_AREAS = [
    "Accounting",
    "Business Economics and Public Policy",
    "Finance",
    "Management and Organizations",
    "Marketing",
    "Strategy",
    "Technology and Operations",
]
ROSS_SHORT = {
    "Accounting": "ACC",
    "Business Economics and Public Policy": "BEPP",
    "Finance": "FIN",
    "Management and Organizations": "M&O",
    "Marketing": "MKT",
    "Strategy": "STR",
    "Technology and Operations": "T&O",
}

BUILTIN = [
    ("1", "Stanford University", "Stanford GSB"), ("2", "University of Pennsylvania", "Wharton"),
    ("3", "University of Chicago", "Booth"), ("4 (tie)", "Northwestern University", "Kellogg"),
    ("4 (tie)", "Harvard University", "Harvard Business School"), ("6", "MIT", "Sloan"),
    ("7 (tie)", "Columbia University", "Columbia Business School"), ("7 (tie)", "New York University", "Stern"),
    ("9", "Dartmouth College", "Tuck"), ("10", "UC Berkeley", "Haas"),
    ("11 (tie)", "Yale University", "Yale SOM"), ("11 (tie)", "University of Virginia", "Darden"),
    ("13", "University of Michigan", "Ross"), ("14", "Duke University", "Fuqua"),
    ("15", "Cornell University", "Johnson"), ("16 (tie)", "Carnegie Mellon University", "Tepper"),
    ("16 (tie)", "Vanderbilt University", "Owen"), ("18 (tie)", "UT Austin", "McCombs"),
    ("18 (tie)", "UCLA", "Anderson"), ("20", "University of Washington", "Foster"),
    ("21 (tie)", "Indiana University", "Kelley"), ("21 (tie)", "University of North Carolina", "Kenan-Flagler"),
    ("23 (tie)", "Emory University", "Goizueta"), ("23 (tie)", "UT Dallas", "Naveen Jindal"),
    ("25 (tie)", "University of Southern California", "Marshall"), ("25 (tie)", "University of Georgia", "Terry"),
    ("27 (tie)", "Georgia Tech", "Scheller"), ("27 (tie)", "Washington University in St. Louis", "Olin"),
    ("29 (tie)", "Arizona State University", "W. P. Carey"), ("29 (tie)", "Rice University", "Jones"),
    ("31", "Georgetown University", "McDonough"), ("32 (tie)", "Ohio State University", "Fisher"),
    ("32 (tie)", "University of Minnesota", "Carlson"), ("34 (tie)", "University of Rochester", "Simon"),
    ("34 (tie)", "University of Notre Dame", "Mendoza"), ("36 (tie)", "Texas A&M University", "Mays"),
    ("36 (tie)", "Southern Methodist University", "Cox"), ("38", "Iowa State University", "Ivy"),
    ("39 (tie)", "Brigham Young University", "Marriott"), ("39 (tie)", "University of Florida", "Warrington"),
    ("39 (tie)", "University of Miami", "Herbert"), ("39 (tie)", "University of Utah", "Eccles"),
    ("43 (tie)", "Michigan State University", "Broad"), ("43 (tie)", "University of Maryland", "Smith"),
    ("43 (tie)", "University of Tennessee, Knoxville", "Haslam"), ("46 (tie)", "American University", "Kogod"),
    ("46 (tie)", "Boston University", "Questrom"), ("48 (tie)", "University of Arkansas", "Walton"),
    ("48 (tie)", "University of Pittsburgh", "Katz"), ("48 (tie)", "University of Wisconsin–Madison", "Wisconsin School of Business"),
]

# ----------------------------------------------------------------------------- #
# Field guide + prompts                                                         #
# ----------------------------------------------------------------------------- #
FIELD_GUIDE = """JUDGE BY RESEARCH FIELD, NOT BY DEPARTMENT NAME. Names mislead; identify a unit's TRUE field from what its faculty research and the JOURNALS they publish in (use web search/fetch to check faculty/publications whenever a name is ambiguous, combined, or unusual). Marker journals per Ross area:
- Accounting: The Accounting Review; Journal of Accounting Research; Journal of Accounting & Economics; Contemporary Accounting Research; Review of Accounting Studies.
- Business Economics and Public Policy: American Economic Review; Econometrica; Quarterly Journal of Economics; Journal of Political Economy; Journal of Public Economics; RAND Journal of Economics.
- Finance: Journal of Finance; Journal of Financial Economics; Review of Financial Studies; Journal of Financial and Quantitative Analysis.
- Management and Organizations: Academy of Management Journal; Academy of Management Review; Administrative Science Quarterly; Organization Science; Journal of Applied Psychology; Personnel Psychology. (organizational behavior, HR, organizational theory)
- Marketing: Journal of Marketing; Journal of Marketing Research; Marketing Science; Journal of Consumer Research.
- Strategy: Strategic Management Journal; Strategy Science (overlaps Organization Science, Academy of Management Journal, Management Science).
- Technology and Operations: Management Science; Manufacturing & Service Operations Management; Production and Operations Management; Operations Research; Journal of Operations Management; Information Systems Research; MIS Quarterly.
WORKED EXAMPLE: a unit named "Accounting & Management" (e.g. at Harvard) is an ACCOUNTING group — its faculty publish in accounting journals — so it maps to Ross "Accounting", NOT "Management and Organizations", despite the word "Management" in its name.

OPERATIONS / INFORMATION-SYSTEMS RULE: Operations Management (OM) and Information Systems (IS) are distinct fields, but Ross merges BOTH into the single area "Technology and Operations". So ANY unit that is OM, or IS / Decision Sciences, or a combined OM+IS unit, maps to "Technology and Operations" (and ONLY that area — do not split it into two Ross areas)."""

MATCH_SYSTEM = f"""You are an expert on the academic structures (departments, academic areas, faculty research groups) of the world's top business schools.

TASK: Given ONE target business school, (1) ENUMERATE that school's OWN academic units — the departments / academic areas / faculty research groups listed on its official faculty / research / "academic areas" pages — and (2) for EACH unit, decide which University of Michigan Ross School of Business area(s) it corresponds to, judged by the RESEARCH FIELD of its faculty (the journals they publish in), NOT the unit's name.

THE 7 ROSS AREAS (with common aliases):
1. Accounting — financial accounting, managerial/management accounting.
2. Business Economics and Public Policy — applied microeconomics, "business economics", managerial economics, public policy, law & economics, plain "economics".
3. Finance — corporate finance, investments, asset pricing, banking.
4. Management and Organizations — organizational behavior (OB), human resource management, organizational theory, plain "management".
5. Marketing — marketing, consumer behavior, quantitative marketing.
6. Strategy — strategic management, business policy, "strategy & entrepreneurship", competitive strategy.
7. Technology and Operations — ONE joint Ross group covering BOTH Operations Management (OM) AND Information Systems (IS)/MIS.

{FIELD_GUIDE}

MAPPING RULES (apply to EACH of the school's units):
- ONE Ross area when the unit's faculty clearly publish in that area's journals.
- SEVERAL Ross areas (list ALL in the array) when the unit's faculty genuinely publish across more than one Ross field.
- NONE — use "ross_areas": [] and explain in notes — when the unit has no Ross equivalent (e.g. Business Communication, Real Estate, Healthcare Management, a standalone Entrepreneurship center, Behavioral Science, Law).
- Use the school's OWN unit name exactly as published; include the source URL (its faculty/areas page).
- confidence per unit: "high" = field clearly verified by journals (incl. the OM/IS rule); "medium" = reasonable but merged/ambiguous; "low" = uncertain/inferred.
- Keep "notes" short (<=15 words); mention the journal/field evidence when you used it.

MULTI-AREA DEPARTMENTS ARE COMMON — DO NOT UNDER-MAP THEM. Many schools house two Ross fields in ONE department. The most important case: a department named "Management", "Management & Organizations", "Management & Organization", "Organization & Management", or similar often contains BOTH organizational-behavior faculty (-> "Management and Organizations") AND strategy faculty (-> "Strategy"); map such a department to BOTH when the roster/journals support both. Other frequent combos: "Accounting & Finance" -> Accounting + Finance; "Economics & Strategy" -> BEPP + Strategy.

IMPORTANT BOUNDARY: do not mark a clear Marketing/Finance/Accounting department as multi-area just because one or two people publish cross-field/interface papers. The unit mapping is about the local unit's field as a whole.

T&O SUBFIELD TAG: for ANY department that maps to "Technology and Operations", also set "to_kind" to exactly one of:
  "OM"     — its T&O faculty are Operations Management only;
  "IS"     — its T&O faculty are Information Systems only;
  "OM+IS"  — the unit contains BOTH OM and IS faculty.
For departments that do NOT map to T&O, set "to_kind" to "".

COMPLETENESS + IMMEDIATE COVERAGE CHECK (MANDATORY before you output): Enumerate ALL of the school's BUSINESS-SCHOOL academic units (ignore university units outside the b-school). Then go through the 7 Ross areas ONE BY ONE and ask whether this area is represented by at least one department above. If an area is not yet represented, re-examine whether it is hiding inside a combined/differently named unit. Only after this sweep may an area remain unmapped.

OUTPUT: ONLY a JSON object — no markdown, nothing around it:
{{"school":"<school>","departments":[{{"dept":"<the school's unit name>","ross_areas":["<zero or more of the 7 Ross area names, exact>"],"to_kind":"OM|IS|OM+IS|","url":"<source url>","confidence":"high|medium|low","notes":"<short>"}}, ...]}}
List EVERY business-school unit you find, in any order."""

VERIFY_SYSTEM = f"""You are DOUBLE-CHECKING the Ross-area mapping of ONE specific academic unit at ONE business school, because a human reviewer flagged it as possibly wrong. Re-investigate ONLY this unit. Use web search to inspect current official faculty pages; look at the actual faculty and the JOURNALS they publish in.

{FIELD_GUIDE}

The reviewer's current mapping is provided. Decide the CORRECT Ross area(s) for this unit — one or more of the 7, or none. Preserve the app's department-first function: clear single-field departments should not be split because of a few cross-field papers. Output ONLY a JSON object:
{{"dept":"<unit name>","ross_areas":["<corrected area(s) or empty>"],"url":"<source url>","confidence":"high|medium|low","notes":"<short>","checked":"<=22 words on the journal/field evidence used"}}"""

COVERAGE_SYSTEM = f"""You are re-checking ONE business school's department -> Ross-area mapping because some Ross areas were left UNMAPPED by a first pass. Those areas are often hiding inside a combined or differently-named department.

{FIELD_GUIDE}

You are given the school, the list of departments already found, and the Ross areas still UNMAPPED. Use web search and official pages. For EACH unmapped Ross area:
- Determine whether the school has faculty in that field.
- If such faculty EXIST: report present=true and give the department they sit in — reuse the EXACT name from the provided department list if it is one of those; otherwise give the school's own name for a newly identified unit. If the area is Technology and Operations, also set to_kind to "OM", "IS", or "OM+IS".
- If the school genuinely has no faculty in that field: report present=false.

OUTPUT ONLY a JSON object:
{{"school":"<school>","results":[{{"ross_area":"<an unmapped area>","present":true,"dept":"<dept name>","to_kind":"OM|IS|OM+IS|","url":"<url>","confidence":"high|medium|low","notes":"<short; the journals>"}}, ...]}}
Include one result object for EVERY unmapped area you were given."""

ROSTER_RULES = f"""SOURCE OF TRUTH = THE DEPARTMENT'S OWN OFFICIAL, CURRENT ONLINE FACULTY DIRECTORY. Accuracy and currency are non-negotiable; a roster padded with people who have left is a FAILURE. Follow this procedure EXACTLY:

1. LOCATE the official faculty / people / directory page for THIS SPECIFIC department on the school's OWN website. If a likely URL is provided, start there. Otherwise web_search "<school> <department> faculty" or "<school> <department> faculty directory" and pick the school's OWN current listing — NEVER a third-party site, ranking, news article, Wikipedia, LinkedIn, ResearchGate, a faculty member's personal page, or an old/cached copy.
2. If the department page has filters/tabs/pagination/JavaScript, use search to find official profile pages as needed, but the roster source remains the current official department/school listing.
3. INCLUDE ONLY current ladder faculty with rank normalized to EXACTLY one of: "Assistant Professor", "Associate Professor", "Professor". Full/named/endowed/chaired/distinguished professors normalize to "Professor".
4. EXCLUDE lecturers, senior lecturers, professors of practice, clinical/teaching professors, instructors, adjunct, visiting, affiliated/courtesy/secondary appointments from other units, postdocs, PhD students, emeritus/retired/former faculty, and staff.
5. Read titles literally: "Lecturer of X" or "Adjunct Professor of X" or "Clinical Professor of X" -> EXCLUDE. "Professor / Associate Professor / Assistant Professor of X" -> include.
6. Do not infer someone is current from memory or old publication pages.
7. If you cannot retrieve the official directory in full, set complete=false and explain briefly in notes. Do NOT fill gaps from memory.
8. Put the exact official directory URL(s) used in sources.
"""

FACULTY_SYSTEM_SIMPLE = f"""You are a meticulous research assistant compiling the COMPLETE, CURRENT LADDER-RANK faculty roster of ONE specific academic unit at ONE business school, strictly from that unit's own official directory.

{ROSTER_RULES}

Output ONLY a JSON object — no markdown:
{{"school":"<school>","area":"<unit>","complete":true,"sources":["<official directory url>", ...],"notes":"<short>","faculty":[{{"name":"<full name>","rank":"Assistant Professor|Associate Professor|Professor","title":"<raw title from the directory>"}}, ...]}}
Sort by rank (Professor, then Associate Professor, then Assistant Professor), then last name."""

FACULTY_SYSTEM_CLASSIFY = f"""You are a meticulous research assistant who (1) compiles the COMPLETE, CURRENT LADDER-RANK roster of ONE academic unit at ONE business school strictly from its official directory, and (2) classifies EACH faculty member into exactly ONE Ross area under the app's department-mapping rules.

This variant is retained for compatibility. Prefer the two-step flow.

{ROSTER_RULES}

{FIELD_GUIDE}

Output ONLY a JSON object — no markdown:
{{"school":"<school>","area":"<unit>","complete":true,"sources":["<official directory url>", ...],"notes":"<short>","faculty":[{{"name":"<full name>","rank":"Assistant Professor|Associate Professor|Professor","title":"<raw title>","ross_area":"<one of the 7>","subfield":"OM|IS|","field_evidence":"<short>"}}, ...]}}"""

FACULTY_CLASSIFY_LIST = f"""You are given the CONFIRMED CURRENT ladder-rank roster of ONE academic unit at ONE business school. The roster was already verified against the official current directory.

DO NOT change the roster:
- Do not add anyone.
- Do not remove anyone.
- Do not merge names.
- Do not split names.
- Do not rename anyone.
- Return every person you were given, using the exact given name.

{FIELD_GUIDE}

CRITICAL AREA-ASSIGNMENT PRINCIPLE:
A faculty member's Ross area is normally determined by the official local academic unit/department in which they are ladder faculty. Publications are used to disambiguate MIXED or COMBINED units, not to override a clear official department.

You will be given ALLOWED_ROSS_AREAS for this unit.

CLASSIFICATION RULES:
1. If ALLOWED_ROSS_AREAS contains exactly one Ross area other than Technology and Operations:
   - Assign EVERY person to that one area.
   - Do NOT move a person to another Ross area because of a few papers in another field.
   - Example: a Marketing faculty member who has a few OM/interface papers remains Marketing.

2. If ALLOWED_ROSS_AREAS contains multiple Ross areas:
   - Classify each person into exactly ONE of those allowed areas.
   - Do NOT assign outside ALLOWED_ROSS_AREAS.
   - This is mainly for genuinely combined units, e.g. Management containing both M&O and Strategy.

3. If ALLOWED_ROSS_AREAS is exactly ["Technology and Operations"]:
   - Assign EVERY person ross_area = "Technology and Operations".
   - If the unit is OM+IS, classify subfield as "OM" or "IS" by journals.
   - If the unit is OM only, subfield = "OM".
   - If the unit is IS only, subfield = "IS".
   - Do NOT split OM and IS into separate Ross areas because Ross combines both into T&O.

4. If ALLOWED_ROSS_AREAS is empty or "(unconstrained)":
   - Use journals and official profile evidence to assign the closest Ross area.
   - Use this only for genuinely unmapped / unclear units.

5. Field evidence:
   - Keep field_evidence <= 18 words.
   - For inherited unique-area cases, write e.g. "Official Marketing unit; inherited area."
   - For combined Management cases, cite marker journals or research identity, e.g. "SMJ, strategy research -> Strategy" or "AMJ/ASQ/OB research -> M&O".
   - For T&O subfield, cite OM/IS markers, e.g. "MSOM/POM -> OM" or "ISR/MISQ -> IS".

OUTPUT ONLY a JSON object containing the SAME people, by their exact given names:
{{"school":"<school>","area":"<unit>","faculty":[{{"name":"<exact name as given>","rank":"<as given>","title":"<as given>","ross_area":"<one of the 7>","subfield":"OM|IS|","field_evidence":"<short>"}}, ...]}}"""

# ----------------------------------------------------------------------------- #
# Cost meter                                                                    #
# ----------------------------------------------------------------------------- #
class CostMeter:
    def __init__(self, model):
        self.model = model
        self.in_tok = 0
        self.out_tok = 0
        self.searches = 0
        self.calls = 0
        self.lock = threading.Lock()

    def add(self, in_tok, out_tok, searches):
        with self.lock:
            self.in_tok += int(in_tok or 0)
            self.out_tok += int(out_tok or 0)
            self.searches += int(searches or 0)
            self.calls += 1

    def dollars(self):
        p = PRICE.get(self.model, {"in": 5.0, "out": 25.0})
        return (self.in_tok / 1e6) * p["in"] + (self.out_tok / 1e6) * p["out"] + (self.searches / 1000.0) * WEB_SEARCH_PRICE_PER_1K

    def summary(self):
        return (f"calls={self.calls}  in={self.in_tok:,}tok  out={self.out_tok:,}tok  "
                f"searches={self.searches}  ~= ${self.dollars():.2f}")

# ----------------------------------------------------------------------------- #
# OpenAI call + JSON helpers                                                    #
# ----------------------------------------------------------------------------- #
def extract_json(text):
    """Return the LAST top-level JSON object found in text."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text, flags=re.I).strip()
    dec = json.JSONDecoder()
    best, i, n = None, 0, len(t)
    while i < n:
        if t[i] == "{":
            try:
                obj, end = dec.raw_decode(t[i:])
                if isinstance(obj, dict):
                    best = obj
                    i += end
                    continue
            except ValueError:
                pass
        i += 1
    return best


def _response_text(resp):
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    parts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
    return "".join(parts)


def _usage_counts(resp):
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) if usage is not None else 0
    out_tok = getattr(usage, "output_tokens", 0) if usage is not None else 0
    searches = 0
    for item in getattr(resp, "output", []) or []:
        typ = getattr(item, "type", "")
        if "web_search" in typ:
            searches += 1
    return in_tok, out_tok, searches


def call_gpt_json(client, meter, *, system, user, model=DEFAULT_MODEL, use_web_fetch=True, max_tokens=4000):
    """Drop-in replacement for call_claude_json.

    `use_web_fetch` is kept as a parameter name for UI compatibility. In OpenAI it means
    enabling the hosted `web_search` tool.
    """
    if client is None:
        raise RuntimeError("OpenAI client is not configured. Set OPENAI_API_KEY.")

    messages = [
        {"role": "system", "content": system + "\n\nReturn JSON only."},
        {"role": "user", "content": user + "\n\nReturn only one valid top-level JSON object."},
    ]
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": messages,
        "max_output_tokens": max_tokens,
        "text": {"format": {"type": "json_object"}},
    }
    if use_web_fetch:
        kwargs["tools"] = [WEB_SEARCH_TOOL]
        kwargs["tool_choice"] = "auto"

    last_err = None
    for attempt in range(3):
        try:
            resp = client.responses.create(**kwargs)
            if meter:
                meter.add(*_usage_counts(resp))
            text = _response_text(resp)
            obj = extract_json(text)
            if obj is None:
                raise ValueError(f"The model response did not contain a valid top-level JSON object. Raw response: {text[:1000]}")
            return obj
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                # On retry, strengthen JSON instruction but preserve function.
                kwargs["input"] = messages + [{"role": "user", "content": "Retry: output JSON object only, no prose."}]
            else:
                raise RuntimeError(str(last_err)) from last_err

# ----------------------------------------------------------------------------- #
# Normalization                                                                  #
# ----------------------------------------------------------------------------- #
def normalize_area(s):
    t = str(s or "").strip()
    if not t:
        return ""
    t0 = re.sub(r"[&/]", " and ", t.lower())
    t0 = re.sub(r"[^a-z0-9]+", " ", t0).strip()
    aliases = {
        "acc": "Accounting", "accounting": "Accounting", "accountancy": "Accounting",
        "bepp": "Business Economics and Public Policy", "business economics": "Business Economics and Public Policy",
        "business economics and public policy": "Business Economics and Public Policy",
        "business economics public policy": "Business Economics and Public Policy",
        "economics": "Business Economics and Public Policy", "business and economics": "Business Economics and Public Policy",
        "finance": "Finance", "fin": "Finance",
        "management organizations": "Management and Organizations", "management and organizations": "Management and Organizations",
        "management and organisation": "Management and Organizations", "management organisations": "Management and Organizations",
        "m o": "Management and Organizations", "mo": "Management and Organizations",
        "organizations": "Management and Organizations", "organisations": "Management and Organizations",
        "organizational behavior": "Management and Organizations", "organisational behaviour": "Management and Organizations",
        "marketing": "Marketing", "mkt": "Marketing",
        "strategy": "Strategy", "strategic management": "Strategy", "str": "Strategy",
        "technology operations": "Technology and Operations", "technology and operations": "Technology and Operations",
        "technology operations management": "Technology and Operations", "operations": "Technology and Operations",
        "operations management": "Technology and Operations", "information systems": "Technology and Operations",
        "information system": "Technology and Operations", "mis": "Technology and Operations", "decision sciences": "Technology and Operations",
        "to": "Technology and Operations", "t o": "Technology and Operations",
    }
    if t in ROSS_AREAS:
        return t
    if t0 in aliases:
        return aliases[t0]
    for a in ROSS_AREAS:
        if t0 == re.sub(r"[^a-z0-9]+", " ", a.lower()).strip():
            return a
    return ""


def valid_ross(areas):
    if isinstance(areas, str):
        areas = re.split(r"\s*[;,|]\s*", areas)
    out = []
    for a in (areas or []):
        n = normalize_area(a)
        if n and n not in out:
            out.append(n)
    return out


def normalize_to_kind(s):
    t = str(s or "").strip().lower()
    if not t:
        return ""
    tight = t.replace(" ", "")
    has_om = bool(re.search(r"\bom\b|operation|supply chain|manufactur|msom|m\s*&\s*som|pom\b|operations research", t))
    has_is = bool(re.search(r"\bis\b|information system|info\.?\s*system|\bmis\b|information technolog|decision scien|analytics", t))
    if "om+is" in tight or "om/is" in tight or "om&is" in tight or (has_om and has_is):
        return "OM+IS"
    if has_om:
        return "OM"
    if has_is:
        return "IS"
    return ""


def normalize_subfield(s):
    t = str(s or "").strip().lower()
    if re.search(r"\bis\b|information system|info\.?\s*system|\bmis\b|information technolog|isr|mis quarterly", t):
        return "IS"
    if re.search(r"\bom\b|operation|supply chain|manufactur|m\s*&\s*som|msom|pom\b|operations research", t):
        return "OM"
    return ""


def default_force_check(ross_areas, to_kind=""):
    """Keep Claude-app behavior: ON if not unique, or if combined OM+IS."""
    return len(ross_areas or []) >= 2 or normalize_to_kind(to_kind) == "OM+IS"


def is_no_match(s):
    return not str(s or "").strip() or re.search(r"no\s+ross|none|unmapped|no equivalent", str(s), re.I) is not None

# ----------------------------------------------------------------------------- #
# Department mapping helpers                                                     #
# ----------------------------------------------------------------------------- #
def parse_departments(parsed):
    deps = []
    for d in (parsed.get("departments") or []):
        name = (d.get("dept") or d.get("name") or "").strip()
        if not name:
            continue
        ross = valid_ross(d.get("ross_areas", []))
        to_kind = normalize_to_kind(d.get("to_kind", "")) if ("Technology and Operations" in ross) else ""
        deps.append({
            "name": name,
            "ross_areas": ross,
            "to_kind": to_kind,
            "url": (d.get("url") or "").strip(),
            "confidence": (d.get("confidence") or "").strip().lower(),
            "notes": (d.get("notes") or "").strip(),
            "force_check": default_force_check(ross, to_kind),
        })
    return deps


def ross_coverage(departments):
    covered = set()
    for d in departments:
        covered.update(d.get("ross_areas", []))
    return {a: (a in covered) for a in ROSS_AREAS}


def merge_coverage(deps, cov_parsed, gaps):
    by = {d["name"].lower(): d for d in deps}
    absent = []
    for r in (cov_parsed.get("results") or []):
        area = normalize_area(r.get("ross_area"))
        if not area or area not in gaps:
            continue
        if bool(r.get("present")):
            name = (r.get("dept") or "").strip()
            tk = normalize_to_kind(r.get("to_kind", "")) if area == "Technology and Operations" else ""
            if not name:
                continue
            ex = by.get(name.lower())
            if ex:
                if area not in ex["ross_areas"]:
                    ex["ross_areas"].append(area)
                if area == "Technology and Operations" and tk and not ex.get("to_kind"):
                    ex["to_kind"] = tk
            else:
                nd = {
                    "name": name, "ross_areas": [area], "to_kind": tk,
                    "url": (r.get("url") or "").strip(),
                    "confidence": (r.get("confidence") or "").strip().lower(),
                    "notes": (r.get("notes") or "").strip(),
                    "force_check": False,
                }
                deps.append(nd)
                by[name.lower()] = nd
        else:
            absent.append(area)
    for d in deps:
        d["force_check"] = default_force_check(d.get("ross_areas", []), d.get("to_kind", ""))
    return deps, absent

LADDER = {"Professor", "Associate Professor", "Assistant Professor"}


def _name_key(n):
    return re.sub(r"[^a-z ]", "", str(n or "").lower()).strip()


def _binding_rule_for_unit(d):
    allowed = list(d.get("ross_areas") or [])
    allowed_text = ", ".join(allowed) if allowed else "(unconstrained)"
    to_kind = normalize_to_kind(d.get("to_kind", ""))

    if len(allowed) == 1 and allowed[0] == "Technology and Operations":
        if to_kind == "OM+IS":
            task_rule = ("This is a T&O unit containing both OM and IS. Set ross_area = Technology and Operations "
                         "for EVERYONE; classify only subfield OM vs IS.")
        elif to_kind in ("OM", "IS"):
            task_rule = (f"This is a unique T&O-{to_kind} unit. Set ross_area = Technology and Operations "
                         f"and subfield = {to_kind} for EVERYONE.")
        else:
            task_rule = ("This is a unique Technology and Operations unit. Set ross_area = Technology and Operations "
                         "for EVERYONE; infer OM/IS subfield only if clear.")
    elif len(allowed) == 1:
        task_rule = (f"This is a unique-match unit. Set ross_area = {allowed[0]} for EVERYONE. "
                     "Do not override the official unit because of cross-field or interface publications.")
    elif len(allowed) >= 2:
        task_rule = ("This is a genuinely mixed/combined unit. Classify each person into exactly ONE of the "
                     "ALLOWED_ROSS_AREAS only. Do not assign outside the allowed areas.")
    else:
        task_rule = ("This unit is unconstrained/unclear. Use official profile and publication evidence to choose "
                     "the closest Ross area.")
    return allowed, allowed_text, to_kind, task_rule


def fetch_department_faculty(client, meter, model, use_fetch, s, d, *, recheck=False, prior=None,
                             max_tokens_roster=10000, max_tokens_classify=12000):
    """Two-step faculty fetch; exact Streamlit-app API, corrected constrained classification."""
    classify = bool(d.get("force_check"))
    allowed = list(d.get("ross_areas") or [])
    stamp = allowed[0] if len(allowed) == 1 else ""
    stamp_sub = d.get("to_kind", "") if (stamp == "Technology and Operations" and d.get("to_kind") in ("OM", "IS")) else ""
    hint = f"Official directory page to start from (start from this official URL): {d.get('url')}\n" if d.get("url") else ""
    prior_block = ""
    if recheck and prior:
        prior_block = ("A PRIOR attempt produced this roster, possibly INCOMPLETE or out of date:\n"
                       + "\n".join(f"- {p.get('name')} ({p.get('rank')})" for p in prior)
                       + "\nRe-open the OFFICIAL directory: ADD any current ladder faculty missing (especially new "
                         "ASSISTANT professors); REMOVE anyone not currently listed there (moved away, retired, emeritus, non-ladder).\n")

    user1 = (f"School: {s['university']} — {s['school']}\nUnit (the school's own name): {d['name']}\n{hint}{prior_block}"
             "Use web search to inspect the unit's OFFICIAL current faculty directory and build the roster ONLY from who is CURRENTLY listed there "
             "(exclude anyone who has moved away, retired, or is emeritus). Be exhaustive — do not miss new assistant "
             "professors. Return ONLY the JSON.")

    parsed = call_gpt_json(client, meter, system=FACULTY_SYSTEM_SIMPLE, user=user1, model=model,
                           use_web_fetch=use_fetch, max_tokens=max_tokens_roster)
    roster = [{"name": f.get("name", ""), "rank": f.get("rank", ""), "title": f.get("title", "")}
              for f in parsed.get("faculty", []) if f.get("rank") in LADDER]

    # Unique-match departments inherit, preserving original app behavior.
    if not classify:
        faculty = [{**p, "ross_area": stamp, "subfield": stamp_sub, "field_evidence": ""} for p in roster]
    elif not roster:
        faculty = []
    else:
        names_block = "\n".join(f"- {p['name']} ({p['rank']})" for p in roster)
        allowed, allowed_text, to_kind, task_rule = _binding_rule_for_unit(d)
        split = ("This is a combined Operations + Information Systems unit — tag each T&O person OM or IS, but keep ross_area Technology and Operations.\n"
                 if to_kind == "OM+IS" else "")
        user2 = (
            f"School: {s['university']} — {s['school']}\n"
            f"Unit: {d['name']}\n"
            f"ALLOWED_ROSS_AREAS: {allowed_text}\n"
            f"T&O department-level tag: {to_kind}\n"
            f"Binding rule for this unit: {task_rule}\n"
            f"{split}"
            f"CONFIRMED CURRENT roster to classify, return exactly these people unchanged:\n{names_block}\n"
            "Return ONLY the JSON."
        )
        cls: Dict[str, Tuple[str, str, str]] = {}
        try:
            p2 = call_gpt_json(client, meter, system=FACULTY_CLASSIFY_LIST, user=user2, model=model,
                               use_web_fetch=use_fetch, max_tokens=max_tokens_classify)
            for f in p2.get("faculty", []):
                area = normalize_area(f.get("ross_area"))
                sub = normalize_subfield(f.get("subfield")) if area == "Technology and Operations" else ""
                ev = f.get("field_evidence", "")

                # Python guardrail 1: unique non-T&O department always inherits its mapped area.
                if len(allowed) == 1 and allowed[0] != "Technology and Operations":
                    area = allowed[0]
                    sub = ""
                    ev = ev or f"Official {allowed[0]} unit; inherited area."

                # Python guardrail 2: any T&O department remains T&O; OM/IS is subfield only.
                elif len(allowed) == 1 and allowed[0] == "Technology and Operations":
                    area = "Technology and Operations"
                    if to_kind in ("OM", "IS"):
                        sub = to_kind
                    elif to_kind == "OM+IS":
                        sub = sub if sub in ("OM", "IS") else ""
                    ev = ev or "Official T&O unit; area inherited."

                # Python guardrail 3: multi-area departments can assign only within mapped allowed areas.
                elif len(allowed) >= 2:
                    if area not in allowed:
                        area = allowed[0]
                        sub = ""
                        ev = f"Model chose outside allowed set; defaulted to {allowed[0]}."
                    if area != "Technology and Operations":
                        sub = ""

                # Unconstrained/unmapped: accept model if valid; otherwise leave unclassified.
                else:
                    if area not in ROSS_AREAS:
                        area = "Unclassified"
                        sub = ""

                cls[_name_key(f.get("name"))] = (area, sub, ev)
        except Exception:
            cls = {}

        faculty = []
        for p in roster:
            area, sub, ev = cls.get(_name_key(p["name"]), (None, "", ""))
            if not area:
                area = stamp or "Unclassified"
                sub = stamp_sub if area == "Technology and Operations" else ""
                ev = ev or (f"Official {area} unit; inherited area." if stamp else "Unclassified after failed person classification.")
            faculty.append({**p, "ross_area": area, "subfield": sub, "field_evidence": ev})

    return {"complete": bool(parsed.get("complete")), "sources": parsed.get("sources", []),
            "notes": parsed.get("notes", ""), "faculty": faculty, "classified": classify}

# ----------------------------------------------------------------------------- #
# Resumable cache                                                               #
# ----------------------------------------------------------------------------- #
class Cache:
    def __init__(self, enabled, root=".ross_cache"):
        self.enabled = enabled
        self.root = Path(root)
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        return self.root / (hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".json")

    def get(self, key):
        if not self.enabled:
            return None
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def put(self, key, value):
        if self.enabled:
            try:
                self._path(key).write_text(json.dumps(value))
            except Exception:
                pass

# ----------------------------------------------------------------------------- #
# CSV / CLI compatibility helpers                                               #
# ----------------------------------------------------------------------------- #
def csv_cell(v):
    return "" if v is None else v


def _read_table(path):
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ImportError:
            sys.exit("Reading .xlsx needs pandas/openpyxl: pip install pandas openpyxl")
        return pd.read_excel(p).fillna("").to_dict("records")
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _pick(cols, *pats):
    for p in pats:
        for c in cols:
            if re.search(p, str(c), re.I):
                return c
    return None


def read_school_list(path=None):
    if not path:
        return [{"ranking": r, "university": u, "school": s} for r, u, s in BUILTIN]
    rows = _read_table(path)
    if not rows:
        return []
    cols = list(rows[0].keys())
    rk = _pick(cols, r"rank")
    uk = _pick(cols, r"univ")
    sk = _pick(cols, r"business\s*school", r"^school$", r"b-?school", r"school")
    out = []
    for r in rows:
        item = {"ranking": str(r.get(rk, "")).strip() if rk else "",
                "university": str(r.get(uk, "")).strip() if uk else "",
                "school": str(r.get(sk, "")).strip() if sk else ""}
        if item["university"] or item["school"]:
            out.append(item)
    return out


def read_carry(path):
    rows = _read_table(path)
    if not rows:
        return []
    cols = list(rows[0].keys())
    dept_c = _pick(cols, r"school\s*department", r"department.*own", r"^department$", r"unit")
    rk = _pick(cols, r"rank")
    uk = _pick(cols, r"univ")
    sk = _pick(cols, r"business\s*school", r"^school$", r"b-?school", r"school$")
    rossk = _pick(cols, r"ross\s*area")
    ck = _pick(cols, r"check\s*fac", r"individual")
    urlk = _pick(cols, r"url|source")
    confk = _pick(cols, r"conf")
    tkk = _pick(cols, r"t&o\s*subfield", r"t.o\s*subfield", r"subfield")
    if not dept_c:
        raise ValueError("No department/unit column found.")
    by = {}
    for r in rows:
        uni = str(r.get(uk, "")).strip() if uk else ""
        sch = str(r.get(sk, "")).strip() if sk else ""
        dept = str(r.get(dept_c, "")).strip()
        if not dept or not (uni or sch):
            continue
        e = by.setdefault((uni, sch), {"ranking": str(r.get(rk, "")).strip() if rk else "",
                                      "university": uni, "school": sch, "departments": []})
        ross = valid_ross(str(r.get(rossk, "")) if rossk else "")
        to_kind = normalize_to_kind(str(r.get(tkk, ""))) if (tkk and "Technology and Operations" in ross) else ""
        force = None
        if ck:
            force = str(r.get(ck, "")).strip().lower() in ("yes", "true", "1", "on")
        e["departments"].append({"name": dept, "ross_areas": ross, "to_kind": to_kind,
                                  "url": str(r.get(urlk, "")).strip() if urlk else "",
                                  "confidence": str(r.get(confk, "")).strip().lower() if confk else "",
                                  "notes": "", "force_check": force if force is not None else default_force_check(ross, to_kind)})
    return list(by.values())

# Minimal CLI compatibility, mainly for local runs. Streamlit app uses functions above.
def stage1(args):
    client = OpenAI() if OpenAI else None
    meter = CostMeter(args.model)
    cache = Cache(not args.no_cache, root=".ross_cache/match")
    schools = read_school_list(args.input)
    if args.limit:
        schools = schools[: args.limit]
    results = {}

    def work(s):
        key = f"matchR2::{args.model}::{s['university']}||{s['school']}"
        cached = cache.get(key)
        if cached is not None:
            return s, cached["deps"], cached.get("absent", []), True
        user = (f"Target school: {s['university']} — {s['school']}.\n"
                "Enumerate this school's OWN academic units and map EACH unit to Ross area(s). Return ONLY JSON.")
        parsed = call_gpt_json(client, meter, system=MATCH_SYSTEM, user=user, model=args.model,
                               use_web_fetch=not args.no_web_search, max_tokens=3000)
        deps = parse_departments(parsed)
        absent = []
        gaps = [a for a, ok in ross_coverage(deps).items() if not ok]
        if gaps:
            cu = (f"School: {s['university']} — {s['school']}\n"
                  f"Departments already found: {'; '.join(d['name'] for d in deps) or '(none found)'}\n"
                  f"Ross areas still UNMAPPED: {', '.join(gaps)}\nReturn ONLY JSON.")
            try:
                cov = call_gpt_json(client, meter, system=COVERAGE_SYSTEM, user=cu, model=args.model,
                                    use_web_fetch=not args.no_web_search, max_tokens=1500)
                deps, absent = merge_coverage(deps, cov, gaps)
            except Exception:
                pass
        cache.put(key, {"deps": deps, "absent": absent})
        return s, deps, absent, False

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, s) for s in schools]
        for fut in cf.as_completed(futs):
            s, deps, absent, cached = fut.result()
            results[(s["university"], s["school"])] = deps
            print(f"{s['university']} — {s['school']}: {len(deps)} departments")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Ranking", "University", "Business School", "School Department", "Ross Area", "T&O Subfield", "Check Faculty Individually", "URL", "Confidence", "Notes"])
        for s in schools:
            for d in results.get((s["university"], s["school"]), []):
                w.writerow([s.get("ranking", ""), s["university"], s["school"], d["name"], "; ".join(d["ross_areas"]), d.get("to_kind", ""), "Yes" if d.get("force_check") else "No", d.get("url", ""), d.get("confidence", ""), d.get("notes", "")])
    print(meter.summary())


def stage2(args):
    client = OpenAI() if OpenAI else None
    meter = CostMeter(args.model)
    schools = read_carry(args.input)
    rows = []
    for s in schools[: args.limit if args.limit else None]:
        for d in s.get("departments", []):
            if not args.include_unmapped and not d.get("ross_areas"):
                continue
            try:
                out = fetch_department_faculty(client, meter, args.model, not args.no_web_search, s, d)
                for f in out.get("faculty", []):
                    rows.append([s.get("ranking", ""), s.get("university", ""), s.get("school", ""), d.get("name", ""), f.get("name", ""), f.get("rank", ""), f.get("title", ""), f.get("ross_area", ""), f.get("subfield", ""), f.get("field_evidence", ""), "; ".join(out.get("sources", [])), out.get("notes", "")])
            except Exception as e:
                print(f"ERROR {s.get('school')} / {d.get('name')}: {e}", file=sys.stderr)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Ranking", "University", "Business School", "School Department", "Faculty", "Rank", "Title", "Ross Area", "Subfield", "Field Evidence", "Sources", "Notes"])
        w.writerows(rows)
    print(meter.summary())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("match")
    p1.add_argument("--input")
    p1.add_argument("--out", default="ross_area_matches.csv")
    p1.add_argument("--model", default=DEFAULT_MODEL)
    p1.add_argument("--concurrency", type=int, default=3)
    p1.add_argument("--no-web-search", action="store_true")
    p1.add_argument("--no-cache", action="store_true")
    p1.add_argument("--limit", type=int)
    p2 = sub.add_parser("faculty")
    p2.add_argument("--input", "--in", dest="input", required=True)
    p2.add_argument("--out", default="ross_faculty_by_area.csv")
    p2.add_argument("--model", default=DEFAULT_MODEL)
    p2.add_argument("--no-web-search", action="store_true")
    p2.add_argument("--include-unmapped", action="store_true")
    p2.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    if args.cmd == "match":
        stage1(args)
    elif args.cmd == "faculty":
        stage2(args)


if __name__ == "__main__":
    main()
