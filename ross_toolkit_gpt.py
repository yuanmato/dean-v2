
"""
ross_toolkit_gpt.py

Evidence-first GPT/OpenAI engine for mapping business-school faculty to Ross areas.

Core design:
1. Map each school's local academic units/departments to Ross areas.
2. Collect current active ladder/tenure-track faculty from each official unit roster.
3. Unique-match departments inherit the mapped Ross area.
4. Person-level classification is used ONLY for genuinely mixed/combined units.
5. Marketing stays Marketing even if a person has a few OM/interface papers.
6. T&O units stay Technology & Operations; OM vs IS is only a subfield split.

Requires:
    openai>=1.88.0
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


ROSS_AREAS = [
    "Accounting",
    "Business & Economics",
    "Finance",
    "Management & Organizations",
    "Marketing",
    "Strategy",
    "Technology & Operations",
]

ROSS_AREAS_PLUS_UNCLASSIFIED = ROSS_AREAS + ["Unclassified"]

TO_KINDS = ["", "OM", "IS", "OM+IS"]

# Words that normally indicate not ladder/tenure-track research faculty.
EXCLUDE_TITLE_KEYWORDS = [
    "lecturer",
    "senior lecturer",
    "adjunct",
    "visiting",
    "clinical",
    "professor of practice",
    "practice professor",
    "teaching professor",
    "teaching faculty",
    "instructor",
    "emeritus",
    "emerita",
    "postdoctoral",
    "postdoc",
    "research fellow",
    "fellow",
    "staff",
    "phd student",
    "doctoral student",
    "student",
    "affiliate",
    "affiliated",
    "courtesy",
]

# Phrases that normally indicate active ladder/tenure-track or tenured research faculty.
INCLUDE_TITLE_PATTERNS = [
    "assistant professor",
    "associate professor",
    "professor",
    "chaired professor",
    "endowed professor",
    "distinguished professor",
    "named professor",
]

# Local administrative unit aliases. This is deliberately department-first.
# Journals should not override these unique official homes.
LOCAL_AREA_ALIASES: Dict[str, Tuple[List[str], str]] = {
    # Accounting
    "accounting": (["Accounting"], ""),
    "accounting and information management": (["Accounting"], ""),
    "accountancy": (["Accounting"], ""),

    # Business & Economics
    "business economics": (["Business & Economics"], ""),
    "business economics and public policy": (["Business & Economics"], ""),
    "business economics & public policy": (["Business & Economics"], ""),
    "economics": (["Business & Economics"], ""),
    "managerial economics": (["Business & Economics"], ""),
    "applied economics": (["Business & Economics"], ""),
    "economics and public policy": (["Business & Economics"], ""),
    "public policy": (["Business & Economics"], ""),

    # Finance
    "finance": (["Finance"], ""),
    "banking and finance": (["Finance"], ""),

    # Marketing
    "marketing": (["Marketing"], ""),
    "marketing and sales": (["Marketing"], ""),

    # Strategy
    "strategy": (["Strategy"], ""),
    "strategic management": (["Strategy"], ""),
    "strategy and entrepreneurship": (["Strategy"], ""),
    "entrepreneurship and strategy": (["Strategy"], ""),
    "entrepreneurship": (["Strategy"], ""),

    # Management & Organizations
    "organizational behavior": (["Management & Organizations"], ""),
    "organisational behaviour": (["Management & Organizations"], ""),
    "organizations": (["Management & Organizations"], ""),
    "organisations": (["Management & Organizations"], ""),
    "organization and management": (["Management & Organizations"], ""),
    "organisation and management": (["Management & Organizations"], ""),
    "management and organizations": (["Management & Organizations"], ""),
    "management & organizations": (["Management & Organizations"], ""),
    "management and organisations": (["Management & Organizations"], ""),
    "human resources": (["Management & Organizations"], ""),
    "human resource management": (["Management & Organizations"], ""),
    "leadership": (["Management & Organizations"], ""),

    # Mixed Management groups: use individual split only within allowed set.
    "management": (["Management & Organizations", "Strategy"], ""),
    "management and entrepreneurship": (["Management & Organizations", "Strategy"], ""),
    "organizations and strategy": (["Management & Organizations", "Strategy"], ""),
    "organisation and strategy": (["Management & Organizations", "Strategy"], ""),
    "organization and strategy": (["Management & Organizations", "Strategy"], ""),

    # Technology & Operations
    "operations": (["Technology & Operations"], "OM"),
    "operations management": (["Technology & Operations"], "OM"),
    "operations and supply chain": (["Technology & Operations"], "OM"),
    "supply chain": (["Technology & Operations"], "OM"),
    "supply chain management": (["Technology & Operations"], "OM"),
    "production and operations management": (["Technology & Operations"], "OM"),
    "decision sciences": (["Technology & Operations"], "OM+IS"),
    "operations research": (["Technology & Operations"], "OM"),
    "management science": (["Technology & Operations"], "OM"),
    "technology and operations": (["Technology & Operations"], "OM+IS"),
    "technology & operations": (["Technology & Operations"], "OM+IS"),
    "technology and operations management": (["Technology & Operations"], "OM+IS"),
    "operations, information and technology": (["Technology & Operations"], "OM+IS"),
    "operations, information & technology": (["Technology & Operations"], "OM+IS"),
    "operations information technology": (["Technology & Operations"], "OM+IS"),
    "operations information and decisions": (["Technology & Operations"], "OM+IS"),
    "operations, information and decisions": (["Technology & Operations"], "OM+IS"),
    "operations, information & decisions": (["Technology & Operations"], "OM+IS"),
    "information systems": (["Technology & Operations"], "IS"),
    "information technology": (["Technology & Operations"], "IS"),
    "information management": (["Technology & Operations"], "IS"),
    "analytics": (["Technology & Operations"], "OM+IS"),
    "business analytics": (["Technology & Operations"], "OM+IS"),
}


FIELD_GUIDE = """
Ross canonical areas:
- Accounting: accounting, auditing, tax, financial reporting, managerial accounting.
- Business & Economics: business economics, applied economics, public policy, industrial organization,
  health economics, labor, international economics, political economy, econometrics in economics units.
- Finance: asset pricing, corporate finance, banking, household finance, market microstructure.
- Management & Organizations: organizational behavior, organization theory, HR, leadership, teams,
  negotiations, social psychology in organizations.
- Marketing: quantitative marketing, consumer behavior, marketing strategy, advertising, pricing when
  the official home is Marketing.
- Strategy: strategic management, entrepreneurship, innovation strategy, competitive strategy,
  nonmarket strategy.
- Technology & Operations: operations management, operations research, supply chain, service ops,
  healthcare ops, information systems, information technology, analytics, decision sciences.

Important mapping principle:
- The official local academic unit is the primary area signal.
- Publications are used to split genuinely mixed/combined units, not to override a clear department.
- Example: a person in Marketing with some OM/interface papers remains Marketing.
- Example: a broad Management group may need person-level split into Management & Organizations vs Strategy.
- Example: an Operations/Information/Decision Sciences group maps to Ross Technology & Operations; OM vs IS is
  a subfield, not a different Ross area.
"""


SYSTEM_INSTRUCTIONS = f"""
You are a careful academic faculty-roster and area-mapping assistant.

You must be conservative and evidence-based.

Use official university/school pages whenever possible. Prefer current official faculty directories,
department/area rosters, and official profile pages. Personal websites, Google Scholar, SSRN, OpenAlex,
and journal pages are secondary evidence only.

Never include non-ladder/non-tenure-track people as current ladder faculty unless the official title
clearly supports inclusion.

{FIELD_GUIDE}

Return only valid JSON matching the supplied schema.
"""


@dataclass
class SchoolConfig:
    university: str
    school: str
    faculty_directory_url: str = ""
    notes: str = ""


@dataclass
class UnitMapping:
    unit_name: str
    unit_url: str
    local_group_label: str
    ross_areas: List[str]
    to_kind: str
    is_mixed: bool
    mixed_reason: str
    source_evidence: str
    confidence: float
    needs_review: bool


@dataclass
class FacultyRecord:
    school: str
    university: str
    local_unit: str
    unit_url: str
    name: str
    rank: str
    title: str
    profile_url: str
    official_area: str
    is_ladder: bool
    status_evidence: str
    source_url: str
    ross_area: str
    subfield: str
    field_evidence: str
    area_assignment_method: str
    confidence: float
    needs_review: bool
    review_note: str


def _json_schema_format(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": True,
        }
    }


def response_output_text(response: Any) -> str:
    """Robust extraction of text from Responses API output."""
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # Fallback for older or altered SDK representations.
    try:
        parts = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", "")
                    if text:
                        parts.append(text)
        return "\n".join(parts)
    except Exception:
        return str(response)


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a top-level JSON object, with defensive cleanup for non-strict fallback cases."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("Model response JSON was not a top-level object.")
        return obj
    except json.JSONDecodeError:
        pass

    # Defensive fallbacks; structured outputs should make these rare.
    if text.startswith("```"):
        cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse a valid top-level JSON object.")


def call_openai_json(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    schema_name: str,
    schema: Dict[str, Any],
    use_web_search: bool = True,
    require_search: bool = True,
    search_context_size: str = "medium",
    retries: int = 2,
) -> Dict[str, Any]:
    """Call Responses API with strict JSON schema and optional hosted web search."""
    tools: List[Dict[str, Any]] = []
    tool_choice: Any = "auto"

    if use_web_search:
        tools = [
            {
                "type": "web_search",
                "search_context_size": search_context_size,
            }
        ]
        if require_search:
            tool_choice = "required"

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "instructions": SYSTEM_INSTRUCTIONS,
                "input": prompt,
                "text": _json_schema_format(schema_name, schema),
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            response = client.responses.create(**kwargs)
            return parse_json_object(response_output_text(response))
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"OpenAI JSON call failed: {last_err}") from exc

    raise RuntimeError(f"OpenAI JSON call failed: {last_err}")


def clean(s: Any) -> str:
    return "" if s is None else str(s).strip()


def normalize_key(s: str) -> str:
    s = clean(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_ross_area(area: Any) -> str:
    raw = clean(area)
    key = normalize_key(raw)
    alias = {
        "business economics": "Business & Economics",
        "business and economics": "Business & Economics",
        "business economics public policy": "Business & Economics",
        "management organizations": "Management & Organizations",
        "management and organizations": "Management & Organizations",
        "management organisation": "Management & Organizations",
        "management organisations": "Management & Organizations",
        "technology operations": "Technology & Operations",
        "technology and operations": "Technology & Operations",
        "technology operations management": "Technology & Operations",
        "to": "Technology & Operations",
        "t o": "Technology & Operations",
        "m o": "Management & Organizations",
        "mo": "Management & Organizations",
        "bepp": "Business & Economics",
    }
    if raw in ROSS_AREAS:
        return raw
    if key in alias:
        return alias[key]
    for a in ROSS_AREAS:
        if key == normalize_key(a):
            return a
    return "Unclassified"


def normalize_to_kind(x: Any) -> str:
    x = clean(x).upper().replace(" ", "")
    if x in {"OM", "IS", "OM+IS"}:
        return x
    if x in {"OMIS", "OM/IS", "OM&IS"}:
        return "OM+IS"
    return ""


def normalize_rank(title_or_rank: str) -> str:
    t = clean(title_or_rank)
    low = t.lower()
    if "assistant professor" in low:
        return "Assistant Professor"
    if "associate professor" in low:
        return "Associate Professor"
    # Professor but avoid professor of practice/teaching handled elsewhere.
    if "professor" in low:
        return "Professor"
    return t


def title_is_ladder(title: str) -> Tuple[bool, str]:
    """Conservative deterministic ladder faculty title screen."""
    t = normalize_key(title)

    if not t:
        return False, "Missing title."

    for bad in EXCLUDE_TITLE_KEYWORDS:
        if normalize_key(bad) in t:
            return False, f"Excluded by title keyword: {bad}."

    if "assistant professor" in t:
        return True, "Assistant Professor title."
    if "associate professor" in t:
        return True, "Associate Professor title."

    # "Professor" includes named chairs, full professor, dean and professor.
    # Already excluded clinical/visiting/teaching/practice/etc above.
    if re.search(r"\bprofessor\b", t):
        return True, "Professor title with no excluded modifier."

    return False, "No clear ladder-rank professor title."


def alias_unit_mapping(unit_name: str) -> Tuple[List[str], str]:
    key = normalize_key(unit_name)
    if key in LOCAL_AREA_ALIASES:
        return LOCAL_AREA_ALIASES[key]

    # Partial robust matching, longest keys first.
    for alias in sorted(LOCAL_AREA_ALIASES, key=len, reverse=True):
        ak = normalize_key(alias)
        if ak and ak in key:
            return LOCAL_AREA_ALIASES[alias]

    return [], ""


def dedupe_units(units: List[UnitMapping]) -> List[UnitMapping]:
    seen = set()
    out: List[UnitMapping] = []
    for u in units:
        key = normalize_key(u.unit_name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def dedupe_faculty(records: List[FacultyRecord]) -> List[FacultyRecord]:
    seen = set()
    out: List[FacultyRecord] = []
    for r in records:
        key = (normalize_key(r.name), normalize_key(r.school), normalize_key(r.local_unit))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


UNIT_MAPPING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "university": {"type": "string"},
        "school": {"type": "string"},
        "units": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "unit_name": {"type": "string"},
                    "unit_url": {"type": "string"},
                    "local_group_label": {"type": "string"},
                    "ross_areas": {
                        "type": "array",
                        "items": {"type": "string", "enum": ROSS_AREAS},
                    },
                    "to_kind": {"type": "string", "enum": TO_KINDS},
                    "is_mixed": {"type": "boolean"},
                    "mixed_reason": {"type": "string"},
                    "source_evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                    "needs_review": {"type": "boolean"},
                },
                "required": [
                    "unit_name",
                    "unit_url",
                    "local_group_label",
                    "ross_areas",
                    "to_kind",
                    "is_mixed",
                    "mixed_reason",
                    "source_evidence",
                    "confidence",
                    "needs_review",
                ],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["university", "school", "units", "notes"],
}


ROSTER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "university": {"type": "string"},
        "school": {"type": "string"},
        "unit_name": {"type": "string"},
        "unit_url": {"type": "string"},
        "faculty": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "rank": {"type": "string"},
                    "title": {"type": "string"},
                    "profile_url": {"type": "string"},
                    "official_area": {"type": "string"},
                    "is_ladder": {"type": "boolean"},
                    "status_evidence": {"type": "string"},
                    "source_url": {"type": "string"},
                    "confidence": {"type": "number"},
                    "exclusion_note": {"type": "string"},
                },
                "required": [
                    "name",
                    "rank",
                    "title",
                    "profile_url",
                    "official_area",
                    "is_ladder",
                    "status_evidence",
                    "source_url",
                    "confidence",
                    "exclusion_note",
                ],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["university", "school", "unit_name", "unit_url", "faculty", "notes"],
}


MIXED_CLASSIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "university": {"type": "string"},
        "school": {"type": "string"},
        "unit_name": {"type": "string"},
        "faculty": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "rank": {"type": "string"},
                    "title": {"type": "string"},
                    "ross_area": {"type": "string", "enum": ROSS_AREAS_PLUS_UNCLASSIFIED},
                    "subfield": {"type": "string", "enum": TO_KINDS},
                    "field_evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                    "needs_review": {"type": "boolean"},
                    "review_note": {"type": "string"},
                },
                "required": [
                    "name",
                    "rank",
                    "title",
                    "ross_area",
                    "subfield",
                    "field_evidence",
                    "confidence",
                    "needs_review",
                    "review_note",
                ],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["university", "school", "unit_name", "faculty", "notes"],
}


def build_unit_mapping_prompt(
    school: SchoolConfig,
    provided_units_text: str = "",
    max_units: int = 40,
) -> str:
    return f"""
Task: Build the school's local academic-unit to Ross-area mapping.

University: {school.university}
Business school: {school.school}
Official faculty directory URL: {school.faculty_directory_url}
User notes: {school.notes}

User-provided units, if any:
{provided_units_text}

Instructions:
1. Use the official business-school website and current faculty/department/area pages.
2. If the user supplied units, map those units; do not invent extra units unless the provided list is empty.
3. If the user did not supply units, discover the school's current research faculty areas/departments/groups
   from the official directory or official faculty-research pages.
4. Map each LOCAL unit to Ross's canonical area(s).
5. Department-first rule:
   - A clear Marketing department maps only to Marketing.
   - A clear Finance department maps only to Finance.
   - A clear Accounting department maps only to Accounting.
   - A clear T&O/OID/Decision Sciences/IS/Operations unit maps to Technology & Operations.
   - Publications by individual faculty do not make a clear unit mixed.
6. Mark is_mixed = true ONLY if the official local unit itself combines multiple Ross categories,
   such as a broad Management group containing both OB/M&O and Strategy, Economics & Strategy,
   or Accounting & Finance.
7. For Technology & Operations:
   - to_kind = "OM" for OM/supply chain/operations-only units.
   - to_kind = "IS" for information systems / information management-only units.
   - to_kind = "OM+IS" for combined operations + information + analytics / decision-science units.
   - ross_areas should still be ["Technology & Operations"].
8. Do not create separate Ross areas for OM and IS; both are Ross Technology & Operations.
9. Return no more than {max_units} units.
10. Evidence should be short and cite the official unit name/page text.
"""


def map_units(
    client: OpenAI,
    school: SchoolConfig,
    *,
    model: str,
    provided_units_text: str = "",
    max_units: int = 40,
    use_web_search: bool = True,
    require_search: bool = True,
    search_context_size: str = "medium",
) -> List[UnitMapping]:
    prompt = build_unit_mapping_prompt(school, provided_units_text, max_units)
    obj = call_openai_json(
        client,
        model=model,
        prompt=prompt,
        schema_name="unit_mapping",
        schema=UNIT_MAPPING_SCHEMA,
        use_web_search=use_web_search,
        require_search=require_search,
        search_context_size=search_context_size,
    )

    units: List[UnitMapping] = []
    for item in obj.get("units", [])[:max_units]:
        name = clean(item.get("unit_name"))
        model_areas = [normalize_ross_area(x) for x in item.get("ross_areas", [])]
        model_areas = [x for x in model_areas if x in ROSS_AREAS]
        model_to = normalize_to_kind(item.get("to_kind"))

        alias_areas, alias_to = alias_unit_mapping(name)

        # Alias is used as a guardrail only when it is a clear unique or known mixed label.
        # It prevents the model from marking Marketing/Finance/etc. as mixed because of interdisciplinary papers.
        if alias_areas:
            areas = alias_areas
            to_kind = alias_to or model_to
            guardrail_note = ""
            if set(model_areas) != set(alias_areas):
                guardrail_note = f" Python alias guardrail applied: model areas={model_areas}, alias areas={alias_areas}."
        else:
            areas = model_areas
            to_kind = model_to
            guardrail_note = ""

        areas = [a for a in areas if a in ROSS_AREAS]
        is_mixed = len(areas) >= 2
        if areas == ["Technology & Operations"]:
            is_mixed = False  # OM+IS is not mixed across Ross areas.
        if not areas:
            areas = []
            is_mixed = False

        confidence = float(item.get("confidence") or 0.0)
        needs_review = bool(item.get("needs_review")) or confidence < 0.75 or not areas

        units.append(
            UnitMapping(
                unit_name=name,
                unit_url=clean(item.get("unit_url")),
                local_group_label=clean(item.get("local_group_label")),
                ross_areas=areas,
                to_kind=to_kind,
                is_mixed=is_mixed,
                mixed_reason=(clean(item.get("mixed_reason")) + guardrail_note).strip(),
                source_evidence=clean(item.get("source_evidence")),
                confidence=confidence,
                needs_review=needs_review,
            )
        )

    return dedupe_units(units)


def build_roster_prompt(
    school: SchoolConfig,
    unit: UnitMapping,
    max_faculty: int = 120,
) -> str:
    return f"""
Task: Extract current active ladder-rank faculty from ONE official academic unit.

University: {school.university}
Business school: {school.school}
Official school faculty directory URL: {school.faculty_directory_url}
Unit name: {unit.unit_name}
Unit URL: {unit.unit_url}
Unit mapped Ross areas: {", ".join(unit.ross_areas) if unit.ross_areas else "(unmapped)"}
User/school notes: {school.notes}

Instructions:
1. Use the official current unit roster, area roster, and official profile pages.
2. Return all visible people associated with this official unit, marking is_ladder accurately.
3. Ladder/tenure-track/tenured faculty generally include:
   Assistant Professor, Associate Professor, Professor, named/chair/endowed Professor.
4. Exclude or mark is_ladder=false for:
   lecturer, adjunct, visiting, clinical, teaching professor, professor of practice,
   emeritus/emerita, affiliated-only, courtesy-only, research fellow, postdoc, staff, students.
5. Do not include people merely because they publish in this field.
6. Do not add faculty from another unit unless the official roster/profile shows this unit as their primary/core home.
7. If unsure, include the person but set is_ladder=false and explain in exclusion_note.
8. Return no more than {max_faculty} people.
9. Use exact current names and titles from official pages.
"""


def collect_roster(
    client: OpenAI,
    school: SchoolConfig,
    unit: UnitMapping,
    *,
    model: str,
    max_faculty: int = 120,
    use_web_search: bool = True,
    require_search: bool = True,
    search_context_size: str = "medium",
) -> List[Dict[str, Any]]:
    prompt = build_roster_prompt(school, unit, max_faculty)
    obj = call_openai_json(
        client,
        model=model,
        prompt=prompt,
        schema_name="faculty_roster",
        schema=ROSTER_SCHEMA,
        use_web_search=use_web_search,
        require_search=require_search,
        search_context_size=search_context_size,
    )

    rows: List[Dict[str, Any]] = []
    for f in obj.get("faculty", [])[:max_faculty]:
        title = clean(f.get("title"))
        deterministic_ladder, deterministic_reason = title_is_ladder(title)
        model_ladder = bool(f.get("is_ladder"))

        # Conservative: require the model OR deterministic title screen, but exclude if deterministic title explicitly catches bad terms.
        # If model says ladder and deterministic title says no due to missing title only, keep model if confidence high.
        is_ladder = model_ladder and deterministic_ladder
        if model_ladder and not deterministic_ladder and "Missing title" in deterministic_reason:
            is_ladder = float(f.get("confidence") or 0.0) >= 0.85

        exclusion_note = clean(f.get("exclusion_note"))
        if not is_ladder:
            exclusion_note = exclusion_note or deterministic_reason

        rows.append(
            {
                "name": clean(f.get("name")),
                "rank": clean(f.get("rank")) or normalize_rank(title),
                "title": title,
                "profile_url": clean(f.get("profile_url")),
                "official_area": clean(f.get("official_area")) or unit.unit_name,
                "is_ladder": bool(is_ladder),
                "status_evidence": clean(f.get("status_evidence")) or deterministic_reason,
                "source_url": clean(f.get("source_url")) or unit.unit_url,
                "confidence": float(f.get("confidence") or 0.0),
                "exclusion_note": exclusion_note,
            }
        )

    # Deduplicate by name inside unit.
    seen = set()
    deduped = []
    for r in rows:
        k = normalize_key(r["name"])
        if k and k not in seen:
            seen.add(k)
            deduped.append(r)
    return deduped


def build_mixed_classify_prompt(
    school: SchoolConfig,
    unit: UnitMapping,
    faculty_rows: List[Dict[str, Any]],
) -> str:
    allowed = unit.ross_areas[:]
    roster_json = json.dumps(faculty_rows, ensure_ascii=False, indent=2)

    if len(allowed) == 1 and allowed[0] == "Technology & Operations":
        if unit.to_kind == "OM+IS":
            binding_rule = (
                "This is a Technology & Operations unit containing OM and IS. "
                "Set ross_area='Technology & Operations' for EVERYONE. "
                "Use publications/profile only to classify subfield as OM or IS."
            )
        elif unit.to_kind in {"OM", "IS"}:
            binding_rule = (
                f"This is a unique Technology & Operations unit. "
                f"Set ross_area='Technology & Operations' and subfield='{unit.to_kind}' for EVERYONE."
            )
        else:
            binding_rule = (
                "This is a Technology & Operations unit. "
                "Set ross_area='Technology & Operations' for EVERYONE. Infer subfield only if clear."
            )
    elif len(allowed) == 1:
        binding_rule = (
            f"This is a unique-match unit. Set ross_area='{allowed[0]}' for EVERYONE. "
            "Do not override the official department because of cross-field or interface publications."
        )
    elif len(allowed) >= 2:
        binding_rule = (
            "This is a genuinely mixed/combined unit. Classify each person into exactly ONE of "
            f"these allowed Ross areas only: {allowed}. Do not assign outside this allowed set."
        )
    else:
        binding_rule = (
            "This unit is unmapped/unclear. Use official profile and publication evidence to choose the closest Ross area."
        )

    return f"""
Task: Assign Ross area for a confirmed roster from one official local unit.

University: {school.university}
Business school: {school.school}
Unit: {unit.unit_name}
Allowed Ross areas for this unit: {allowed if allowed else "(unconstrained)"}
T&O kind: {unit.to_kind}
Binding rule: {binding_rule}

CRITICAL:
- Return the same people you are given, using exact names.
- Do not add/remove/rename people.
- A clear official department controls the Ross area.
- Publications are used only for mixed/combined units, or OM/IS subfield inside T&O.
- Marketing faculty remain Marketing even if they publish a few OM/interface papers.
- Finance faculty remain Finance even if they publish econ/strategy-adjacent papers.
- T&O faculty remain Technology & Operations; OM vs IS is a subfield only.

Confirmed current ladder roster:
{roster_json}

For evidence:
- Unique inherited cases: "Official [unit] unit; inherited area."
- Mixed management split: cite profile/research/journal signals briefly, e.g. "SMJ/strategy research -> Strategy"
  or "OB/AMJ/ASQ research -> M&O".
- T&O subfield: cite OM/IS markers, e.g. "MSOM/POM -> OM" or "ISR/MISQ -> IS".
"""


def assign_areas_for_unit(
    client: OpenAI,
    school: SchoolConfig,
    unit: UnitMapping,
    roster_rows: List[Dict[str, Any]],
    *,
    model: str,
    use_web_search: bool = True,
    require_search: bool = True,
    search_context_size: str = "medium",
) -> List[FacultyRecord]:
    """Assign Ross areas using department-first constrained logic."""
    ladder_rows = [r for r in roster_rows if r.get("is_ladder")]
    records: List[FacultyRecord] = []

    allowed = unit.ross_areas[:]
    method = ""

    # Unique non-T&O department: inherit. No model classification needed.
    if len(allowed) == 1 and allowed[0] != "Technology & Operations":
        method = "unique_unit_inherited"
        for r in ladder_rows:
            records.append(
                FacultyRecord(
                    school=school.school,
                    university=school.university,
                    local_unit=unit.unit_name,
                    unit_url=unit.unit_url,
                    name=r["name"],
                    rank=r["rank"],
                    title=r["title"],
                    profile_url=r["profile_url"],
                    official_area=r["official_area"],
                    is_ladder=True,
                    status_evidence=r["status_evidence"],
                    source_url=r["source_url"],
                    ross_area=allowed[0],
                    subfield="",
                    field_evidence=f"Official {unit.unit_name} unit; inherited {allowed[0]}.",
                    area_assignment_method=method,
                    confidence=min(0.98, max(float(r.get("confidence") or 0.0), unit.confidence)),
                    needs_review=unit.needs_review,
                    review_note="Review unit mapping only." if unit.needs_review else "",
                )
            )
        return records

    # Unique T&O department. T&O is final Ross area; optionally classify OM/IS subfield by model if OM+IS.
    # If the subfield is known at department level, no model needed.
    if len(allowed) == 1 and allowed[0] == "Technology & Operations" and unit.to_kind in {"OM", "IS"}:
        method = "unique_to_unit_inherited_subfield"
        for r in ladder_rows:
            records.append(
                FacultyRecord(
                    school=school.school,
                    university=school.university,
                    local_unit=unit.unit_name,
                    unit_url=unit.unit_url,
                    name=r["name"],
                    rank=r["rank"],
                    title=r["title"],
                    profile_url=r["profile_url"],
                    official_area=r["official_area"],
                    is_ladder=True,
                    status_evidence=r["status_evidence"],
                    source_url=r["source_url"],
                    ross_area="Technology & Operations",
                    subfield=unit.to_kind,
                    field_evidence=f"Official {unit.unit_name} unit; inherited T&O-{unit.to_kind}.",
                    area_assignment_method=method,
                    confidence=min(0.98, max(float(r.get("confidence") or 0.0), unit.confidence)),
                    needs_review=unit.needs_review,
                    review_note="Review unit mapping only." if unit.needs_review else "",
                )
            )
        return records

    # Mixed department, T&O OM+IS subfield split, or unmapped unit: use constrained model.
    prompt = build_mixed_classify_prompt(school, unit, ladder_rows)
    obj = call_openai_json(
        client,
        model=model,
        prompt=prompt,
        schema_name="mixed_faculty_classification",
        schema=MIXED_CLASSIFY_SCHEMA,
        use_web_search=use_web_search,
        require_search=require_search,
        search_context_size=search_context_size,
    )

    classified_by_name: Dict[str, Dict[str, Any]] = {}
    for f in obj.get("faculty", []):
        classified_by_name[normalize_key(f.get("name", ""))] = f

    for r in ladder_rows:
        f = classified_by_name.get(normalize_key(r["name"]), {})
        area = normalize_ross_area(f.get("ross_area"))
        sub = normalize_to_kind(f.get("subfield"))
        ev = clean(f.get("field_evidence"))
        conf = float(f.get("confidence") or 0.0)
        needs_review = bool(f.get("needs_review"))
        review = clean(f.get("review_note"))

        # Python guardrail 1: unique non-T&O department always inherits its area.
        if len(allowed) == 1 and allowed[0] != "Technology & Operations":
            area = allowed[0]
            sub = ""
            ev = ev or f"Official {unit.unit_name} unit; inherited {allowed[0]}."
            method = "python_guardrail_unique_unit_inherited"

        # Python guardrail 2: any T&O unit remains Ross T&O.
        elif len(allowed) == 1 and allowed[0] == "Technology & Operations":
            area = "Technology & Operations"
            if unit.to_kind in {"OM", "IS"}:
                sub = unit.to_kind
            elif unit.to_kind == "OM+IS":
                sub = sub if sub in {"OM", "IS"} else ""
                if not sub:
                    needs_review = True
                    review = review or "T&O unit; OM/IS subfield unclear."
            else:
                sub = sub if sub in {"OM", "IS"} else ""
            ev = ev or "Official T&O unit; area inherited."
            method = "to_unit_area_inherited_subfield_classified"

        # Python guardrail 3: mixed units can only assign within allowed mapped Ross areas.
        elif len(allowed) >= 2:
            if area not in allowed:
                needs_review = True
                review = review or f"Model chose {area}, outside allowed set {allowed}; defaulted to {allowed[0]}."
                area = allowed[0]
            if area != "Technology & Operations":
                sub = ""
            method = "mixed_unit_person_level_with_allowed_set"

        # Unmapped unit.
        else:
            if area not in ROSS_AREAS:
                area = "Unclassified"
                needs_review = True
                review = review or "Unmapped unit and person-level classification unclear."
            if area != "Technology & Operations":
                sub = ""
            method = "unmapped_unit_person_level"

        if conf <= 0:
            conf = min(0.75, max(float(r.get("confidence") or 0.0), unit.confidence))

        records.append(
            FacultyRecord(
                school=school.school,
                university=school.university,
                local_unit=unit.unit_name,
                unit_url=unit.unit_url,
                name=r["name"],
                rank=r["rank"],
                title=r["title"],
                profile_url=r["profile_url"],
                official_area=r["official_area"],
                is_ladder=True,
                status_evidence=r["status_evidence"],
                source_url=r["source_url"],
                ross_area=area,
                subfield=sub,
                field_evidence=ev,
                area_assignment_method=method,
                confidence=conf,
                needs_review=needs_review or unit.needs_review or conf < 0.75,
                review_note=review,
            )
        )

    return records


def run_school_pipeline(
    client: OpenAI,
    school: SchoolConfig,
    *,
    model: str,
    provided_units_text: str = "",
    max_units: int = 40,
    max_faculty_per_unit: int = 120,
    use_web_search: bool = True,
    require_search: bool = True,
    search_context_size: str = "medium",
    progress_callback=None,
) -> Tuple[List[UnitMapping], List[Dict[str, Any]], List[FacultyRecord], List[Dict[str, str]]]:
    """
    Full school pipeline.

    Returns:
        units, raw_rosters, final_records, errors
    """
    errors: List[Dict[str, str]] = []
    raw_rosters: List[Dict[str, Any]] = []
    final_records: List[FacultyRecord] = []

    if progress_callback:
        progress_callback("Mapping local units to Ross areas...")

    units = map_units(
        client,
        school,
        model=model,
        provided_units_text=provided_units_text,
        max_units=max_units,
        use_web_search=use_web_search,
        require_search=require_search,
        search_context_size=search_context_size,
    )

    for idx, unit in enumerate(units, start=1):
        try:
            if progress_callback:
                progress_callback(f"[{idx}/{len(units)}] Collecting roster: {unit.unit_name}")

            roster = collect_roster(
                client,
                school,
                unit,
                model=model,
                max_faculty=max_faculty_per_unit,
                use_web_search=use_web_search,
                require_search=require_search,
                search_context_size=search_context_size,
            )

            for r in roster:
                rr = dict(r)
                rr.update(
                    {
                        "university": school.university,
                        "school": school.school,
                        "local_unit": unit.unit_name,
                        "unit_url": unit.unit_url,
                        "mapped_ross_areas": "; ".join(unit.ross_areas),
                        "to_kind": unit.to_kind,
                    }
                )
                raw_rosters.append(rr)

            if progress_callback:
                progress_callback(f"[{idx}/{len(units)}] Assigning Ross areas: {unit.unit_name}")

            assigned = assign_areas_for_unit(
                client,
                school,
                unit,
                roster,
                model=model,
                use_web_search=use_web_search,
                require_search=require_search,
                search_context_size=search_context_size,
            )
            final_records.extend(assigned)

        except Exception as exc:
            errors.append(
                {
                    "unit_name": unit.unit_name,
                    "unit_url": unit.unit_url,
                    "error": str(exc),
                }
            )

    return units, raw_rosters, dedupe_faculty(final_records), errors


def units_to_dicts(units: List[UnitMapping]) -> List[Dict[str, Any]]:
    return [asdict(u) for u in units]


def records_to_dicts(records: List[FacultyRecord]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in records]
