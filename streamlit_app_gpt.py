
"""
streamlit_app_gpt.py

Corrected Streamlit app for cross-school faculty mapping to Ross areas.

Run locally:
    streamlit run streamlit_app_gpt.py

Streamlit Cloud:
    - main file: streamlit_app_gpt.py
    - secrets:
        OPENAI_API_KEY = "sk-..."
"""

from __future__ import annotations

from io import BytesIO
from typing import List

import pandas as pd
import streamlit as st
from openai import OpenAI

from ross_toolkit_gpt import (
    ROSS_AREAS,
    SchoolConfig,
    run_school_pipeline,
    units_to_dicts,
    records_to_dicts,
)


st.set_page_config(
    page_title="Ross Area Mapper — GPT",
    page_icon="🎓",
    layout="wide",
)


def get_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Cloud → App → Settings → Secrets.")
        st.code('OPENAI_API_KEY = "sk-your-real-key"', language="toml")
        st.stop()
    return OpenAI(api_key=api_key)


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()


def parse_school_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    return pd.read_csv(uploaded_file).fillna("")


def default_schools_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "university": "University of Michigan",
                "school": "Ross School of Business",
                "faculty_directory_url": "https://michiganross.umich.edu/faculty-research/faculty",
                "notes": "Use official Ross faculty and area pages.",
            }
        ]
    )


def dataframe_editor_for_schools() -> pd.DataFrame:
    st.markdown(
        "Enter one or more schools. The app maps each school’s own units to Ross areas, "
        "then collects current ladder faculty from official pages."
    )

    uploaded = st.file_uploader(
        "Optional: upload schools CSV",
        type=["csv"],
        help="Columns: university, school, faculty_directory_url, notes",
    )

    if uploaded:
        df0 = parse_school_csv(uploaded)
    else:
        df0 = default_schools_df()

    required = ["university", "school", "faculty_directory_url", "notes"]
    for c in required:
        if c not in df0.columns:
            df0[c] = ""

    edited = st.data_editor(
        df0[required],
        num_rows="dynamic",
        use_container_width=True,
        key="schools_editor",
    ).fillna("")

    return edited


st.title("🎓 Ross Area Mapper — GPT/OpenAI corrected version")

st.write(
    "This version preserves the department-first logic: unique local departments inherit their Ross area; "
    "person-level classification is used only for genuinely mixed units such as Management containing both "
    "M&O and Strategy, or for OM/IS subfield inside T&O."
)

with st.expander("Classification rules used by this app", expanded=True):
    st.markdown(
        """
**Ross canonical areas**

`Accounting`, `Business & Economics`, `Finance`, `Management & Organizations`,
`Marketing`, `Strategy`, `Technology & Operations`.

**Core rule**

- A clear **Marketing** department maps to **Marketing**. A Marketing professor with a few OM/interface papers remains Marketing.
- A clear **Finance**, **Accounting**, **Strategy**, **M&O**, or **T&O** department inherits its mapped Ross area.
- A broad **Management** group can be split person-by-person into **Management & Organizations** vs **Strategy**.
- A broad **T&O / OID / Decision Sciences / Operations + Information** group remains **Technology & Operations**; the app may split only the subfield as `OM` or `IS`.
- Lecturers, adjuncts, visiting, clinical, teaching professors, professors of practice, emeritus, fellows, staff, and students are excluded from final included faculty.
        """
    )

with st.sidebar:
    st.header("OpenAI settings")

    model = st.text_input(
        "Model",
        value="gpt-5.5",
        help="Use a model your API project supports. If unavailable, try gpt-4.1.",
    )

    use_web_search = st.checkbox(
        "Use hosted web search",
        value=True,
        help="Recommended for current faculty directory pages.",
    )

    require_search = st.checkbox(
        "Require web search",
        value=True,
        help="Forces the model to call web_search. Turn off only for pasted/static evidence.",
    )

    search_context_size = st.selectbox(
        "Search context size",
        ["low", "medium", "high"],
        index=1,
    )

    max_units = st.slider("Max units per school", 1, 80, 30, 1)
    max_faculty_per_unit = st.slider("Max people per unit", 10, 250, 120, 10)

    st.markdown("---")
    st.caption("Streamlit secret required:")
    st.code('OPENAI_API_KEY = "sk-..."', language="toml")


tab1, tab2, tab3 = st.tabs(["1. Schools", "2. Optional unit hints", "3. Run + export"])

with tab1:
    schools_df = dataframe_editor_for_schools()

with tab2:
    st.markdown(
        """
Optional. If you already know the school's units, paste them here.
This prevents the model from discovering extra units.

Format can be loose text, but this CSV-like format is best:

```text
unit_name,unit_url
Marketing,https://...
Management and Organizations,https://...
Operations Information and Decisions,https://...
```

You can leave this blank.
        """
    )
    provided_units_text = st.text_area(
        "Provided unit list / mapping hints",
        value="",
        height=220,
        placeholder="unit_name,unit_url\nMarketing,https://...\nManagement,https://...",
    )

with tab3:
    run = st.button("Run full mapping pipeline", type="primary")

    if run:
        client = get_client()

        all_units: List[dict] = []
        all_rosters: List[dict] = []
        all_records: List[dict] = []
        all_errors: List[dict] = []

        schools_clean = schools_df.fillna("").to_dict("records")
        schools_clean = [
            s for s in schools_clean
            if str(s.get("university", "")).strip() and str(s.get("school", "")).strip()
        ]

        if not schools_clean:
            st.error("Please enter at least one school.")
            st.stop()

        progress = st.progress(0)
        status = st.empty()

        for i, s in enumerate(schools_clean, start=1):
            school = SchoolConfig(
                university=str(s.get("university", "")).strip(),
                school=str(s.get("school", "")).strip(),
                faculty_directory_url=str(s.get("faculty_directory_url", "")).strip(),
                notes=str(s.get("notes", "")).strip(),
            )

            status.write(f"Running {school.university} — {school.school}")

            def cb(msg: str):
                status.write(f"{school.university} — {msg}")

            try:
                units, rosters, records, errors = run_school_pipeline(
                    client,
                    school,
                    model=model,
                    provided_units_text=provided_units_text,
                    max_units=max_units,
                    max_faculty_per_unit=max_faculty_per_unit,
                    use_web_search=use_web_search,
                    require_search=require_search,
                    search_context_size=search_context_size,
                    progress_callback=cb,
                )

                all_units.extend(units_to_dicts(units))
                all_rosters.extend(rosters)
                all_records.extend(records_to_dicts(records))
                for e in errors:
                    e["university"] = school.university
                    e["school"] = school.school
                    all_errors.append(e)

            except Exception as exc:
                all_errors.append(
                    {
                        "university": school.university,
                        "school": school.school,
                        "unit_name": "",
                        "unit_url": "",
                        "error": str(exc),
                    }
                )

            progress.progress(i / len(schools_clean))

        status.write("Done.")

        units_df = pd.DataFrame(all_units)
        roster_df = pd.DataFrame(all_rosters)
        final_df = pd.DataFrame(all_records)
        errors_df = pd.DataFrame(all_errors)

        st.session_state["units_df"] = units_df
        st.session_state["roster_df"] = roster_df
        st.session_state["final_df"] = final_df
        st.session_state["errors_df"] = errors_df

    units_df = st.session_state.get("units_df", pd.DataFrame())
    roster_df = st.session_state.get("roster_df", pd.DataFrame())
    final_df = st.session_state.get("final_df", pd.DataFrame())
    errors_df = st.session_state.get("errors_df", pd.DataFrame())

    if not units_df.empty or not final_df.empty or not errors_df.empty:
        st.subheader("A. Unit mapping")
        if not units_df.empty:
            st.dataframe(units_df, use_container_width=True)
        else:
            st.info("No unit mapping rows.")

        st.subheader("B. Raw roster audit")
        if not roster_df.empty:
            st.dataframe(roster_df, use_container_width=True)
        else:
            st.info("No roster rows.")

        st.subheader("C. Final included ladder faculty")
        if not final_df.empty:
            ordered_cols = [
                "university",
                "school",
                "name",
                "rank",
                "title",
                "local_unit",
                "ross_area",
                "subfield",
                "area_assignment_method",
                "confidence",
                "needs_review",
                "field_evidence",
                "profile_url",
                "source_url",
                "review_note",
            ]
            shown_cols = [c for c in ordered_cols if c in final_df.columns] + [
                c for c in final_df.columns if c not in ordered_cols
            ]
            final_df = final_df[shown_cols]
            st.dataframe(final_df, use_container_width=True)

            st.metric("Included ladder faculty", len(final_df))

            if "ross_area" in final_df.columns:
                counts = final_df["ross_area"].value_counts().rename_axis("ross_area").reset_index(name="count")
                st.subheader("Counts by Ross area")
                st.dataframe(counts, use_container_width=True)

            review_df = final_df[final_df.get("needs_review", False) == True] if "needs_review" in final_df.columns else pd.DataFrame()
            st.subheader("D. Manual review queue")
            if not review_df.empty:
                st.dataframe(review_df, use_container_width=True)
            else:
                st.success("No final faculty rows flagged for manual review.")
        else:
            st.info("No final included faculty rows.")

        st.subheader("E. Errors")
        if not errors_df.empty:
            st.dataframe(errors_df, use_container_width=True)
        else:
            st.success("No pipeline errors.")

        sheets = {
            "unit_mapping": units_df,
            "raw_roster_audit": roster_df,
            "final_included_faculty": final_df,
            "errors": errors_df,
        }
        if not final_df.empty and "needs_review" in final_df.columns:
            sheets["manual_review"] = final_df[final_df["needs_review"] == True]

        excel_bytes = to_excel_bytes(sheets)

        st.download_button(
            "Download Excel workbook",
            data=excel_bytes,
            file_name="ross_area_mapping_audit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if not final_df.empty:
            st.download_button(
                "Download final included faculty CSV",
                data=final_df.to_csv(index=False).encode("utf-8"),
                file_name="final_included_faculty.csv",
                mime="text/csv",
            )

        if not units_df.empty:
            st.download_button(
                "Download unit mapping CSV",
                data=units_df.to_csv(index=False).encode("utf-8"),
                file_name="unit_mapping.csv",
                mime="text/csv",
            )
