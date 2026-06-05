# Ross Area & Faculty Toolkit — GPT/OpenAI exact-functionality port

This package is designed to preserve the Claude app's UI and workflow while using OpenAI/GPT.

## Files

- `streamlit_app_gpt.py` — GPT Streamlit app. Use this as the Streamlit Cloud main file.
- `streamlit_app.py` — same app under the original Claude app filename, for convenience.
- `ross_toolkit_gpt.py` — GPT/OpenAI backend with the same public function names used by the UI.
- `requirements.txt` — Streamlit Cloud dependency file.
- `requirements_gpt.txt` — duplicate for compatibility with your previous naming.
- `PROMPTS.md` — prompt-design notes and the key corrected rule.

## Streamlit Cloud setup

Set main file:

```text
streamlit_app_gpt.py
```

Set Secrets:

```toml
OPENAI_API_KEY = "sk-your-real-key"
```

Make sure the dependency file is named exactly:

```text
requirements.txt
```

## Functionality preserved from the Claude app

The app keeps the same two-page workflow:

1. **Match Areas**
   - Load built-in US News Top 50 or upload a school list.
   - For each school, enumerate the school's own departments / academic areas / research groups.
   - Map each department to zero, one, or multiple Ross areas.
   - Auto-check missing Ross areas.
   - Display a coverage matrix.
   - Allow per-department `Check faculty individually` toggles.
   - Export `ross_area_matches.csv`.

2. **Find Faculty**
   - Use Stage-1 matches or upload a Stage-1 CSV.
   - Pull current ladder faculty from official directories.
   - Unique-match departments inherit their single Ross area.
   - Flagged/mixed departments are classified person-by-person.
   - Export `ross_faculty_by_area.csv`.

## Corrected mapping rule

The key rule is department-first, not publication-overrides-department:

- Marketing department + a few OM/interface papers -> **Marketing**.
- Finance department + economics-adjacent papers -> **Finance**.
- Broad Management group containing OB + Strategy -> split person-by-person into **Management and Organizations** vs **Strategy**.
- Operations / IS / Decision Sciences group -> **Technology and Operations**; OM/IS is only the `T&O Subfield`.

