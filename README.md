# Ross Area Mapper — GPT/OpenAI corrected version

## Files

- `streamlit_app_gpt.py` — Streamlit UI
- `ross_toolkit_gpt.py` — OpenAI/Responses API engine
- `requirements.txt` — dependencies for Streamlit Cloud

## Streamlit Cloud setup

Set main file:

```text
streamlit_app_gpt.py
```

Set Secrets:

```toml
OPENAI_API_KEY = "sk-your-real-key"
```

## Classification rule

This app is department-first:

- Clear Marketing department -> Marketing.
- Clear Finance department -> Finance.
- Clear Accounting department -> Accounting.
- Clear T&O/OID/Decision Sciences/IS/Operations group -> Technology & Operations.
- Publications are used to split only genuinely mixed units.
- A Marketing faculty member with a few OM papers remains Marketing.
- A broad Management group can split into Management & Organizations vs Strategy.
- A T&O group can split subfield OM vs IS, but final Ross area remains Technology & Operations.
