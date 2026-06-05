# Prompt design for the GPT port

The GPT port keeps the Claude tool's original structure:

```text
Stage 1: target school's own departments -> Ross area(s)
Stage 2: department roster -> ladder faculty -> conditional person-level classification
```

## Stage 1 prompt goal

The model must enumerate the school's own academic units and map each unit to Ross's seven areas:

```text
Accounting
Business Economics and Public Policy
Finance
Management and Organizations
Marketing
Strategy
Technology and Operations
```

Stage 1 intentionally allows multi-area departments. This is necessary for cases such as:

```text
Management -> Management and Organizations + Strategy
Accounting & Finance -> Accounting + Finance
Operations, Information & Decisions -> Technology and Operations, to_kind = OM+IS
```

## Stage 2 prompt correction

The important correction is that person-level classification is constrained by the department mapping.

The old unconstrained instruction was dangerous:

```text
Classify each person by the journals THEY publish in. The unit's likely Ross areas are a hint, not a constraint.
```

The corrected instruction is:

```text
If ALLOWED_ROSS_AREAS has one area, every faculty member inherits that area.
Do not override a clear department because of cross-field or interface publications.

If ALLOWED_ROSS_AREAS has multiple areas, classify each person into exactly one of those allowed areas only.
Do not assign outside the allowed set.

If ALLOWED_ROSS_AREAS is Technology and Operations, keep ross_area = Technology and Operations for everyone.
Use OM/IS only as subfield.
```

## Examples

```text
Marketing department + a few OM papers -> Marketing
Management group with OB and Strategy people -> split M&O vs Strategy
Operations / IS / Decision Sciences group -> Technology and Operations, subfield OM or IS
```

This preserves the Claude app's function while preventing GPT from overusing publication signals.
