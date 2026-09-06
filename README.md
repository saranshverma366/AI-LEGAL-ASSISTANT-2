# AI Legal Assistant (India) — BNS 2023

A single-file, self-contained AI legal assistant built around the **Bharatiya
Nyaya Sanhita (BNS), 2023** — the law that replaced the Indian Penal Code on
1 July 2024.

> ⚠️ **This is a general-information tool, not a law firm.** Nothing it
> outputs is a substitute for advice from a licensed advocate. Always verify
> section numbers and procedure against the official Bare Act or a lawyer
> before relying on this for a real case. For free legal aid, call **15100**.

## What's inside

| File | What it is |
|---|---|
| `app.py` | **Everything integrated**: Flask backend + BERT semantic search + FIR builder + the frontend (HTML/CSS/JS embedded and served directly). This is the file you run. |
| `legal_database.json` | Standalone copy of the full database: 15 legal-advice playbooks, 42 BNS sections (with old IPC numbers & punishments), and DLSA/helpline data. |
| `frontend.html` | Standalone copy of just the frontend, for reference or if you want to host it separately / restyle it. It calls the same relative `/api/...` endpoints as `app.py`, so it needs to be served by (or proxied to) the Flask backend to actually work — opening it as a bare local file won't reach the API. |
| `requirements.txt` | Python dependencies. |

## How the "BERT" part works

The assistant matches a user's free-text question ("my phone was snatched
yesterday...") to the right legal-advice entry using **sentence embeddings**
from `all-MiniLM-L6-v2`, a distilled BERT model from the
`sentence-transformers` library — the standard, lightweight choice for this
exact task (semantic similarity search).

- If `sentence-transformers` is installed and can download the model
  (~80MB, needs internet once), the app uses real BERT embeddings.
- If not (e.g. no internet, or you didn't install it), the app **automatically
  falls back** to a TF-IDF + cosine-similarity matcher (`scikit-learn`), so it
  always works end-to-end. You'll see which backend is active in the top-right
  badge of the UI, and at `GET /api/status`.

This fallback is why the app was fully tested and runs correctly in this
sandbox even without internet access — you don't need to do anything extra
to try it locally; installing `sentence-transformers` just upgrades the
matching quality.

## Running it

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. That's it — one process,
one port, frontend and backend together.

## What the app does

1. **Ask a Legal Question** — type a plain-English description of a problem
   (theft, domestic violence, cyber fraud, harassment, tenant dispute, etc.).
   The model returns the closest-matching guidance, a confidence score, and
   the exact BNS sections that apply.
2. **BNS Sections** — search 42 curated sections of the Bharatiya Nyaya
   Sanhita by number, offence name, or old IPC number, each with the
   description and punishment.
3. **DLSA / Free Legal Aid** — national helplines (NALSA 15100, Cyber Crime
   1930, Women's Helpline 181, Police 112, Child Helpline 1098, Elder Line
   14567) plus an explanation of how the NALSA → SLSA → DLSA structure works
   and how to apply for a free panel lawyer.
4. **FIR Builder** — a structured form (complainant, incident, accused,
   evidence, witnesses) that generates a clean, print-ready **draft**
   complaint you can take to the police station, citing the relevant BNS
   sections. You can click a section chip in the "Ask" tab to auto-add it
   to the FIR draft.

## Accuracy notes on the legal data

Every BNS section number and old-IPC cross-reference in `legal_database.json`
was checked against multiple independent legal sources (PRS Legislative
Research, NCRB training material, NALSA's own site, and specialist legal
sites) as of September 2026. A few areas to flag honestly:

- The database covers **42 commonly-needed sections**, not all 358 sections
  of the BNS. It's meant as a practical assistant for the most common
  situations (theft, assault, cheating, domestic cruelty, sexual offences,
  cybercrime-adjacent offences), not a full legal code.
- The **DLSA locator** ships with verified *national* helpline numbers and
  one confirmed sample district entry (New Delhi). India has 700+ districts,
  each with its own DLSA — full district-by-district contact data would need
  to be scraped/maintained from each State Legal Services Authority's
  website, which wasn't fabricated here to avoid giving you wrong addresses.
  Calling **15100** connects you automatically to the correct DLSA for your
  location, and the JSON structure (`dlsa_locator`) is ready for you to
  extend with verified per-district records.
- Punishments described are the **base provisions**; many sections have
  aggravated sub-clauses (repeat offenders, use of weapons, victim being a
  minor, etc.) that increase the sentence — the descriptions note the main
  ones but always check the full Bare Act text for a real matter.

## Extending the database

Everything the model searches over comes from `legal_database.json`. To add
a new topic:

```json
{
  "id": "adv_new_topic",
  "category": "Your Category",
  "title": "Short title of the issue",
  "keywords": ["a few", "search terms"],
  "advice": "Step-by-step guidance...",
  "related_bns": ["<section numbers that apply>"]
}
```

Restart `app.py` and it's immediately searchable — no retraining needed,
since the matcher just re-embeds the corpus at startup.
