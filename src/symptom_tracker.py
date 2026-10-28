"""
symptom_tracker.py
Lightweight symptom-tracking module for the PMOS Screening Aid.
Stores entries as JSON in the project root. No personal data collected.
"""
import json
import datetime
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "symptom_log.json"

SYMPTOMS = [
    "Irregular or missed periods",
    "Heavy or painful periods",
    "Acne / oily skin",
    "Excess hair growth (face, chest, back)",
    "Hair thinning / hair loss",
    "Weight gain",
    "Difficulty losing weight",
    "Skin darkening (neck, groin, underarms)",
    "Fatigue",
    "Mood changes / anxiety",
    "Pelvic pain",
    "Bloating",
]


def load_log():
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_log(entries: list):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def add_entry(symptoms_checked: list, notes: str, date_str: str):
    entries = load_log()
    entries.append({
        "date": date_str,
        "symptoms": symptoms_checked,
        "notes": notes.strip(),
    })
    save_log(entries)


def delete_entry(index: int):
    entries = load_log()
    if 0 <= index < len(entries):
        entries.pop(index)
        save_log(entries)


def render_symptom_tracker():
    import streamlit as st
    import pandas as pd

    st.markdown(
        '<div class="sec-title" style="font-size:1.3rem">🗓 Symptom Tracker</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Record symptoms you experience over time. This is a personal log — "
        "no single symptom confirms PMOS. Share with your doctor."
    )

    # ── Record new entry ──────────────────────────────────────────────
    with st.expander("➕ Record today's symptoms", expanded=True):
        date_val = st.date_input(
            "Date", value=datetime.date.today(), key="st_date"
        )
        checked = st.multiselect(
            "Symptoms present today",
            SYMPTOMS,
            key="st_symptoms",
            placeholder="Select all that apply…",
        )
        notes = st.text_area(
            "Notes (optional)",
            key="st_notes",
            placeholder="Anything else to note…",
            height=80,
        )
        if st.button("💾 Save entry", type="primary", key="st_save"):
            if not checked:
                st.warning("Please select at least one symptom before saving.")
            else:
                add_entry(checked, notes, str(date_val))
                st.success(f"Entry saved for {date_val}.")
                st.rerun()

    # ── History ───────────────────────────────────────────────────────
    entries = load_log()
    if not entries:
        st.info("No entries yet. Record your first entry above.")
        return

    st.markdown("---")
    st.markdown(f"**{len(entries)} entries recorded**")

    # Trend chart
    rows = []
    for e in entries:
        for s in e["symptoms"]:
            rows.append({"date": e["date"], "symptom": s})
    if rows:
        df = pd.DataFrame(rows)
        counts = (
            df.groupby(["date", "symptom"])
            .size()
            .reset_index(name="count")
        )
        # Symptom frequency over time
        pivot = (
            df.groupby("date")["symptom"]
            .count()
            .reset_index(name="total_symptoms")
        )
        pivot["date"] = pd.to_datetime(pivot["date"])
        pivot = pivot.sort_values("date")
        st.markdown("**Symptom count per entry**")
        st.bar_chart(pivot.set_index("date")["total_symptoms"])

    # Table
    st.markdown("**Full history**")
    for i, e in enumerate(reversed(entries)):
        idx = len(entries) - 1 - i
        with st.expander(f"📅 {e['date']} — {len(e['symptoms'])} symptom(s)"):
            st.write("**Symptoms:**", ", ".join(e["symptoms"]))
            if e.get("notes"):
                st.write("**Notes:**", e["notes"])
            if st.button("🗑 Delete this entry", key=f"del_{idx}"):
                delete_entry(idx)
                st.rerun()

