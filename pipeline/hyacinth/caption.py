"""Malayalam + English captions.

Facts stay in code (aggregate.jam_facts). The model (OpenAI, OPENAI_MODEL) only
phrases them, under a JSON schema, and is told to use no names that are not in the facts. A deterministic
template is used when no API credentials are available or the call fails, and
the uncertainty disclaimer is always appended in code.
"""

from __future__ import annotations

import json
import os

import openai

DISCLAIMER_EN = "Drift likelihood from an uncalibrated simulation, not a guarantee."
DISCLAIMER_ML = "ഇത് ഒരു സിമുലേഷൻ അടിസ്ഥാനമാക്കിയുള്ള സാധ്യതാ പ്രവചനം മാത്രമാണ്, ഉറപ്പല്ല."
SYNTHETIC_EN = "[TEST RUN ON SYNTHETIC DATA - NOT A REAL FORECAST]"
SYNTHETIC_ML = "[പരീക്ഷണം മാത്രം - യഥാർത്ഥ പ്രവചനമല്ല]"

SCHEMA = {
    "type": "object",
    "properties": {
        "ml": {"type": "string", "description": "Malayalam caption, Malayalam script"},
        "en": {"type": "string", "description": "English caption"},
    },
    "required": ["ml", "en"],
    "additionalProperties": False,
}

SYSTEM = """You write the daily caption for Hyacinth, a free forecast of where water hyacinth \
mats are likely to jam Vembanad lake and the Kuttanad canals in Alappuzha. \
Readers are ferry and houseboat operators, fishers and panchayat clearing crews who see it \
forwarded on WhatsApp, often on a phone.

Rules:
- Use ONLY the facts in the JSON you are given. Never add a place, canal, time, number or \
cause that is not in it.
- Refer to a segment by its "place" and/or "waterway" exactly as given. In Malayalam, use \
"place_ml" / "waterway_ml" when present; otherwise keep the English name unchanged in Latin \
script. Never transliterate or translate a name yourself.
- For timing use "by_day_en" in English and "by_day_ml" in Malayalam (add -യോടെ for "by").
- Say "likely" / "expected"; this is a likelihood, not a certainty.
- Mention at most the top 3 segments. Each caption at most 3 short sentences, plain words, \
no hashtags, no emoji.
- If there are no segments, say no new jams are expected in the next {hours} hours.
- Do not add any disclaimer; it is appended separately."""


def _template(facts: dict) -> dict:
    segs = facts["segments"][:3]
    hours = facts["window_hours"]
    if not segs:
        en = f"No new hyacinth jams expected in the next {hours} hours."
        ml = f"അടുത്ത {hours} മണിക്കൂറിൽ പുതിയ കുളവാഴ കെട്ട് പ്രതീക്ഷിക്കുന്നില്ല."
        return {"en": en, "ml": ml}
    by_day: dict[str, list[dict]] = {}
    for s in sorted(segs, key=lambda s: s["eta_hours"]):
        by_day.setdefault(s["by_day_en"], []).append(s)
    en_parts, ml_parts = [], []
    for day, group in by_day.items():
        names_en = [" / ".join(x for x in (s["place"], s["waterway"]) if x) for s in group]
        names_ml = [" / ".join(x for x in (s["place_ml"] or s["place"], s["waterway_ml"] or s["waterway"]) if x)
                    for s in group]
        day_ml = group[0]["by_day_ml"]
        if len(names_en) == 1:
            en_list, noun = names_en[0], "stretch"
        else:
            en_list, noun = ", ".join(names_en[:-1]) + " and " + names_en[-1], "stretches"
        en_parts.append(f"Mats likely to block the {en_list} {noun} by {day}.")
        ml_parts.append(f"{', '.join(names_ml)} ഭാഗങ്ങളിൽ {day_ml}യോടെ കുളവാഴ നിറഞ്ഞ് ഗതാഗതം തടസ്സപ്പെടാൻ സാധ്യത."
                        if len(names_ml) > 1 else
                        f"{names_ml[0]} ഭാഗത്ത് {day_ml}യോടെ കുളവാഴ നിറഞ്ഞ് ഗതാഗതം തടസ്സപ്പെടാൻ സാധ്യത.")
    return {"en": " ".join(en_parts), "ml": " ".join(ml_parts)}


def _model_caption(facts: dict, model: str) -> dict:
    client = openai.OpenAI()  # reads OPENAI_API_KEY
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=4000,
        response_format={"type": "json_schema", "json_schema": {"name": "caption", "strict": True, "schema": SCHEMA}},
        messages=[
            {"role": "system", "content": SYSTEM.replace("{hours}", str(facts["window_hours"]))},
            {"role": "user", "content": "Facts:\n" + json.dumps(facts, ensure_ascii=False, indent=1)},
        ],
    )
    choice = resp.choices[0]
    if choice.message.refusal:
        raise RuntimeError("model declined")
    if choice.finish_reason == "length":
        raise RuntimeError("caption truncated")
    out = json.loads(choice.message.content or "")
    if not out.get("en") or not out.get("ml"):
        raise RuntimeError("empty caption")
    return {"en": out["en"].strip(), "ml": out["ml"].strip()}


def make_caption(facts: dict, use_model: bool = True) -> dict:
    body, source = None, "template"
    model = os.environ.get("OPENAI_MODEL")
    if use_model and os.environ.get("OPENAI_API_KEY") and model:
        try:
            body, source = _model_caption(facts, model), model
        except (openai.OpenAIError, RuntimeError, json.JSONDecodeError) as e:
            print(f"[caption] model caption failed ({type(e).__name__}: {e}); using template")
    if body is None:
        body = _template(facts)

    ml = f"{body['ml']}\n{DISCLAIMER_ML}"
    en = f"{body['en']}\n{DISCLAIMER_EN}"
    if facts.get("synthetic"):
        ml, en = f"{SYNTHETIC_ML}\n{ml}", f"{SYNTHETIC_EN}\n{en}"
    meta = f"Barrage {facts['barrage']} · wind mostly from the {facts['wind']['prevailing_from']}, " \
           f"avg {facts['wind']['mean_speed_ms']} m/s"
    if facts.get("scene_date"):
        meta += f" · satellite {facts['scene_date']}"
    text = f"Hyacinth · {facts['forecast_start_ist']} · next {facts['window_hours']} h\n\n{ml}\n\n{en}\n\n{meta}"
    return {"text": text, "ml": ml, "en": en, "source": source}
