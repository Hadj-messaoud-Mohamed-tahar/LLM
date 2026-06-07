from __future__ import annotations

import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


REFUSAL_PHRASE = "Je suis désolé, mais je ne peux répondre qu'aux questions concernant la plateforme Creaexpertech et ses services."

STOPWORDS = {
    "a",
    "à",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "cette",
    "cet",
    "c",
    "ça",
    "dans",
    "de",
    "des",
    "du",
    "en",
    "et",
    "est",
    "faire",
    "il",
    "elle",
    "on",
    "je",
    "tu",
    "nous",
    "vous",
    "ils",
    "elles",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "ma",
    "mon",
    "mes",
    "ta",
    "ton",
    "tes",
    "sa",
    "son",
    "ses",
    "un",
    "une",
    "ou",
    "où",
    "pour",
    "par",
    "pas",
    "plus",
    "moins",
    "sur",
    "se",
    "s",
    "y",
    "d",
    "l",
    "m",
    "t",
    "qu",
    "que",
    "qui",
    "quoi",
    "dont",
    "est-ce",
    "estce",
    "comment",
    "combien",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "peux",
    "peut",
    "pouvez",
    "peux-tu",
    "svp",
    "stp",
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
}

SHORT_TOKEN_ALLOWLIST = {"ia", "rh", "seo", "cv", "rgpd", "sso", "opco"}


def clean_text(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\wàâäéèêëîïôöùûüç0-9\s-]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    t = clean_text(text)
    out: List[str] = []
    for x in re.split(r"\s+", t):
        tok = str(x).strip()
        if not tok:
            continue
        if tok in STOPWORDS:
            continue
        if len(tok) < 3 and (tok not in SHORT_TOKEN_ALLOWLIST) and (not tok.isdigit()):
            continue
        out.append(tok)
    return out


def score_text(haystack: str, query: str) -> int:
    hay_tokens = set(tokenize(haystack))
    q_tokens = tokenize(query)
    score = 0
    for tok in q_tokens:
        if tok in hay_tokens:
            score += 1
    return score


def parse_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    s2 = s.strip("[](){}")
    parts = re.split(r"[;,|]\s*|\s{2,}", s2)
    out: List[str] = []
    for p in parts:
        p2 = str(p).strip().strip('"').strip("'")
        if p2:
            out.append(p2)
    return out


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        s = str(value).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class ModuleRow:
    title: str
    description: str
    difficulty: str
    duration_minutes: Optional[int]
    tags: Tuple[str, ...]
    job: str
    tool: str


@dataclass(frozen=True)
class ToolRow:
    title: str
    promise: str
    type: str
    metier: str
    niveau: str
    plan_required: str
    tags: Tuple[str, ...]


def format_module_line(m: ModuleRow) -> str:
    title = safe_str(m.title)
    desc = safe_str(m.description)
    tags = ", ".join([x for x in list(m.tags) if safe_str(x)][:8])
    difficulty = safe_str(m.difficulty)
    dur = f"{m.duration_minutes} min" if isinstance(m.duration_minutes, int) else ""
    job = safe_str(m.job)
    tool = safe_str(m.tool)
    meta = " | ".join([x for x in [difficulty, dur, f"métier:{job}" if job else "", f"outil:{tool}" if tool else ""] if x])
    tail = " | ".join([x for x in [meta, f"tags:{tags}" if tags else ""] if x])
    return f"- {title}{f' — {desc}' if desc else ''}{f' ({tail})' if tail else ''}"


def format_tool_line(t: ToolRow, plan: str) -> str:
    title = safe_str(t.title)
    promise = safe_str(t.promise)
    tags = ", ".join([x for x in list(t.tags) if safe_str(x)][:8])
    pr = safe_str(t.plan_required) or "discovery"
    unlocked = True if plan == "pro" else (pr in ("", "discovery", "free", "gratuit", "basic", "essential"))
    meta = " | ".join(
        [
            x
            for x in [
                f"type:{safe_str(t.type)}" if safe_str(t.type) else "",
                f"métier:{safe_str(t.metier)}" if safe_str(t.metier) else "",
                f"niveau:{safe_str(t.niveau)}" if safe_str(t.niveau) else "",
                f"plan:{pr}" if pr else "",
                "débloqué:oui" if unlocked else "débloqué:non",
            ]
            if x
        ]
    )
    tail = " | ".join([x for x in [meta, f"tags:{tags}" if tags else ""] if x])
    best_desc = promise
    return f"- {title}{f' — {best_desc}' if best_desc else ''}{f' ({tail})' if tail else ''}"


def build_system_prompt(plan: str, module_lines: str, tool_lines: str, pricing_lines: str) -> str:
    return (
        'Tu es l’assistant virtuel officiel "Assistant Creaexpertech" de la plateforme Creaexpertech.\n'
        "Règles strictes :\n"
        "0) Cette requête a déjà été validée comme étant dans le périmètre Creaexpertech. Tu dois répondre et ne pas refuser.\n"
        "1) Tu réponds UNIQUEMENT aux questions sur Creaexpertech : plateforme, formations/modules, outils, abonnement, paiement, crédits, accès, fonctionnement.\n"
        "2) Si la question est hors sujet, tu dois répondre EXACTEMENT : \"Je suis désolé, mais je ne peux répondre qu'aux questions concernant la plateforme Creaexpertech et ses services.\" (et rien d’autre).\n"
        "3) N’invente jamais. Si l’info n’est pas dans le contexte ou si tu n’es pas sûr, dis-le et propose où trouver l’info dans la plateforme (Catalogue, Boîte à outils, Tarifs).\n"
        "4) Quand c’est pertinent, propose 2 à 4 recommandations de formations en priorité, puis éventuellement 1 à 2 outils, uniquement parmi les éléments listés dans le contexte.\n"
        "\n"
        f"Contexte à jour (plan actuel utilisateur : {plan})\n"
        "Formations (extraits pertinents) :\n"
        f"{module_lines if module_lines else '- (aucune formation disponible)'}\n\n"
        "Outils (extraits pertinents) :\n"
        f"{tool_lines if tool_lines else '- (aucun outil disponible)'}\n\n"
        "Tarifs / Abonnements (extraits pertinents) :\n"
        f"{pricing_lines if pricing_lines else '- (aucune info tarifs disponible)'}\n"
    )


def is_greeting(text: str) -> bool:
    return bool(re.match(r"^(bonjour|bonsoir|salut|salam|hello|hey|yo|coucou|cc|wesh)(\s|!|\.|,|$)", str(text or ""), flags=re.IGNORECASE))


def is_navigation_question(text: str) -> bool:
    return bool(re.search(r"\b(acceder|accéder|ouvrir|aller|ou|où|page|lien|liens|trouver)\b", str(text or ""), flags=re.IGNORECASE))


def wants_subscription(text: str) -> bool:
    return bool(
        re.search(
            r"\b(abonnement|abonnements|paiement|payer|prix|tarif|tarifs|facture|carte|stripe|plan|essai|trial|checkout|annuler|résilier|resilier|upgrade|downgrade)\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def wants_tools(text: str) -> bool:
    return bool(re.search(r"\b(outil|outils|boite|boîte|template|prompt|kit|workflow|checklist)\b", str(text or ""), flags=re.IGNORECASE))


def wants_catalogue(text: str) -> bool:
    return bool(re.search(r"\b(formation|formations|module|modules|cours|catalogue|certificat|certificats)\b", str(text or ""), flags=re.IGNORECASE))


def platform_keyword_hit(text: str) -> bool:
    platform_keywords = {
        "creaexpertech",
        "plateforme",
        "formation",
        "formations",
        "module",
        "modules",
        "outil",
        "outils",
        "abonnement",
        "abonnements",
        "paiement",
        "facture",
        "prix",
        "plan",
        "crédit",
        "credits",
        "débloquer",
        "debloquer",
        "catalogue",
        "boite",
        "boîte",
    }
    return any(tok in platform_keywords for tok in tokenize(text))


def strip_html(fragment: str) -> str:
    s = str(fragment or "")
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</p\s*>", "\n", s)
    s = re.sub(r"(?is)</div\s*>", "\n", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = unescape(s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_div_block(html: str, start_at: int) -> str:
    i = html.find("<div", start_at)
    if i < 0:
        return ""
    depth = 0
    for m in re.finditer(r"(?is)<div\b|</div\s*>", html[i:]):
        token = m.group(0).lower()
        if token.startswith("<div"):
            depth += 1
        else:
            depth -= 1
        if depth == 0:
            end = i + m.end()
            return html[i:end]
    return html[i:]


@dataclass(frozen=True)
class PricingPlan:
    name: str
    tagline: str
    audience: str
    price: str
    annual_hint: str
    access: Tuple[str, ...]
    extras: Tuple[str, ...]
    kind: str  # "b2c" | "b2b"


def _parse_pricing_plans_from_html(html: str) -> Dict[str, Any]:
    plans: List[PricingPlan] = []

    annual_discount = ""
    m = re.search(r"Tous les plans annuels[^<]+</div>", html, flags=re.IGNORECASE)
    if m:
        annual_discount = strip_html(m.group(0))

    for label, kind in [("DÉCOUVERTE", "b2c"), ("ESSENTIEL", "b2c"), ("PRATICIEN", "b2c"), ("AVANCÉ", "b2c")]:
        idx = html.find(f"<!-- {label}")
        if idx < 0:
            continue
        block = _extract_div_block(html, idx)
        if not block:
            continue

        name = strip_html(re.search(r'(?is)<div class="plan-name"[^>]*>(.*?)</div>', block).group(1)) if re.search(r'(?is)<div class="plan-name"[^>]*>(.*?)</div>', block) else label
        tagline = strip_html(re.search(r'(?is)<div class="plan-tagline"[^>]*>(.*?)</div>', block).group(1)) if re.search(r'(?is)<div class="plan-tagline"[^>]*>(.*?)</div>', block) else ""
        audience = strip_html(re.search(r'(?is)<span class="plan-reco"[^>]*>(.*?)</span>', block).group(1)) if re.search(r'(?is)<span class="plan-reco"[^>]*>(.*?)</span>', block) else ""

        price_num = strip_html(re.search(r'(?is)<span class="price-num"[^>]*>(.*?)</span>', block).group(1)) if re.search(r'(?is)<span class="price-num"[^>]*>(.*?)</span>', block) else ""
        price_unit = strip_html(re.search(r'(?is)<span class="price-unit"[^>]*>(.*?)</span>', block).group(1)) if re.search(r'(?is)<span class="price-unit"[^>]*>(.*?)</span>', block) else ""
        price = f"{price_num}{price_unit}".strip()

        annual_hint = strip_html(re.search(r'(?is)<div class="price-annual"[^>]*>(.*?)</div>', block).group(1)) if re.search(r'(?is)<div class="price-annual"[^>]*>(.*?)</div>', block) else ""

        access: List[str] = []
        for m2 in re.finditer(
            r'(?is)<span class="tier-pill[^"]*">(.*?)</span>\s*<div class="access-text[^"]*">(.*?)</div>',
            block,
        ):
            tier = strip_html(m2.group(1))
            txt = strip_html(m2.group(2))
            if tier and txt:
                access.append(f"{tier}: {txt}")

        extras: List[str] = []
        for m3 in re.finditer(r'(?is)<div class="extra-item"[^>]*>(.*?)</div>', block):
            t = strip_html(m3.group(1))
            if t:
                extras.append(t)

        plans.append(
            PricingPlan(
                name=name,
                tagline=tagline,
                audience=audience,
                price=price,
                annual_hint=annual_hint,
                access=tuple(access),
                extras=tuple(extras),
                kind=kind,
            )
        )

    for label in ["ÉQUIPES", "ORGANISATION"]:
        idx = html.find(f"<!-- {label}")
        if idx < 0:
            continue
        block = _extract_div_block(html, idx)
        if not block:
            continue

        name = strip_html(re.search(r'(?is)<div class="plan-name"[^>]*>(.*?)</div>', block).group(1)) if re.search(r'(?is)<div class="plan-name"[^>]*>(.*?)</div>', block) else label
        tagline = strip_html(re.search(r'(?is)<div class="plan-tagline"[^>]*>(.*?)</div>', block).group(1)) if re.search(r'(?is)<div class="plan-tagline"[^>]*>(.*?)</div>', block) else ""
        audience = strip_html(re.search(r'(?is)<span class="plan-reco"[^>]*>(.*?)</span>', block).group(1)) if re.search(r'(?is)<span class="plan-reco"[^>]*>(.*?)</span>', block) else ""

        if label == "ORGANISATION":
            price = strip_html(re.search(r'(?is)<div class="plan-price"[^>]*>(.*?)</div>', block).group(1)) if re.search(r'(?is)<div class="plan-price"[^>]*>(.*?)</div>', block) else "Sur devis"
            annual_hint = ""
        else:
            price_num = strip_html(re.search(r'(?is)<span class="price-num"[^>]*>(.*?)</span>', block).group(1)) if re.search(r'(?is)<span class="price-num"[^>]*>(.*?)</span>', block) else ""
            price_unit = strip_html(re.search(r'(?is)<span class="price-unit"[^>]*>(.*?)</span>', block).group(1)) if re.search(r'(?is)<span class="price-unit"[^>]*>(.*?)</span>', block) else ""
            price = f"{price_num}{price_unit}".strip()
            annual_hint = strip_html(re.search(r'(?is)Min\.\s*.*?</div>', block).group(0)) if re.search(r"(?is)Min\.\s*.*?</div>", block) else ""

        features: List[str] = []
        for m4 in re.finditer(r'(?is)<div class="bottom-item"[^>]*>(.*?)</div>', block):
            t = strip_html(m4.group(1))
            if t:
                features.append(t)

        plans.append(
            PricingPlan(
                name=name,
                tagline=tagline,
                audience=audience,
                price=price,
                annual_hint=annual_hint,
                access=tuple(features),
                extras=tuple(),
                kind="b2b",
            )
        )

    return {"plans": plans, "annual_discount": annual_discount}


@st.cache_data(show_spinner=False)
def load_pricing(pricing_html_path: str, cache_buster: str) -> Dict[str, Any]:
    p = Path(pricing_html_path)
    if not p.exists():
        return {"plans": [], "annual_discount": "", "source": pricing_html_path}
    html = p.read_text(encoding="utf-8", errors="ignore")
    parsed = _parse_pricing_plans_from_html(html)
    parsed["source"] = pricing_html_path
    return parsed


def answer_subscription_question(question: str, pricing: Dict[str, Any]) -> str:
    plans: List[PricingPlan] = pricing.get("plans") or []
    discount = safe_str(pricing.get("annual_discount"))
    if not plans:
        return "Je n'ai pas l'information tarifs dans la démo. Vérifie l’onglet Tarifs."

    scored: List[Tuple[int, PricingPlan]] = []
    for p in plans:
        hay = " ".join([p.name, p.tagline, p.audience, p.price, p.annual_hint, " ".join(p.access), " ".join(p.extras), p.kind])
        s = score_text(hay, question)
        if s > 0:
            scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [p for _, p in scored[:2]] if scored else [p for p in plans if p.kind == "b2c"][:4]

    lines: List[str] = []
    for p in selected:
        head = f"{p.name} — {p.price}".strip(" —")
        lines.append(head)
        if p.annual_hint:
            lines.append(f"- {p.annual_hint}")
        if p.tagline:
            lines.append(f"- {p.tagline}")
        if p.audience:
            lines.append(f"- {p.audience}")
        if p.access:
            for a in list(p.access)[:6]:
                lines.append(f"- {a}")
        if p.extras:
            for x in list(p.extras)[:4]:
                lines.append(f"- {x}")
        lines.append("")

    if discount:
        lines.append(discount)
    return "\n".join([x for x in lines]).strip()


def build_pricing_context(question: str, pricing: Dict[str, Any], k: int = 3) -> str:
    plans: List[PricingPlan] = pricing.get("plans") or []
    discount = safe_str(pricing.get("annual_discount"))
    if not plans:
        return ""

    scored: List[Tuple[int, PricingPlan]] = []
    for p in plans:
        hay = " ".join(
            [
                p.name,
                p.tagline,
                p.audience,
                p.price,
                p.annual_hint,
                " ".join(p.access),
                " ".join(p.extras),
                p.kind,
            ]
        )
        s = score_text(hay, question)
        if s > 0:
            scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [p for _, p in scored[: max(1, int(k))]] if scored else plans[: max(1, int(k))]

    out: List[str] = []
    for p in selected:
        head = f"- {p.name} — {p.price}".strip()
        out.append(head)
        if p.annual_hint:
            out.append(f"  - {p.annual_hint}")
        if p.tagline:
            out.append(f"  - {p.tagline}")
        if p.audience:
            out.append(f"  - {p.audience}")
        for a in list(p.access)[:8]:
            out.append(f"  - {a}")
        for x in list(p.extras)[:6]:
            out.append(f"  - {x}")
    if discount:
        out.append("")
        out.append(discount)
    return "\n".join(out).strip()


def ollama_get_models(base_url: str) -> List[str]:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=4)
        r.raise_for_status()
        data = r.json()
        models = data.get("models") if isinstance(data, dict) else None
        if isinstance(models, list):
            out = []
            for m in models:
                if isinstance(m, dict) and isinstance(m.get("name"), str):
                    out.append(m["name"])
            return sorted(list(dict.fromkeys(out)))
    except Exception:
        return []
    return []


def ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]]) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        return ""
    msg = data.get("message") or {}
    if isinstance(msg, dict):
        return str(msg.get("content") or "").strip()
    return ""


def extract_job_tool_from_content(content: str) -> Tuple[str, str]:
    s = str(content or "").strip()
    if not s or not s.startswith("{"):
        return "", ""
    try:
        obj = json.loads(s)
        meta = obj.get("meta") if isinstance(obj, dict) else None
        if isinstance(meta, dict):
            job = safe_str(meta.get("job"))
            tool = safe_str(meta.get("tool"))
            return job, tool
    except Exception:
        return "", ""
    return "", ""


@st.cache_data(show_spinner=False)
def load_csv_frames(modules_path: str, tools_path: str, cache_buster: str) -> Dict[str, Any]:
    modules_df = pd.read_csv(modules_path)
    tools_df = pd.read_csv(tools_path)
    return {"modules_raw": modules_df, "tools_raw": tools_df, "loaded_at": time.time(), "cache_buster": cache_buster}


def normalize_space(value: str) -> str:
    s = safe_str(value)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_tags_list(tags: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in tags:
        x = normalize_space(t)
        if not x:
            continue
        key = x.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def clean_modules_df(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df_raw.copy()

    for col in ["title", "description", "difficulty", "job", "tool"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_space)

    if "estimated_duration" in df.columns:
        df["duration_minutes"] = df["estimated_duration"].map(safe_int)
    else:
        df["duration_minutes"] = None

    if "tags" in df.columns:
        df["tags_list"] = df["tags"].map(parse_tags).map(clean_tags_list)
    else:
        df["tags_list"] = [[] for _ in range(len(df))]

    if "job" in df.columns and "tool" in df.columns and "content" in df.columns:
        missing_mask = (df["job"] == "") | (df["tool"] == "")
        for idx in df.index[missing_mask]:
            cj, ct = extract_job_tool_from_content(safe_str(df.at[idx, "content"]))
            if df.at[idx, "job"] == "":
                df.at[idx, "job"] = cj
            if df.at[idx, "tool"] == "":
                df.at[idx, "tool"] = ct

    if "difficulty" in df.columns:
        d = df["difficulty"].map(lambda x: normalize_space(x).lower())
        d = d.replace({"foundation": "fondation", "beginner": "débutant", "intermediate": "intermédiaire", "advanced": "avancé"})
        d = d.map(lambda x: x if x in ("débutant", "fondation", "intermédiaire", "avancé") else None)
        df["difficulty"] = d

    before_n = len(df_raw)
    title_key = df["title"].map(lambda x: normalize_space(x).casefold())
    df["_title_key"] = title_key
    dup_count = int(df.duplicated(subset=["_title_key"]).sum())
    df = df.drop_duplicates(subset=["_title_key"], keep="first").copy()
    df = df.drop(columns=["_title_key"], errors="ignore")

    cols = ["title", "description", "difficulty", "duration_minutes", "tags_list", "job", "tool"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df_clean = df[cols].copy()

    invalid_duration = int(df_clean["duration_minutes"].isna().sum())
    summary = {
        "before_rows": int(before_n),
        "after_rows": int(len(df_clean)),
        "duplicates_removed_title": int(dup_count),
        "invalid_duration_to_null": int(invalid_duration),
        "transforms": [
            "trim + normalisation espaces (title/description/difficulty/job/tool)",
            "normalisation casse (difficulty: mapping beginner/intermediate/advanced)",
            "split/clean tags (JSON list -> liste, trimming, dédoublonnage)",
            "duration_minutes: conversion en int, invalides -> null",
            "suppression doublons sur title (case-insensitive)",
            "job/tool: extraction depuis content.meta si manquant",
        ],
    }
    return df_clean, summary


def clean_tools_df(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df_raw.copy()

    for col in ["title", "promise", "short_description", "type", "metier", "niveau", "plan_required"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_space)

    if "promise" in df.columns:
        base = df["promise"]
    else:
        base = ""
    if "short_description" in df.columns:
        df["promise_final"] = base.where(base != "", df["short_description"])
    else:
        df["promise_final"] = base

    if "tags" in df.columns:
        df["tags_list"] = df["tags"].map(parse_tags).map(clean_tags_list)
    else:
        df["tags_list"] = [[] for _ in range(len(df))]

    if "niveau" in df.columns:
        n = df["niveau"].map(lambda x: normalize_space(x).lower())
        n = n.replace({"foundation": "fondation", "beginner": "débutant", "intermediate": "intermédiaire", "advanced": "avancé"})
        df["niveau"] = n.map(lambda x: x if x in ("débutant", "fondation", "intermédiaire", "avancé") else None)

    if "plan_required" in df.columns:
        df["plan_required"] = df["plan_required"].map(lambda x: normalize_space(x).lower())
        df["plan_required"] = df["plan_required"].replace({"free": "discovery"})
        df["plan_required"] = df["plan_required"].map(lambda x: x if x else "discovery")
    else:
        df["plan_required"] = "discovery"

    before_n = len(df_raw)
    title_key = df["title"].map(lambda x: normalize_space(x).casefold())
    df["_title_key"] = title_key
    dup_count = int(df.duplicated(subset=["_title_key"]).sum())
    df = df.drop_duplicates(subset=["_title_key"], keep="first").copy()
    df = df.drop(columns=["_title_key"], errors="ignore")

    cols = ["title", "promise_final", "type", "metier", "niveau", "plan_required", "tags_list"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df_clean = df[cols].copy()
    df_clean = df_clean.rename(columns={"promise_final": "promise"})

    summary = {
        "before_rows": int(before_n),
        "after_rows": int(len(df_clean)),
        "duplicates_removed_title": int(dup_count),
        "transforms": [
            "trim + normalisation espaces (title/promise/type/metier/niveau/plan_required)",
            "normalisation casse (niveau + plan_required)",
            "split/clean tags (JSON list -> liste, trimming, dédoublonnage)",
            "suppression doublons sur title (case-insensitive)",
        ],
    }
    return df_clean, summary


def build_catalog_objects(modules_df_clean: pd.DataFrame, tools_df_clean: pd.DataFrame) -> Dict[str, Any]:
    modules: List[ModuleRow] = []
    for _, row in modules_df_clean.iterrows():
        modules.append(
            ModuleRow(
                title=safe_str(row.get("title")),
                description=safe_str(row.get("description")),
                difficulty=safe_str(row.get("difficulty")),
                duration_minutes=safe_int(row.get("duration_minutes")),
                tags=tuple([safe_str(x) for x in (row.get("tags_list") or []) if safe_str(x)]),
                job=safe_str(row.get("job")),
                tool=safe_str(row.get("tool")),
            )
        )

    tools: List[ToolRow] = []
    for _, row in tools_df_clean.iterrows():
        tools.append(
            ToolRow(
                title=safe_str(row.get("title")),
                promise=safe_str(row.get("promise")),
                type=safe_str(row.get("type")),
                metier=safe_str(row.get("metier")),
                niveau=safe_str(row.get("niveau")),
                plan_required=safe_str(row.get("plan_required")) or "discovery",
                tags=tuple([safe_str(x) for x in (row.get("tags_list") or []) if safe_str(x)]),
            )
        )
    return {"modules": modules, "tools": tools}


def load_catalog(modules_path: str, tools_path: str, cache_buster: str) -> Dict[str, Any]:
    frames = load_csv_frames(modules_path, tools_path, cache_buster)
    modules_raw: pd.DataFrame = frames["modules_raw"]
    tools_raw: pd.DataFrame = frames["tools_raw"]

    modules_clean, modules_summary = clean_modules_df(modules_raw)
    tools_clean, tools_summary = clean_tools_df(tools_raw)
    objects = build_catalog_objects(modules_clean, tools_clean)
    return {
        **objects,
        "modules_raw": modules_raw,
        "tools_raw": tools_raw,
        "modules_clean": modules_clean,
        "tools_clean": tools_clean,
        "cleaning_summary": {"modules": modules_summary, "tools": tools_summary},
        "loaded_at": frames["loaded_at"],
        "cache_buster": cache_buster,
    }


def rank_modules(modules: List[ModuleRow], query: str, k: int = 10) -> List[ModuleRow]:
    scored = []
    for m in modules:
        hay = " ".join([m.title, m.description, " ".join(m.tags), m.difficulty, m.job, m.tool])
        s = score_text(hay, query)
        if s > 0:
            scored.append((s, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]


def rank_tools(tools: List[ToolRow], query: str, k: int = 10) -> List[ToolRow]:
    scored = []
    for t in tools:
        hay = " ".join([t.title, t.promise, t.type, t.metier, t.niveau, t.plan_required, " ".join(t.tags)])
        s = score_text(hay, query)
        if s > 0:
            scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 0:
        return 0.0
    return dot / denom


def ollama_embed(base_url: str, embed_model: str, text: str) -> Optional[List[float]]:
    prompt = safe_str(text)
    if not prompt:
        return None
    url = base_url.rstrip("/") + "/api/embeddings"
    payload = {"model": embed_model, "prompt": prompt}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding") if isinstance(data, dict) else None
        if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
            return [float(x) for x in emb]
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def embed_texts_cached(base_url: str, embed_model: str, texts: Tuple[str, ...], cache_buster: str) -> List[Optional[List[float]]]:
    out: List[Optional[List[float]]] = []
    for t in texts:
        out.append(ollama_embed(base_url, embed_model, t))
    return out


def rank_modules_embeddings(
    modules: List[ModuleRow],
    query: str,
    base_url: str,
    embed_model: str,
    cache_buster: str,
    enabled: bool,
    k: int = 10,
) -> List[ModuleRow]:
    if not enabled:
        return rank_modules(modules, query, k=k)
    item_texts = tuple([" ".join([m.title, m.description, " ".join(m.tags), m.difficulty, m.job, m.tool]).strip() for m in modules])
    item_embs = embed_texts_cached(base_url, embed_model, item_texts, cache_buster)
    q_emb = ollama_embed(base_url, embed_model, query)
    if q_emb is None or not any(item_embs):
        return rank_modules(modules, query, k=k)
    scored: List[Tuple[float, ModuleRow]] = []
    for m, e in zip(modules, item_embs):
        if e is None:
            continue
        scored.append((_cosine(q_emb, e), m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for s, m in scored if s > 0][:k]


def rank_tools_embeddings(
    tools: List[ToolRow],
    query: str,
    base_url: str,
    embed_model: str,
    cache_buster: str,
    enabled: bool,
    k: int = 10,
) -> List[ToolRow]:
    if not enabled:
        return rank_tools(tools, query, k=k)
    item_texts = tuple([" ".join([t.title, t.promise, t.type, t.metier, t.niveau, t.plan_required, " ".join(t.tags)]).strip() for t in tools])
    item_embs = embed_texts_cached(base_url, embed_model, item_texts, cache_buster)
    q_emb = ollama_embed(base_url, embed_model, query)
    if q_emb is None or not any(item_embs):
        return rank_tools(tools, query, k=k)
    scored: List[Tuple[float, ToolRow]] = []
    for t, e in zip(tools, item_embs):
        if e is None:
            continue
        scored.append((_cosine(q_emb, e), t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for s, t in scored if s > 0][:k]


def recommendation_answer(plan: str, selected_modules: List[ModuleRow], selected_tools: List[ToolRow]) -> str:
    top_mods = selected_modules[:4]
    top_tls = selected_tools[:2]
    lines: List[str] = []
    if top_mods:
        lines.append("Recommandations de formations :")
        for m in top_mods:
            lines.append(format_module_line(m))
    if top_tls:
        lines.append("")
        lines.append("Outils recommandés :")
        for t in top_tls:
            lines.append(format_tool_line(t, plan=plan))
    if not top_mods and not top_tls:
        lines.append("Je n'ai pas trouvé de contenu correspondant dans le catalogue pour le moment.")
    return "\n".join(lines).strip()


def ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "reload_token" not in st.session_state:
        st.session_state.reload_token = 0


def basic_data_audit(df: pd.DataFrame, key_cols: List[str]) -> Dict[str, Any]:
    n = int(len(df))
    missing_pct: Dict[str, float] = {}
    for c in df.columns:
        try:
            missing_pct[c] = float(df[c].isna().mean() * 100.0)
        except Exception:
            missing_pct[c] = 0.0

    dup_title = 0
    if "title" in df.columns:
        keys = df["title"].map(lambda x: normalize_space(x).casefold())
        dup_title = int(keys.duplicated().sum())

    lengths: Dict[str, Dict[str, float]] = {}
    for c in key_cols:
        if c not in df.columns:
            continue
        s = df[c].fillna("").map(lambda x: len(str(x)))
        if len(s) == 0:
            continue
        lengths[c] = {
            "min": float(s.min()),
            "p50": float(s.quantile(0.5)),
            "p90": float(s.quantile(0.9)),
            "max": float(s.max()),
            "mean": float(s.mean()),
        }

    return {"rows": n, "missing_pct": missing_pct, "dup_title": dup_title, "lengths": lengths}


def distribution_table(df: pd.DataFrame, column: str, top_n: int = 30) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    s = df[column].fillna("").map(lambda x: normalize_space(x))
    vc = s.value_counts(dropna=False).head(top_n)
    out = vc.reset_index()
    out.columns = [column, "count"]
    return out


@st.cache_data(show_spinner=False)
def load_eval_set(eval_path: str, cache_buster: str) -> List[Dict[str, Any]]:
    p = Path(eval_path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(data, list):
        out = []
        for x in data:
            if isinstance(x, dict) and isinstance(x.get("question"), str):
                out.append(x)
        return out
    return []


def predict_intent_rule(question: str) -> str:
    q = safe_str(question)
    if is_greeting(q):
        return "greeting"
    if wants_subscription(q):
        return "pricing"
    if wants_tools(q) and not wants_catalogue(q):
        return "tools"
    if wants_catalogue(q):
        return "catalogue"
    return "off_topic"


def compute_binary_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and (not p))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def evaluate_retrieval(
    eval_rows: List[Dict[str, Any]],
    modules: List[ModuleRow],
    tools: List[ToolRow],
    retrieval_mode: str,
    base_url: str,
    embed_model: str,
    embed_enabled: bool,
    cache_buster: str,
    k: int,
) -> Tuple[float, float, pd.DataFrame]:
    hits = []
    recalls = []
    details: List[Dict[str, Any]] = []

    for ex in eval_rows:
        q = safe_str(ex.get("question"))
        expected_items = ex.get("expected_items") or []
        expected_items = [safe_str(x) for x in expected_items] if isinstance(expected_items, list) else []
        if not expected_items:
            continue
        intent = safe_str(ex.get("intent"))
        if intent not in ("catalogue", "tools"):
            continue

        if intent == "catalogue":
            top = (
                rank_modules_embeddings(modules, q, base_url=base_url, embed_model=embed_model, cache_buster=cache_buster, enabled=embed_enabled, k=k)
                if retrieval_mode == "embeddings"
                else rank_modules(modules, q, k=k)
            )
            top_titles = [m.title for m in top]
        else:
            top = (
                rank_tools_embeddings(tools, q, base_url=base_url, embed_model=embed_model, cache_buster=cache_buster, enabled=embed_enabled, k=k)
                if retrieval_mode == "embeddings"
                else rank_tools(tools, q, k=k)
            )
            top_titles = [t.title for t in top]

        expected_set = {safe_str(x).casefold() for x in expected_items if safe_str(x)}
        got_set = {safe_str(x).casefold() for x in top_titles if safe_str(x)}
        hit = bool(expected_set & got_set)
        recall = (len(expected_set & got_set) / len(expected_set)) if expected_set else 0.0
        hits.append(hit)
        recalls.append(recall)
        details.append(
            {
                "question": q,
                "intent": intent,
                "expected_items": " | ".join(expected_items),
                "top_k": " | ".join(top_titles),
                "hit": hit,
                "recall": recall,
            }
        )

    hit_at_k = sum(1 for x in hits if x) / len(hits) if hits else 0.0
    recall_at_k = sum(recalls) / len(recalls) if recalls else 0.0
    return float(hit_at_k), float(recall_at_k), pd.DataFrame(details)


def run_evaluation(
    eval_rows: List[Dict[str, Any]],
    modules: List[ModuleRow],
    tools: List[ToolRow],
    pricing: Dict[str, Any],
    plan: str,
    base_url: str,
    chat_model: str,
    embed_model: str,
    embed_enabled: bool,
    cache_buster: str,
    retrieval_mode: str,
    k: int,
    use_ollama_for_latency: bool,
) -> Dict[str, Any]:
    y_true_in_scope: List[bool] = []
    y_pred_in_scope: List[bool] = []
    y_true_refusal: List[bool] = []
    y_pred_refusal: List[bool] = []
    latencies_ms: List[float] = []

    per_rows: List[Dict[str, Any]] = []

    for ex in eval_rows:
        q = safe_str(ex.get("question"))
        label_intent = safe_str(ex.get("intent"))
        expected_refusal = bool(ex.get("expected_refusal"))
        expected_items = ex.get("expected_items") or []
        expected_items = [safe_str(x) for x in expected_items] if isinstance(expected_items, list) else []

        t0 = time.perf_counter()
        pred_intent = predict_intent_rule(q)

        if retrieval_mode == "embeddings":
            top_mods = rank_modules_embeddings(modules, q, base_url=base_url, embed_model=embed_model, cache_buster=cache_buster, enabled=embed_enabled, k=k)
            top_tls = rank_tools_embeddings(tools, q, base_url=base_url, embed_model=embed_model, cache_buster=cache_buster, enabled=embed_enabled, k=k)
        else:
            top_mods = rank_modules(modules, q, k=k)
            top_tls = rank_tools(tools, q, k=k)

        has_catalogue_signal = bool(top_mods or top_tls)
        kw_hit = platform_keyword_hit(q)
        predicted_refusal = (not is_greeting(q)) and (not wants_subscription(q)) and ((not kw_hit) and (not has_catalogue_signal))

        if use_ollama_for_latency and (not predicted_refusal) and (not is_greeting(q)):
            selected_modules = (top_mods if top_mods else modules[:8])[:12]
            selected_tools = (top_tls if top_tls else tools[:8])[:12]
            module_lines = "\n".join([format_module_line(m) for m in selected_modules])
            tool_lines = "\n".join([format_tool_line(t, plan=plan) for t in selected_tools])
            pricing_lines = build_pricing_context(q, pricing)
            system_prompt = build_system_prompt(plan=plan, module_lines=module_lines, tool_lines=tool_lines, pricing_lines=pricing_lines)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
            try:
                _ = ollama_chat(base_url=base_url, model=chat_model, messages=messages)
            except Exception:
                pass

        latency_ms = float((time.perf_counter() - t0) * 1000.0)
        latencies_ms.append(latency_ms)

        true_in_scope = label_intent != "off_topic"
        pred_in_scope = pred_intent != "off_topic"
        y_true_in_scope.append(true_in_scope)
        y_pred_in_scope.append(pred_in_scope)
        y_true_refusal.append(expected_refusal)
        y_pred_refusal.append(predicted_refusal)

        top_k_titles: List[str] = []
        hit = None
        recall = None
        if label_intent == "catalogue":
            top_k_titles = [m.title for m in top_mods[:k]]
        elif label_intent == "tools":
            top_k_titles = [t.title for t in top_tls[:k]]

        if expected_items and top_k_titles:
            exp = {x.casefold() for x in expected_items if x}
            got = {x.casefold() for x in top_k_titles if x}
            hit = bool(exp & got)
            recall = (len(exp & got) / len(exp)) if exp else 0.0

        per_rows.append(
            {
                "question": q,
                "label_intent": label_intent,
                "pred_intent": pred_intent,
                "expected_refusal": expected_refusal,
                "pred_refusal": predicted_refusal,
                "latency_ms": latency_ms,
                "expected_items": " | ".join(expected_items),
                "top_k": " | ".join(top_k_titles),
                "hit": hit,
                "recall": recall,
                "retrieval_mode": retrieval_mode,
            }
        )

    acc_scope = sum(1 for t, p in zip(y_true_in_scope, y_pred_in_scope) if t == p) / len(y_true_in_scope) if y_true_in_scope else 0.0
    refusal_metrics = compute_binary_metrics(y_true_refusal, y_pred_refusal)
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    hit_at_k, recall_at_k, reco_details = evaluate_retrieval(
        eval_rows=eval_rows,
        modules=modules,
        tools=tools,
        retrieval_mode=retrieval_mode,
        base_url=base_url,
        embed_model=embed_model,
        embed_enabled=embed_enabled,
        cache_buster=cache_buster,
        k=k,
    )

    return {
        "metrics": {
            "scope_accuracy": float(acc_scope),
            "refusal": refusal_metrics,
            "hit_at_k": float(hit_at_k),
            "recall_at_k": float(recall_at_k),
            "avg_latency_ms": float(avg_latency),
            "n_eval": int(len(eval_rows)),
            "k": int(k),
            "retrieval_mode": retrieval_mode,
            "use_ollama_for_latency": bool(use_ollama_for_latency),
        },
        "per_question": pd.DataFrame(per_rows),
        "reco_details": reco_details,
    }


st.set_page_config(page_title="Projet — Démonstrateur Chatbot Creaexpertech", layout="wide")
ensure_state()

base_dir = Path(__file__).resolve().parent
default_modules_path = str(base_dir / "modules_rows.csv")
default_tools_path = str(base_dir / "tools_rows.csv")
default_pricing_path = str((base_dir.parent / "pricing exemple.html").resolve())

st.title("Démonstrateur Chatbot Creaexpertech")
st.caption("Application Streamlit — modèle local via Ollama — données de démonstration (CSV + page tarifs HTML)")

default_eval_path = str(base_dir / "eval_set.json")

with st.sidebar:
    st.subheader("Navigation")
    page = st.radio("Page", ["Chat", "Présentation", "Data audit", "Évaluation", "Résultats"], index=0)

    st.subheader("Ollama")
    base_url = st.text_input("URL Ollama", value="http://localhost:11434")
    available_models = ollama_get_models(base_url)
    chat_model = st.selectbox("Modèle chat", options=available_models, index=0) if available_models else st.text_input("Modèle chat", value="llama3:latest")
    embed_model = st.text_input("Modèle embeddings", value="nomic-embed-text")
    embed_available = bool(available_models) and (embed_model in available_models)
    if embed_model and (not embed_available):
        st.caption("Embeddings: modèle non installé (la comparaison embeddings sera ignorée)")
        st.caption(f"Installer: ollama pull {embed_model}")
    retrieval_mode = st.selectbox("Récupération", options=["token-overlap", "embeddings"], index=0)
    retrieval_mode_effective = "token-overlap" if (retrieval_mode == "embeddings" and not embed_available) else retrieval_mode

    st.subheader("Plan (démo)")
    plan = st.radio("Plan utilisateur", ["discovery", "pro"], index=0, horizontal=True)

    st.subheader("Fichiers")
    modules_path = st.text_input("CSV formations", value=default_modules_path)
    tools_path = st.text_input("CSV outils", value=default_tools_path)
    pricing_path = st.text_input("HTML abonnements", value=default_pricing_path)
    eval_path = st.text_input("Eval set (JSON)", value=default_eval_path)

    if st.button("Recharger données"):
        st.session_state.reload_token += 1

    cache_buster = (
        f"{st.session_state.reload_token}:"
        f"{os.path.getmtime(modules_path) if os.path.exists(modules_path) else 0}:"
        f"{os.path.getmtime(tools_path) if os.path.exists(tools_path) else 0}:"
        f"{os.path.getmtime(pricing_path) if os.path.exists(pricing_path) else 0}:"
        f"{os.path.getmtime(eval_path) if os.path.exists(eval_path) else 0}"
    )

    try:
        catalog = load_catalog(modules_path, tools_path, cache_buster)
        st.caption(f"Formations: {len(catalog['modules'])} | Outils: {len(catalog['tools'])}")
    except Exception as e:
        st.error(str(e))
        st.stop()

    pricing = load_pricing(pricing_path, cache_buster)
    st.caption(f"Tarifs: {len(pricing.get('plans') or [])} offres détectées")


modules: List[ModuleRow] = catalog["modules"]
tools: List[ToolRow] = catalog["tools"]


if page == "Présentation":
    st.subheader("Présentation")
    st.markdown(
        """
**Objectif**
- Proposer un assistant strict pour une plateforme de formation (Creaexpertech), utilisable en démonstration.

**Contraintes**
- Hors-sujet → refus exact (phrase imposée).
- Recommandations : 2–4 formations en priorité, puis 1–2 outils.
- Catalogue “à jour” : fichiers CSV rechargés à la demande.
- Abonnements : informations issues de la page HTML des tarifs.

**Scénarios**
- Accueil : “Bonjour/Salut” + boutons Catalogue / Outils / Tarifs.
- Navigation : “où est le catalogue / tarifs / outils ?” → indication de l’onglet.
- Recommandation : sélection d’éléments pertinents avant réponse (token-overlap ou embeddings).
- Hors-sujet : refus exact.
"""
    )


if page == "Data audit":
    st.subheader("Data audit (exploration + nettoyage)")

    m_raw: pd.DataFrame = catalog["modules_raw"]
    t_raw: pd.DataFrame = catalog["tools_raw"]
    m_clean: pd.DataFrame = catalog["modules_clean"]
    t_clean: pd.DataFrame = catalog["tools_clean"]
    summary = catalog["cleaning_summary"]

    st.markdown("### Statistiques (avant)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Formations")
        audit = basic_data_audit(m_raw, key_cols=["title", "description"])
        st.write(f"- Lignes: {audit['rows']}")
        st.write(f"- Doublons (title): {audit['dup_title']}")
        missing = pd.DataFrame([audit["missing_pct"]]).T.reset_index()
        missing.columns = ["colonne", "% manquants"]
        st.dataframe(missing.sort_values("% manquants", ascending=False).head(12), use_container_width=True, height=320)
    with col2:
        st.write("Outils")
        audit = basic_data_audit(t_raw, key_cols=["title", "short_description", "full_description"])
        st.write(f"- Lignes: {audit['rows']}")
        st.write(f"- Doublons (title): {audit['dup_title']}")
        missing = pd.DataFrame([audit["missing_pct"]]).T.reset_index()
        missing.columns = ["colonne", "% manquants"]
        st.dataframe(missing.sort_values("% manquants", ascending=False).head(12), use_container_width=True, height=320)

    st.markdown("### Statistiques (après)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Formations")
        audit = basic_data_audit(m_clean, key_cols=["title", "description"])
        st.write(f"- Lignes: {audit['rows']}")
        st.write(f"- Doublons (title): {audit['dup_title']}")
        missing = pd.DataFrame([audit["missing_pct"]]).T.reset_index()
        missing.columns = ["colonne", "% manquants"]
        st.dataframe(missing.sort_values("% manquants", ascending=False).head(12), use_container_width=True, height=320)
    with col2:
        st.write("Outils")
        audit = basic_data_audit(t_clean, key_cols=["title", "promise"])
        st.write(f"- Lignes: {audit['rows']}")
        st.write(f"- Doublons (title): {audit['dup_title']}")
        missing = pd.DataFrame([audit["missing_pct"]]).T.reset_index()
        missing.columns = ["colonne", "% manquants"]
        st.dataframe(missing.sort_values("% manquants", ascending=False).head(12), use_container_width=True, height=320)

    st.markdown("### Transformations appliquées")
    st.write("Formations")
    st.write(summary["modules"]["transforms"])
    st.write("Outils")
    st.write(summary["tools"]["transforms"])

    st.markdown("### Avant / Après (10 lignes)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Formations — avant")
        cols = [c for c in ["title", "description", "difficulty", "estimated_duration", "tags", "job", "tool"] if c in m_raw.columns]
        st.dataframe(m_raw[cols].head(10), use_container_width=True)
        st.write("Formations — après")
        st.dataframe(m_clean.head(10), use_container_width=True)
    with col2:
        st.write("Outils — avant")
        cols = [c for c in ["title", "promise", "short_description", "type", "metier", "niveau", "plan_required", "tags"] if c in t_raw.columns]
        st.dataframe(t_raw[cols].head(10), use_container_width=True)
        st.write("Outils — après")
        st.dataframe(t_clean.head(10), use_container_width=True)

    st.markdown("### Distributions")
    col1, col2 = st.columns(2)
    with col1:
        st.write("difficulty (modules)")
        st.dataframe(distribution_table(m_clean, "difficulty"), use_container_width=True, height=260)
        st.write("job (modules)")
        st.dataframe(distribution_table(m_clean, "job"), use_container_width=True, height=260)
    with col2:
        st.write("niveau (tools)")
        st.dataframe(distribution_table(t_clean, "niveau"), use_container_width=True, height=260)
        st.write("metier (tools)")
        st.dataframe(distribution_table(t_clean, "metier"), use_container_width=True, height=260)
        st.write("plan_required (tools)")
        st.dataframe(distribution_table(t_clean, "plan_required"), use_container_width=True, height=260)


if page == "Évaluation":
    st.subheader("Évaluation avec métriques")
    eval_rows = load_eval_set(eval_path, cache_buster)
    st.caption(f"Jeu d’évaluation: {len(eval_rows)} questions")
    st.write("Mesures: scope accuracy, refus (precision/recall/F1), Hit@K / Recall@K, latence moyenne.")

    k = st.selectbox("K (top-k)", options=[3, 5], index=0)
    use_ollama_latency = st.checkbox("Mesurer la latence avec appels Ollama (plus lent)", value=False)

    run = st.button("Lancer l’évaluation")
    if run:
        with st.spinner("Évaluation token-overlap..."):
            res_token = run_evaluation(
                eval_rows=eval_rows,
                modules=modules,
                tools=tools,
                pricing=pricing,
                plan=plan,
                base_url=base_url,
                chat_model=chat_model,
                embed_model=embed_model,
                embed_enabled=embed_available,
                cache_buster=cache_buster,
                retrieval_mode="token-overlap",
                k=int(k),
                use_ollama_for_latency=use_ollama_latency,
            )
        res_emb = None
        if embed_available:
            with st.spinner("Évaluation embeddings..."):
                res_emb = run_evaluation(
                    eval_rows=eval_rows,
                    modules=modules,
                    tools=tools,
                    pricing=pricing,
                    plan=plan,
                    base_url=base_url,
                    chat_model=chat_model,
                    embed_model=embed_model,
                    embed_enabled=embed_available,
                    cache_buster=cache_buster,
                    retrieval_mode="embeddings",
                    k=int(k),
                    use_ollama_for_latency=use_ollama_latency,
                )
        st.session_state.eval_results = {"token": res_token, "embeddings": res_emb}

    if "eval_results" in st.session_state:
        res_token = st.session_state.eval_results["token"]
        res_emb = st.session_state.eval_results.get("embeddings")

        st.markdown("### Métriques")
        rows = [
            {
                "mode": "token-overlap",
                "scope_accuracy": res_token["metrics"]["scope_accuracy"],
                "refusal_f1": res_token["metrics"]["refusal"]["f1"],
                "hit@k": res_token["metrics"]["hit_at_k"],
                "recall@k": res_token["metrics"]["recall_at_k"],
                "avg_latency_ms": res_token["metrics"]["avg_latency_ms"],
            }
        ]
        if isinstance(res_emb, dict):
            rows.append(
                {
                    "mode": "embeddings",
                    "scope_accuracy": res_emb["metrics"]["scope_accuracy"],
                    "refusal_f1": res_emb["metrics"]["refusal"]["f1"],
                    "hit@k": res_emb["metrics"]["hit_at_k"],
                    "recall@k": res_emb["metrics"]["recall_at_k"],
                    "avg_latency_ms": res_emb["metrics"]["avg_latency_ms"],
                }
            )
        df_metrics = pd.DataFrame(rows)
        st.dataframe(df_metrics, use_container_width=True)

        st.markdown("### Détails (export CSV)")
        if isinstance(res_emb, dict):
            df_export = pd.concat([res_token["per_question"], res_emb["per_question"]], ignore_index=True)
        else:
            df_export = res_token["per_question"].copy()
        st.dataframe(df_export.head(30), use_container_width=True, height=420)
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger CSV", data=csv_bytes, file_name="evaluation_export.csv", mime="text/csv")


if page == "Résultats":
    st.subheader("Résultats (présentés et commentés)")
    if "eval_results" not in st.session_state:
        eval_rows = load_eval_set(eval_path, cache_buster)
        with st.spinner("Calcul des résultats..."):
            token = run_evaluation(
                eval_rows=eval_rows,
                modules=modules,
                tools=tools,
                pricing=pricing,
                plan=plan,
                base_url=base_url,
                chat_model=chat_model,
                embed_model=embed_model,
                embed_enabled=embed_available,
                cache_buster=cache_buster,
                retrieval_mode="token-overlap",
                k=3,
                use_ollama_for_latency=False,
            )
            emb = None
            if embed_available:
                emb = run_evaluation(
                    eval_rows=eval_rows,
                    modules=modules,
                    tools=tools,
                    pricing=pricing,
                    plan=plan,
                    base_url=base_url,
                    chat_model=chat_model,
                    embed_model=embed_model,
                    embed_enabled=embed_available,
                    cache_buster=cache_buster,
                    retrieval_mode="embeddings",
                    k=3,
                    use_ollama_for_latency=False,
                )
        st.session_state.eval_results = {"token": token, "embeddings": emb}

    res_token = st.session_state.eval_results["token"]["metrics"]
    res_emb_all = st.session_state.eval_results.get("embeddings")
    res_emb = res_emb_all["metrics"] if isinstance(res_emb_all, dict) else None

    rows = [{"mode": "token-overlap", "refusal_f1": res_token["refusal"]["f1"], "recall@k": res_token["recall_at_k"]}]
    if isinstance(res_emb, dict):
        rows.append({"mode": "embeddings", "refusal_f1": res_emb["refusal"]["f1"], "recall@k": res_emb["recall_at_k"]})
    df_plot = pd.DataFrame(rows).set_index("mode")

    st.markdown("### Graphs")
    st.bar_chart(df_plot["refusal_f1"])
    st.bar_chart(df_plot["recall@k"])

    st.markdown("### Commentaire")
    better_recall = "embeddings" if (isinstance(res_emb, dict) and res_emb["recall_at_k"] > res_token["recall_at_k"]) else "token-overlap"
    st.write(
        "\n".join(
            [
                f"- Scope accuracy (token): {res_token['scope_accuracy']:.2f}" + (f" | (embeddings): {res_emb['scope_accuracy']:.2f}" if isinstance(res_emb, dict) else ""),
                f"- Refusal F1 (token): {res_token['refusal']['f1']:.2f}" + (f" | (embeddings): {res_emb['refusal']['f1']:.2f}" if isinstance(res_emb, dict) else ""),
                f"- Recall@K (token): {res_token['recall_at_k']:.2f}" + (f" | (embeddings): {res_emb['recall_at_k']:.2f}" if isinstance(res_emb, dict) else ""),
                f"- Meilleure récupération sur ce set: {better_recall}" if isinstance(res_emb, dict) else "- Embeddings non évalués (modèle embeddings non disponible dans Ollama).",
                "- Limites typiques: dépendance au vocabulaire (token-overlap), qualité du modèle embeddings (si installé), ambiguïtés intent (catalogue vs outils vs tarifs).",
            ]
        )
    )


if page == "Chat":
    tab_chat, tab_catalogue, tab_tools, tab_tarifs = st.tabs(["Chat", "Catalogue", "Outils", "Tarifs"])

    with tab_catalogue:
        st.subheader("Formations")
        df = pd.DataFrame(
            [
                {
                    "title": m.title,
                    "description": m.description,
                    "difficulty": m.difficulty,
                    "duration_minutes": m.duration_minutes,
                    "tags": ", ".join(m.tags),
                    "job": m.job,
                    "tool": m.tool,
                }
                for m in modules
            ]
        )
        st.dataframe(df, use_container_width=True, height=520)

    with tab_tools:
        st.subheader("Boîte à outils")
        df = pd.DataFrame(
            [
                {
                    "title": t.title,
                    "promise": t.promise,
                    "type": t.type,
                    "metier": t.metier,
                    "niveau": t.niveau,
                    "plan_required": t.plan_required,
                    "tags": ", ".join(t.tags),
                }
                for t in tools
            ]
        )
        st.dataframe(df, use_container_width=True, height=520)

    with tab_tarifs:
        st.subheader("Tarifs / Abonnements (démo)")
        p = Path(pricing_path)
        if not p.exists():
            st.warning("Aucune info tarifs détectée. Vérifie le chemin du fichier HTML dans la sidebar.")
        else:
            html = p.read_text(encoding="utf-8", errors="ignore")
            components.html(html, height=1600, scrolling=True)

    with tab_chat:
        st.subheader("Chat")
        st.caption(f"Récupération: {retrieval_mode_effective}")

        if not st.session_state.messages:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Bonjour ! Je suis l’assistant Creaexpertech.\n"
                        "Je réponds uniquement sur la plateforme : formations/modules, boîte à outils, tarifs/abonnements, accès et fonctionnement.\n"
                        "Clique sur un bouton ou pose ta question."
                    ),
                }
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Catalogue"):
                st.session_state.messages.append({"role": "user", "content": "Montre-moi le catalogue des formations"})
        with c2:
            if st.button("Boîte à outils"):
                st.session_state.messages.append({"role": "user", "content": "Quels outils sont disponibles ?"})
        with c3:
            if st.button("Tarifs"):
                st.session_state.messages.append({"role": "user", "content": "Quels sont les tarifs et abonnements ?"})

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        user = st.chat_input("Votre message")
        if user:
            st.session_state.messages.append({"role": "user", "content": user})

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            user_message = st.session_state.messages[-1]["content"]

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.write("...")

                if is_greeting(user_message):
                    answer = (
                        "Bonjour ! Je suis l’assistant Creaexpertech.\n"
                        "Je réponds uniquement sur la plateforme : formations/modules, boîte à outils, tarifs/abonnements, accès et fonctionnement.\n"
                        "Clique sur un bouton ou pose ta question."
                    )
                    placeholder.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.stop()

                if retrieval_mode_effective == "embeddings":
                    top_module_matches = rank_modules_embeddings(
                        modules, user_message, base_url=base_url, embed_model=embed_model, cache_buster=cache_buster, enabled=embed_available, k=10
                    )
                    top_tool_matches = rank_tools_embeddings(
                        tools, user_message, base_url=base_url, embed_model=embed_model, cache_buster=cache_buster, enabled=embed_available, k=10
                    )
                else:
                    top_module_matches = rank_modules(modules, user_message, k=10)
                    top_tool_matches = rank_tools(tools, user_message, k=10)

                has_catalogue_signal = bool(top_module_matches or top_tool_matches)
                kw_hit = platform_keyword_hit(user_message)

                wants_sub_kw = wants_subscription(user_message)
                wants_tools_kw = wants_tools(user_message)
                wants_catalogue_kw = wants_catalogue(user_message)

                links = []
                if wants_catalogue_kw:
                    links.append("Catalogue")
                if wants_tools_kw:
                    links.append("Boîte à outils")
                if wants_sub_kw:
                    links.append("Tarifs")

                if is_navigation_question(user_message) and (wants_sub_kw or wants_tools_kw or wants_catalogue_kw):
                    lines: List[str] = []
                    if wants_sub_kw:
                        lines.append("Pour accéder à la page des abonnements, clique sur “Tarifs” dans le menu (onglet Tarifs).")
                    if wants_catalogue_kw:
                        lines.append("Pour accéder au catalogue, clique sur “Catalogue” (onglet Catalogue).")
                    if wants_tools_kw:
                        lines.append("Pour accéder à la boîte à outils, clique sur “Outils” (onglet Outils).")
                    answer = "\n".join(lines).strip()
                    placeholder.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.stop()

                if (not kw_hit) and (not has_catalogue_signal) and (not wants_sub_kw):
                    placeholder.write(REFUSAL_PHRASE)
                    st.session_state.messages.append({"role": "assistant", "content": REFUSAL_PHRASE})
                    st.stop()

                selected_modules = (top_module_matches if top_module_matches else modules[:8])[:12]
                selected_tools = (top_tool_matches if top_tool_matches else tools[:8])[:12]

                module_lines = "\n".join([format_module_line(m) for m in selected_modules])
                tool_lines = "\n".join([format_tool_line(t, plan=plan) for t in selected_tools])
                pricing_lines = build_pricing_context(user_message, pricing)
                system_prompt = build_system_prompt(
                    plan=plan,
                    module_lines=module_lines,
                    tool_lines=tool_lines,
                    pricing_lines=pricing_lines,
                )

                history = []
                for m in st.session_state.messages[:-1]:
                    role = str(m.get("role", "")).strip()
                    content = str(m.get("content", "")).strip()
                    if role in ("user", "assistant") and content:
                        history.append({"role": role, "content": content})
                history = history[-12:]

                messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_message}]

                try:
                    answer = ollama_chat(base_url=base_url, model=chat_model, messages=messages)
                    normalized = str(answer or "").strip()
                    if wants_sub_kw:
                        if not normalized or normalized == REFUSAL_PHRASE:
                            answer = answer_subscription_question(user_message, pricing)
                    else:
                        if not normalized or normalized == REFUSAL_PHRASE:
                            answer = recommendation_answer(plan=plan, selected_modules=selected_modules, selected_tools=selected_tools)
                    if links:
                        answer = f"{answer}\n\nSections suggérées: {', '.join(links)}"
                    placeholder.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception:
                    answer = (
                        answer_subscription_question(user_message, pricing)
                        if wants_sub_kw
                        else recommendation_answer(plan=plan, selected_modules=selected_modules, selected_tools=selected_tools)
                    )
                    if links:
                        answer = f"{answer}\n\nSections suggérées: {', '.join(links)}"
                    placeholder.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

