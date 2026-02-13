from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import unicodedata
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

from collectors.instagram_collector import (
    collect_from_seeds,
    collect_profiles_by_niche_country,
    enrich_instagram_lead,
)
from collectors.maps_collector import collect_maps_leads_batch
from config import (
    COLUMNS,
    CSV_PATH,
    CITY,
    SEARCH_TERMS,
    MAX_RESULTS_PER_TERM,
    REQUEST_SLEEP_SECONDS,
)
from pipeline.prospect import OUTPUT_COLUMNS, generate_prospect_from_df, generate_prospect_list
from storage.sheet import append_rows, dedupe_key, ensure_csv, load_existing, update_rows_by_key

BASE_DIR = Path(__file__).resolve().parent
LEADS_PATH = (BASE_DIR / CSV_PATH) if not Path(CSV_PATH).is_absolute() else Path(CSV_PATH)
PROSPECT_PATH = BASE_DIR / "prospeccao.csv"
PROSPECT_XLSX_PATH = BASE_DIR / "prospeccao.xlsx"
IG_LOCATOR_PROSPECT_PATH = BASE_DIR / "prospeccao_instagram_locator.csv"
IG_LOCATOR_PROSPECT_XLSX_PATH = BASE_DIR / "prospeccao_instagram_locator.xlsx"
DEFAULT_SEEDS_PATH = BASE_DIR / "seeds_ig.txt"
LAST_COLLECTION_PATH = BASE_DIR / "last_collection.txt"
EXCLUDED_PATH = BASE_DIR / "excluidos.csv"
CONTACT_SETTINGS_PATH = BASE_DIR / "contact_settings.json"
IG_LOCATOR_TAG = "IG Locator"

LEADS_EXTRA_COLUMNS = ["last_contact_at", "last_contact_note"]
LEADS_REQUIRED_COLUMNS = COLUMNS + LEADS_EXTRA_COLUMNS
EXCLUDED_COLUMNS = LEADS_REQUIRED_COLUMNS + ["excluded_at", "excluded_reason"]

DEFAULT_MSG_1_BR = (
    "Ola, tudo bem?\n"
    "Meu nome e Pedro.\n\n"
    "Vi seu trabalho no Instagram e gostei muito. Notei que muitas clientes acabam chamando no direct/WhatsApp e, as vezes, falta um link mais profissional para passar confianca e facilitar o agendamento.\n\n"
    "Eu crio uma landing simples (uma pagina unica na internet, tipo um cartao de visitas profissional, com botao direto para o WhatsApp), com seus servicos, localizacao e depoimentos, para transformar visita em agendamento.\n"
    "Posso te mandar um exemplo rapidinho para voce ver como fica?"
)
DEFAULT_MSG_1_PT = (
    "Ola, tudo bem? O meu nome e Pedro.\n\n"
    "Estive a ver o seu trabalho no Instagram e gostei bastante. Reparei que muitas clientes acabam por enviar mensagem pelo direct ou WhatsApp, mas nem sempre existe um link profissional que transmita confianca e facilite a marcacao.\n\n"
    "Eu crio uma pagina online simples e profissional (como um cartao de visita digital), com botao direto para o WhatsApp, servicos, localizacao e testemunhos, tudo organizado para ajudar a transformar visitas em marcacoes.\n\n"
    "Posso enviar-lhe um exemplo rapido para ver como funciona?"
)
DEFAULT_MSG_2 = (
    "Oi, tudo bem? Meu nome e Pedro.\n"
    "Dei uma olhada no seu Instagram e achei seu trabalho bem profissional.\n\n"
    "Notei que muitas clientes acabam chamando pelo direct ou WhatsApp e, as vezes, ter um link mais organizado ajuda bastante a passar confianca e facilitar o agendamento.\n\n"
    "Eu ajudo justamente com isso: crio uma pagina simples, personalizada para o seu servico, com botao direto para o WhatsApp e as informacoes principais, para deixar tudo mais facil para quem entra no seu perfil.\n\n"
    "Se fizer sentido para voce, posso te mandar um exemplo para ver se gosta da ideia."
)

st.set_page_config(page_title="Painel de Leads", layout="wide")


def apply_responsive_styles() -> None:
    st.markdown(
        """
        <style>
        .stButton > button,
        .stLinkButton > a {
            min-height: 2rem;
            padding: 0.2rem 0.55rem;
            font-size: 0.9rem;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-direction: column;
                gap: 0.5rem;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }
            div[data-testid="stTabs"] button[role="tab"] {
                padding-left: 0.65rem;
                padding-right: 0.65rem;
                white-space: normal;
                line-height: 1.15;
            }
            .stButton > button,
            .stDownloadButton > button {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_google_places_api_key() -> str:
    env_key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        secret_key = str(st.secrets.get("GOOGLE_PLACES_API_KEY", "")).strip()
    except Exception:
        secret_key = ""
    return secret_key


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _is_no(value: str) -> bool:
    return _normalize(value) == "nao"


def ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = ""
    return out


def load_leads() -> pd.DataFrame:
    ensure_csv(LEADS_PATH, LEADS_REQUIRED_COLUMNS)
    df = pd.read_csv(LEADS_PATH, dtype=str, keep_default_na=False)
    df = ensure_columns(df, LEADS_REQUIRED_COLUMNS)
    df = fill_missing_nicho(df)
    df, harmonized = harmonize_city_country_from_phone(df)
    df = fill_missing_pais(df)
    if list(df.columns) != LEADS_REQUIRED_COLUMNS:
        df = df[LEADS_REQUIRED_COLUMNS]
        df.to_csv(LEADS_PATH, index=False, encoding="utf-8-sig")
    elif (
        harmonized > 0
        or (
        ("nicho" in df.columns and df["nicho"].astype(str).str.strip().any())
        or ("pais" in df.columns and df["pais"].astype(str).str.strip().any())
        )
    ):
        df.to_csv(LEADS_PATH, index=False, encoding="utf-8-sig")
    return df


def save_leads(df: pd.DataFrame) -> None:
    df = ensure_columns(df, LEADS_REQUIRED_COLUMNS)
    df = df[LEADS_REQUIRED_COLUMNS]
    df.to_csv(LEADS_PATH, index=False, encoding="utf-8-sig")


def load_csv_optional(path: Path, required_columns: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = ensure_columns(df, required_columns)
    df = fill_missing_nicho(df)
    return fill_missing_pais(df)


def to_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def latest_prospect_mask(df: pd.DataFrame) -> pd.Series:
    if "data_coleta" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    parsed = pd.to_datetime(df["data_coleta"], errors="coerce", utc=True, format="mixed")
    valid = parsed.notna()
    if not valid.any():
        return pd.Series([False] * len(df), index=df.index)

    latest_dt = parsed[valid].max()
    window_start = latest_dt - pd.Timedelta(minutes=20)
    return parsed >= window_start


def append_dedup_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    ensure_csv(LEADS_PATH, LEADS_REQUIRED_COLUMNS)
    existing_keys = load_existing(LEADS_PATH)
    excluded_keys = load_excluded_keys()
    new_rows = []
    for row in rows:
        key = dedupe_key(row)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_rows.append(row)
    append_rows(LEADS_PATH, new_rows)
    return len(new_rows)


def append_note(base: str, extra: str) -> str:
    base = str(base).strip()
    extra = str(extra).strip()
    if not extra:
        return base
    if not base:
        return extra
    return f"{base} | {extra}"


def parse_seed_lines(text: str) -> list[str]:
    seeds = []
    for line in str(text).splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        seeds.append(raw)
    return seeds


def load_seeds(uploaded_file, textarea_value: str) -> list[str]:
    seeds = []

    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            seeds.extend(parse_seed_lines(content))
        except Exception:
            pass

    seeds.extend(parse_seed_lines(textarea_value))

    if not seeds and DEFAULT_SEEDS_PATH.exists():
        seeds.extend(parse_seed_lines(DEFAULT_SEEDS_PATH.read_text(encoding="utf-8", errors="ignore")))

    seen = set()
    unique = []
    for item in seeds:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(item.strip())
    return unique


def write_last_collection(dt_iso: str) -> None:
    LAST_COLLECTION_PATH.write_text(dt_iso, encoding="utf-8")


def read_last_collection() -> str:
    if not LAST_COLLECTION_PATH.exists():
        return ""
    return LAST_COLLECTION_PATH.read_text(encoding="utf-8", errors="ignore").strip()


def infer_nicho_from_row(row: pd.Series) -> str:
    current = str(row.get("nicho", "")).strip()
    if current:
        return current

    observacao = str(row.get("observacao", "")).strip()
    match = re.search(r"encontrado\s+por:\s*(.+)$", observacao, flags=re.IGNORECASE)
    if match:
        term = match.group(1).strip(" .|")
        if term:
            return term

    fonte = _normalize(row.get("fonte", ""))
    if fonte == "instagram":
        if "seed ig" in _normalize(observacao):
            return "Instagram (seed)"
        return "Instagram"
    if fonte in {"maps", "google_maps"}:
        return "Maps"
    return ""


def fill_missing_nicho(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "nicho" not in out.columns:
        out["nicho"] = ""

    missing = out["nicho"].astype(str).str.strip() == ""
    if missing.any():
        out.loc[missing, "nicho"] = out.loc[missing].apply(infer_nicho_from_row, axis=1)
    return out


def harmonize_city_country_from_phone(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    if out.empty:
        return out, 0

    cidade_norm = out.get("cidade", pd.Series([""] * len(out), index=out.index)).map(_normalize)
    phone_blob = (
        out.get("telefone", pd.Series([""] * len(out), index=out.index)).astype(str)
        + " "
        + out.get("whatsapp_link", pd.Series([""] * len(out), index=out.index)).astype(str)
    )
    digits = phone_blob.str.replace(r"\D+", "", regex=True)

    is_city_portugal = cidade_norm.str.contains("portugal", na=False) | cidade_norm.eq("lisboa")
    is_br_phone = digits.str.startswith("55")
    is_sjc_phone = digits.str.startswith("5512")

    changed = 0
    mask_sjc = is_city_portugal & is_sjc_phone
    if mask_sjc.any():
        out.loc[mask_sjc, "cidade"] = "Sao Jose dos Campos - SP"
        out.loc[mask_sjc, "pais"] = "Brasil"
        changed += int(mask_sjc.sum())

    mask_br = is_city_portugal & is_br_phone & (~is_sjc_phone)
    if mask_br.any():
        out.loc[mask_br, "pais"] = "Brasil"
        changed += int(mask_br.sum())

    return out, changed


def infer_pais_from_row(row: pd.Series) -> str:
    cidade = _normalize(row.get("cidade", ""))
    city_guess = ""
    if "brasil" in cidade or "brazil" in cidade:
        city_guess = "Brasil"
    elif "portugal" in cidade:
        city_guess = "Portugal"
    elif any(tok in cidade for tok in ["usa", "united states", "estados unidos"]):
        city_guess = "Estados Unidos"

    uf_markers = {
        "ac", "al", "ap", "am", "ba", "ce", "df", "es", "go", "ma", "mt", "ms", "mg",
        "pa", "pb", "pr", "pe", "pi", "rj", "rn", "rs", "ro", "rr", "sc", "sp", "se", "to",
    }
    m_city = re.search(r"-\s*([a-z]{2})$", cidade)
    if m_city and m_city.group(1) in uf_markers and not city_guess:
        city_guess = "Brasil"

    # Strong local hint for your default market city, even when CSV came without UF suffix.
    if not city_guess and "sao jose dos campos" in cidade:
        city_guess = "Brasil"

    current = str(row.get("pais", "")).strip()
    if current:
        current_norm = _normalize(current)
        # If city clearly indicates Brazil, override mistaken country labels.
        if city_guess == "Brasil" and current_norm != "brasil":
            return "Brasil"
        return current

    if city_guess:
        return city_guess

    raw_numbers = f"{row.get('whatsapp_link', '')} {row.get('telefone', '')}"
    digits = "".join(ch for ch in str(raw_numbers) if ch.isdigit())
    if digits.startswith("55"):
        return "Brasil"
    if digits.startswith("351"):
        return "Portugal"

    return ""


def fill_missing_pais(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pais" not in out.columns:
        out["pais"] = ""

    missing = out["pais"].astype(str).str.strip() == ""
    if missing.any():
        out.loc[missing, "pais"] = out.loc[missing].apply(infer_pais_from_row, axis=1)
    return out


def ig_locator_mask(df: pd.DataFrame) -> pd.Series:
    fonte_norm = df.get("fonte", pd.Series([""] * len(df), index=df.index)).map(_normalize)
    obs = df.get("observacao", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
    return (fonte_norm == "instagram") & obs.str.contains(IG_LOCATOR_TAG.lower(), na=False)


def generate_ig_locator_prospect(
    leads_df: pd.DataFrame,
    niche: str = "",
    country: str = "",
    top_n: int = 50,
):
    if leads_df is None or leads_df.empty:
        empty_df = pd.DataFrame(columns=LEADS_REQUIRED_COLUMNS)
        return generate_prospect_from_df(
            empty_df,
            IG_LOCATOR_PROSPECT_PATH,
            IG_LOCATOR_PROSPECT_XLSX_PATH,
            top_n=top_n,
        )

    work = ensure_columns(leads_df.copy(), LEADS_REQUIRED_COLUMNS)
    mask = ig_locator_mask(work)

    niche_clean = str(niche).strip()
    country_clean = str(country).strip()
    if niche_clean:
        mask = mask & (work["nicho"].map(_normalize) == _normalize(niche_clean))
    if country_clean:
        mask = mask & (work["pais"].map(_normalize) == _normalize(country_clean))

    filtered = work[mask].copy()
    return generate_prospect_from_df(
        filtered,
        IG_LOCATOR_PROSPECT_PATH,
        IG_LOCATOR_PROSPECT_XLSX_PATH,
        top_n=top_n,
        min_priority=4,
    )


def _safe_read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except (PermissionError, OSError):
        return None


def load_excluded() -> pd.DataFrame:
    ensure_csv(EXCLUDED_PATH, EXCLUDED_COLUMNS)
    df = pd.read_csv(EXCLUDED_PATH, dtype=str, keep_default_na=False)
    return ensure_columns(df, EXCLUDED_COLUMNS)


def load_excluded_keys() -> set:
    df = load_excluded()
    return {dedupe_key(row.to_dict()) for _, row in df.iterrows()}


def append_excluded_rows(rows: list[dict], reason: str = "") -> int:
    if not rows:
        return 0

    ensure_csv(EXCLUDED_PATH, EXCLUDED_COLUMNS)
    excluded_df = load_excluded()
    existing_keys = {dedupe_key(row.to_dict()) for _, row in excluded_df.iterrows()}

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    to_add = []
    for row in rows:
        key = dedupe_key(row)
        if key in existing_keys:
            continue
        payload = {col: str(row.get(col, "")) for col in LEADS_REQUIRED_COLUMNS}
        payload["excluded_at"] = now_iso
        payload["excluded_reason"] = reason.strip()
        to_add.append(payload)
        existing_keys.add(key)

    if not to_add:
        return 0

    updated = pd.concat([excluded_df, pd.DataFrame(to_add)], ignore_index=True)
    updated = ensure_columns(updated, EXCLUDED_COLUMNS)[EXCLUDED_COLUMNS]
    updated.to_csv(EXCLUDED_PATH, index=False, encoding="utf-8-sig")
    return len(to_add)


def _status_badge(status: str) -> str:
    s = _normalize(status)
    if s == "enviado":
        return "🟨 ENVIADO"
    if s == "fechou":
        return "🟩 FECHOU"
    if s == "ignorar":
        return "🟥 IGNORAR"
    if s == "respondeu":
        return "🟦 RESPONDEU"
    return "⬜ NOVO"


def add_status_badge_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "status" not in out.columns:
        out["status_visual"] = ""
        return out
    out["status_visual"] = out["status"].map(_status_badge)
    return out


def exclude_selected_leads(selected_df: pd.DataFrame, reason: str) -> tuple[int, int]:
    if selected_df.empty:
        return 0, 0

    leads_df = load_leads()
    selected_keys = {dedupe_key(row.to_dict()) for _, row in selected_df.iterrows()}

    keep_rows = []
    removed_rows = []
    for _, row in leads_df.iterrows():
        row_dict = row.to_dict()
        if dedupe_key(row_dict) in selected_keys:
            removed_rows.append(row_dict)
        else:
            keep_rows.append(row_dict)

    if not removed_rows:
        return 0, 0

    save_leads(pd.DataFrame(keep_rows))
    excluded_added = append_excluded_rows(removed_rows, reason=reason)
    return len(removed_rows), excluded_added


def enrich_instagram_pending(
    limit: int,
    fallback_city: str,
    sleep_seconds: float,
    logger,
    progress_callback=None,
) -> dict:
    df = load_leads()
    pending = df[(df["instagram"].str.strip() == "") & (df["status"].str.strip().str.lower() == "novo")].head(limit)

    total_pending = len(pending)
    updates = []
    found = 0
    no_link = 0
    not_verified = 0
    errors = 0

    if total_pending == 0:
        logger("[IG Enrichment] Nenhum lead elegivel (instagram vazio + status=novo).")
        if progress_callback:
            progress_callback(0, 0, found, errors)
        return {
            "processed": 0,
            "found": 0,
            "no_link": 0,
            "not_verified": 0,
            "updated": 0,
            "errors": 0,
        }

    for idx, (_, row) in enumerate(pending.iterrows(), start=1):
        row_dict = row.to_dict()
        if not str(row_dict.get("cidade", "")).strip():
            row_dict["cidade"] = fallback_city

        lead_name = str(row_dict.get("nome", "")).strip() or "(sem nome)"
        logger(f"[IG Enrichment] {idx}/{total_pending} - buscando IG para: {lead_name}")

        try:
            result = enrich_instagram_lead(row_dict, sleep_seconds, check_bio=False)
        except Exception as exc:
            errors += 1
            logger(f"[IG Enrichment] Erro em '{lead_name}': {exc}")
            if progress_callback:
                progress_callback(idx, total_pending, found, errors)
            continue

        if not result:
            logger(f"[IG Enrichment] {idx}/{total_pending} - sem match confiavel para: {lead_name}")
            if progress_callback:
                progress_callback(idx, total_pending, found, errors)
            continue

        updated_row = result["updated_row"]
        updates.append(updated_row)
        found += 1

        if result["not_verified"]:
            not_verified += 1
            logger(f"[IG Enrichment] {idx}/{total_pending} - IG encontrado, verificacao indisponivel")
        elif _is_no(updated_row.get("tem_link_na_bio", "")):
            no_link += 1
            logger(f"[IG Enrichment] {idx}/{total_pending} - IG encontrado, SEM link na bio")
        else:
            logger(f"[IG Enrichment] {idx}/{total_pending} - IG encontrado, COM link na bio")

        if progress_callback:
            progress_callback(idx, total_pending, found, errors)

    updated = update_rows_by_key(
        LEADS_PATH,
        updates,
        target_fields={"instagram", "tem_link_na_bio", "observacao", "score"},
    )
    return {
        "processed": total_pending,
        "found": found,
        "no_link": no_link,
        "not_verified": not_verified,
        "updated": updated,
        "errors": errors,
    }


def run_collection(
    city: str,
    terms: list[str],
    max_results_per_term: int,
    run_maps: bool,
    run_ig_seeds: bool,
    run_ig_enrichment: bool,
    ig_limit: int,
    seeds: list[str],
    only_without_site: bool = False,
):
    summary = {
        "city": city,
        "maps_added": 0,
        "maps_term_counts": {},
        "ig_seed_added": 0,
        "ig_seed_input": len(seeds),
        "ig_found": 0,
        "ig_no_link": 0,
        "ig_not_verified": 0,
        "ig_updated": 0,
        "ig_errors": 0,
        "filtered_without_site": 0,
        "errors": [],
        "total_csv": 0,
    }

    selected_steps = int(run_maps) + int(run_ig_seeds) + int(run_ig_enrichment)
    progress = st.progress(0)
    step_done = 0

    with st.status("Executando coleta...", expanded=True) as status_box:
        def log(msg: str):
            status_box.write(msg)

        if selected_steps == 0:
            summary["errors"].append("Nenhuma fonte selecionada para coleta.")
        else:
            if run_maps:
                api_key = get_google_places_api_key()
                if not api_key:
                    summary["errors"].append("GOOGLE_PLACES_API_KEY nao encontrada. Maps nao executado.")
                    log("[Maps] Pulado: sem GOOGLE_PLACES_API_KEY.")
                else:
                    clean_terms = [t for t in [str(x).strip() for x in terms] if t]
                    log(f"[Maps] Iniciando em '{city}' com {len(clean_terms)} termo(s).")
                    maps_rows, per_term = collect_maps_leads_batch(
                        api_key=api_key,
                        city=city,
                        search_terms=clean_terms,
                        max_results_per_term=int(max_results_per_term),
                        sleep_seconds=REQUEST_SLEEP_SECONDS,
                        logger=log,
                    )
                    if only_without_site:
                        before_count = len(maps_rows)
                        maps_rows = [
                            r
                            for r in maps_rows
                            if (_normalize(r.get("tem_site", "")) == "nao") or (not str(r.get("site", "")).strip())
                        ]
                        filtered_out = max(before_count - len(maps_rows), 0)
                        summary["filtered_without_site"] += filtered_out
                        log(f"[Filtro sem site] Maps: {filtered_out} lead(s) removidos pelo filtro.")
                    summary["maps_term_counts"] = per_term
                    summary["maps_added"] = append_dedup_rows(maps_rows)
                    log(f"[Maps] Novos adicionados no CSV: {summary['maps_added']}")

                step_done += 1
                progress.progress(min(step_done / max(selected_steps, 1), 1.0))

            if run_ig_seeds:
                log(f"[IG Seeds] Iniciando com {len(seeds)} seed(s).")
                if not seeds:
                    summary["errors"].append("Sem seeds para Instagram (upload/textarea/arquivo padrao).")
                    log("[IG Seeds] Pulado: nenhuma seed disponivel.")
                else:
                    seed_rows = collect_from_seeds(seeds=seeds, city=city, sleep_seconds=REQUEST_SLEEP_SECONDS)
                    if only_without_site:
                        before_count = len(seed_rows)
                        seed_rows = [
                            r
                            for r in seed_rows
                            if (_normalize(r.get("tem_site", "")) == "nao") or (not str(r.get("site", "")).strip())
                        ]
                        filtered_out = max(before_count - len(seed_rows), 0)
                        summary["filtered_without_site"] += filtered_out
                        log(f"[Filtro sem site] IG Seeds: {filtered_out} lead(s) removidos pelo filtro.")
                    summary["ig_seed_added"] = append_dedup_rows(seed_rows)
                    log(f"[IG Seeds] Novos adicionados no CSV: {summary['ig_seed_added']}")

                step_done += 1
                progress.progress(min(step_done / max(selected_steps, 1), 1.0))

            if run_ig_enrichment:
                log(f"[IG Enrichment] Iniciando (limite {int(ig_limit)}) - modo rapido sem checar bio.")
                ig_stage_progress = st.progress(0)

                def _ig_progress(done: int, total: int, found_count: int, error_count: int):
                    if total <= 0:
                        try:
                            ig_stage_progress.progress(0, text="IG Enrichment: sem linhas elegiveis")
                        except TypeError:
                            ig_stage_progress.progress(0)
                        return
                    pct = min(max(done / total, 0.0), 1.0)
                    msg = f"IG Enrichment: {done}/{total} processados | encontrados={found_count} | erros={error_count}"
                    try:
                        ig_stage_progress.progress(pct, text=msg)
                    except TypeError:
                        ig_stage_progress.progress(pct)

                enrich = enrich_instagram_pending(
                    int(ig_limit),
                    city,
                    REQUEST_SLEEP_SECONDS,
                    log,
                    progress_callback=_ig_progress,
                )
                summary["ig_found"] = enrich["found"]
                summary["ig_no_link"] = enrich["no_link"]
                summary["ig_not_verified"] = enrich["not_verified"]
                summary["ig_updated"] = enrich["updated"]
                summary["ig_errors"] = enrich.get("errors", 0)
                log(
                    "[IG Enrichment] "
                    f"Encontrados={enrich['found']} | Sem link={enrich['no_link']} | "
                    f"Nao verificados={enrich['not_verified']} | Atualizados={enrich['updated']} | "
                    f"Erros={enrich.get('errors', 0)}"
                )

                step_done += 1
                progress.progress(min(step_done / max(selected_steps, 1), 1.0))

        total_rows = len(pd.read_csv(LEADS_PATH, dtype=str, keep_default_na=False)) if LEADS_PATH.exists() else 0
        summary["total_csv"] = total_rows

        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_last_collection(now_iso)
        st.session_state["last_collection"] = now_iso
        st.session_state["last_collection_summary"] = summary

        if summary["errors"]:
            status_box.update(label="Coleta concluida com alertas", state="error")
        else:
            status_box.update(label="Coleta concluida com sucesso", state="complete")

    return summary


def apply_filters(df: pd.DataFrame, prefix: str, has_priority: bool) -> pd.DataFrame:
    work = df.copy()
    if work.empty:
        return work

    work = fill_missing_nicho(work)
    work = fill_missing_pais(work)

    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1, 1, 2])

    fonte_opts = sorted([v for v in work["fonte"].dropna().astype(str).unique() if v])
    status_opts = sorted([v for v in work["status"].dropna().astype(str).unique() if v])
    nicho_opts = sorted([v for v in work["nicho"].dropna().astype(str).unique() if v])
    pais_opts = sorted([v for v in work["pais"].dropna().astype(str).unique() if v])
    latest_status_option = "Ultima prospeccao"
    status_filter_opts = [latest_status_option] + status_opts

    fonte_sel = c1.multiselect("Fonte", fonte_opts, key=f"{prefix}_fonte")
    status_sel = c2.multiselect("Status", status_filter_opts, key=f"{prefix}_status")
    nicho_sel = c3.multiselect("Nicho", nicho_opts, key=f"{prefix}_nicho")
    pais_sel = c4.multiselect("Pais", pais_opts, key=f"{prefix}_pais")
    tem_site_sel = c5.multiselect("Tem site", ["sim", "nao"], key=f"{prefix}_tem_site")
    tem_bio_sel = c6.multiselect("Link na bio", ["sim", "nao"], key=f"{prefix}_tem_bio")
    text = c7.text_input("Pesquisar (nome/instagram/bairro)", key=f"{prefix}_text")

    if fonte_sel:
        work = work[work["fonte"].isin(fonte_sel)]
    if status_sel:
        selected_real_status = [s for s in status_sel if s != latest_status_option]
        mask = pd.Series([False] * len(work), index=work.index)
        if selected_real_status:
            mask = mask | work["status"].isin(selected_real_status)
        if latest_status_option in status_sel:
            mask = mask | latest_prospect_mask(work)
        work = work[mask]
    if nicho_sel:
        work = work[work["nicho"].isin(nicho_sel)]
    if pais_sel:
        work = work[work["pais"].isin(pais_sel)]
    if tem_site_sel:
        work = work[work["tem_site"].map(_normalize).isin(tem_site_sel)]
    if tem_bio_sel:
        work = work[work["tem_link_na_bio"].map(_normalize).isin(tem_bio_sel)]
    if text.strip():
        q = text.strip().lower()
        mask = (
            work["nome"].str.lower().str.contains(q, na=False)
            | work["instagram"].str.lower().str.contains(q, na=False)
            | work["bairro"].str.lower().str.contains(q, na=False)
        )
        work = work[mask]

    c8, c9, c10 = st.columns([2, 2, 2])

    score_num = to_int_series(work["score"])
    score_range = (int(score_num.min()), int(score_num.max())) if len(score_num) else (0, 0)
    if score_range[0] == score_range[1]:
        c8.caption(f"Score fixo: {score_range[0]}")
        score_sel = score_range
    else:
        score_sel = c8.slider("Score", score_range[0], score_range[1], score_range, key=f"{prefix}_score")
    work = work[(to_int_series(work["score"]) >= score_sel[0]) & (to_int_series(work["score"]) <= score_sel[1])]

    if has_priority and "priority_score" in work.columns:
        p_num = to_int_series(work["priority_score"])
        p_range = (int(p_num.min()), int(p_num.max())) if len(p_num) else (0, 0)
        if p_range[0] == p_range[1]:
            c9.caption(f"Priority fixo: {p_range[0]}")
            p_sel = p_range
        else:
            p_sel = c9.slider("Priority Score", p_range[0], p_range[1], p_range, key=f"{prefix}_priority")
        work = work[(to_int_series(work["priority_score"]) >= p_sel[0]) & (to_int_series(work["priority_score"]) <= p_sel[1])]

    sort_cols = [c for c in ["priority_score", "score", "nome", "status", "fonte", "data_coleta"] if c in work.columns]
    sort_by = c10.selectbox("Ordenar por", sort_cols, key=f"{prefix}_sort")
    asc = st.checkbox("Ordem crescente", value=False, key=f"{prefix}_asc")
    if sort_by in {"score", "priority_score"}:
        work = work.assign(_sort=to_int_series(work[sort_by])).sort_values("_sort", ascending=asc).drop(columns=["_sort"])
    else:
        work = work.sort_values(sort_by, ascending=asc)

    return work


def update_from_editor(leads_df: pd.DataFrame, edited_df: pd.DataFrame, editable_fields: list[str]) -> int:
    if edited_df.empty:
        return 0

    key_to_row = {dedupe_key(row.to_dict()): row.to_dict() for _, row in leads_df.iterrows()}
    updates = []
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for _, row in edited_df.iterrows():
        row_dict = row.to_dict()
        key = dedupe_key(row_dict)
        current = key_to_row.get(key)
        if not current:
            continue

        changed_fields = {}
        for field in editable_fields:
            if field in row_dict and str(row_dict[field]) != str(current.get(field, "")):
                changed_fields[field] = str(row_dict[field])

        if changed_fields:
            payload = {
                "fonte": row_dict.get("fonte", ""),
                "nome": row_dict.get("nome", ""),
                "telefone": row_dict.get("telefone", ""),
                "link_origem": row_dict.get("link_origem", ""),
            }
            # If status changed in inline editor, register contact timestamp.
            if "status" in changed_fields:
                payload["last_contact_at"] = now_iso
            payload.update(changed_fields)
            updates.append(payload)

    return update_rows_by_key(LEADS_PATH, updates, target_fields=set(editable_fields) | {"last_contact_at"})


def batch_update_status(selected_df: pd.DataFrame, status_value: str, note: str) -> int:
    if selected_df.empty:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    updates = []
    for _, row in selected_df.iterrows():
        row_dict = row.to_dict()
        payload = {
            "fonte": row_dict.get("fonte", ""),
            "nome": row_dict.get("nome", ""),
            "telefone": row_dict.get("telefone", ""),
            "link_origem": row_dict.get("link_origem", ""),
            "status": status_value,
            "last_contact_at": now_iso,
            "last_contact_note": note.strip() if note.strip() else row_dict.get("last_contact_note", ""),
        }
        updates.append(payload)

    return update_rows_by_key(
        LEADS_PATH,
        updates,
        target_fields={"status", "last_contact_at", "last_contact_note"},
    )


def normalize_instagram_url(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if raw.startswith("@"):
        raw = raw[1:]
    return f"https://www.instagram.com/{raw}/"


def _only_digits(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def _phone_to_whatsapp_digits(phone: str) -> str:
    raw = str(phone or "").strip()
    digits = _only_digits(raw)
    if not digits:
        return ""

    if raw.startswith("+"):
        return digits
    if digits.startswith("00") and len(digits) > 2:
        return digits[2:]

    if 11 <= len(digits) <= 15:
        if len(digits) == 11 and digits.startswith("1") and digits[2] != "9":
            return digits
        if len(digits) >= 12:
            return digits

    # Fallback for Brazilian local numbers without country code.
    if len(digits) in {10, 11} and not digits.startswith("55"):
        return f"55{digits}"

    return digits


def _whatsapp_base_link(lead: dict) -> str:
    digits = _phone_to_whatsapp_digits(lead.get("telefone", ""))
    if digits:
        return f"https://wa.me/{digits}"

    existing = str(lead.get("whatsapp_link", "")).strip()
    if existing:
        return existing

    return ""


def _message_for_lead(template: str, lead: dict) -> str:
    text = str(template or "").strip()
    if not text:
        text = "Oi {nome}, tudo bem?"
    return (
        text.replace("{nome}", str(lead.get("nome", "")).strip())
        .replace("{cidade}", str(lead.get("cidade", "")).strip())
        .replace("{bairro}", str(lead.get("bairro", "")).strip())
    )


def _build_whatsapp_send_link(lead: dict, template: str) -> str:
    base = _whatsapp_base_link(lead)
    if not base:
        return ""

    msg = _message_for_lead(template, lead)
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}text={quote_plus(msg)}"


def _default_contact_settings() -> dict:
    return {
        "msg_1_br": DEFAULT_MSG_1_BR,
        "msg_1_pt": DEFAULT_MSG_1_PT,
        "msg_2": DEFAULT_MSG_2,
    }


def load_contact_settings() -> dict:
    defaults = _default_contact_settings()
    if not CONTACT_SETTINGS_PATH.exists():
        return defaults
    try:
        data = json.loads(CONTACT_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    if not isinstance(data, dict):
        return defaults
    out = defaults.copy()
    for key in out.keys():
        value = data.get(key, out[key])
        out[key] = str(value or "").strip() or out[key]
    return out


def save_contact_settings(settings: dict) -> None:
    payload = _default_contact_settings()
    for key in payload.keys():
        payload[key] = str(settings.get(key, payload[key]) or "").strip() or payload[key]
    CONTACT_SETTINGS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _selected_rows_from_state(grid_key: str, total_rows: int) -> list[int]:
    state = st.session_state.get(grid_key)
    if not state:
        return []

    rows = []
    cells = []
    if isinstance(state, dict):
        selection = state.get("selection", {}) or {}
        rows = selection.get("rows", []) or []
        cells = selection.get("cells", []) or []
    else:
        selection = getattr(state, "selection", None)
        if selection is not None:
            rows = getattr(selection, "rows", []) or []
            cells = getattr(selection, "cells", []) or []

    normalized_set = set()
    for item in rows:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < total_rows:
            normalized_set.add(idx)

    # Fallback: if user clicked a cell, infer the row from cell selection.
    for cell in cells:
        row_idx = None
        if isinstance(cell, dict):
            row_idx = cell.get("row")
        elif isinstance(cell, (list, tuple)) and len(cell) >= 1:
            row_idx = cell[0]
        try:
            idx = int(row_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < total_rows:
            normalized_set.add(idx)

    return sorted(normalized_set)


def selected_from_grid(df: pd.DataFrame, grid_key: str) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()
    rows = _selected_rows_from_state(grid_key, len(df))
    if not rows:
        return df.iloc[0:0].copy()
    return df.iloc[rows].copy()


def search_instagram_for_selected(selected_df: pd.DataFrame, key_prefix: str) -> None:
    if selected_df.empty:
        st.warning("Selecione uma linha para buscar Instagram.")
        return

    if len(selected_df) > 1:
        st.info("Mais de uma linha selecionada: usando apenas a primeira.")

    lead = selected_df.iloc[0].to_dict()
    lead_name = str(lead.get("nome", "")).strip() or "(sem nome)"

    with st.spinner(f"Buscando Instagram para: {lead_name}..."):
        result = enrich_instagram_lead(lead, REQUEST_SLEEP_SECONDS, check_bio=False)

    if not result:
        st.warning(f"Nao encontrei Instagram confiavel para: {lead_name}")
        return

    updated_row = result["updated_row"]
    changed = update_rows_by_key(
        LEADS_PATH,
        [updated_row],
        target_fields={"instagram", "tem_link_na_bio", "observacao", "score"},
    )

    if changed:
        generate_prospect_list(LEADS_PATH, PROSPECT_PATH, PROSPECT_XLSX_PATH)
        st.success(f"Instagram atualizado para: {lead_name}")
        st.rerun()

    st.info("Nenhuma alteracao aplicada (possivelmente ja estava preenchido).")


def render_selection_menu(tab_label: str, selected_df: pd.DataFrame) -> None:
    total_selected = len(selected_df)
    if total_selected == 0:
        return

    lead = selected_df.iloc[0].to_dict()
    wa = _whatsapp_base_link(lead)
    ig = normalize_instagram_url(lead.get("instagram", ""))
    origem = str(lead.get("link_origem", "")).strip()
    lead_country = _normalize(lead.get("pais", ""))
    settings = load_contact_settings()
    msg_1 = settings["msg_1_pt"] if lead_country == "portugal" else settings["msg_1_br"]
    msg_2 = settings["msg_2"]
    send_url_1 = _build_whatsapp_send_link(lead, msg_1)
    send_url_2 = _build_whatsapp_send_link(lead, msg_2)
    cols = st.columns([1, 1, 0.9, 0.9, 1, 1, 1, 1, 1, 1, 0.9, 1, 2], gap="small")
    if send_url_1:
        cols[0].link_button("Msg 1", send_url_1, width="stretch")
    else:
        cols[0].button("Msg 1", disabled=True, key=f"{tab_label}_send_wa1_dis", width="stretch")
    if send_url_2:
        cols[1].link_button("Msg 2", send_url_2, width="stretch")
    else:
        cols[1].button("Msg 2", disabled=True, key=f"{tab_label}_send_wa2_dis", width="stretch")
    if wa:
        cols[2].link_button("WA", wa, width="stretch")
    else:
        cols[2].button("WA", disabled=True, key=f"{tab_label}_wa_dis", width="stretch")
    if ig:
        cols[3].link_button("IG", ig, width="stretch")
    else:
        cols[3].button("IG", disabled=True, key=f"{tab_label}_ig_dis", width="stretch")
    if origem:
        cols[4].link_button("Origem", origem, width="stretch")
    else:
        cols[4].button("Origem", disabled=True, key=f"{tab_label}_orig_dis", width="stretch")
    if cols[5].button("Buscar IG", key=f"{tab_label}_search_ig", width="stretch"):
        search_instagram_for_selected(selected_df, tab_label)

    note = cols[12].text_input(
        "Nota",
        key=f"{tab_label}_note",
        label_visibility="collapsed",
        placeholder="nota",
    )

    for idx, label, value in [
        (6, "ENVIAR", "enviado"),
        (7, "IGNORAR", "ignorar"),
        (8, "FECHOU", "fechou"),
        (9, "RESPONDEU", "respondeu"),
        (10, "NOVO", "novo"),
    ]:
        if cols[idx].button(label, key=f"{tab_label}_{value}", width="stretch"):
            updated = batch_update_status(selected_df, value, note)
            if updated == 0:
                st.warning("Nenhuma linha selecionada.")
            else:
                st.success(f"{updated} lead(s) atualizados para '{value}'.")
                generate_prospect_list(LEADS_PATH, PROSPECT_PATH, PROSPECT_XLSX_PATH)
                generate_ig_locator_prospect(load_leads())
                st.rerun()

    if cols[11].button("EXCLUIR", key=f"{tab_label}_exclude", width="stretch"):
        removed, excluded_added = exclude_selected_leads(selected_df, reason=note)
        if removed == 0:
            st.warning("Nenhuma linha selecionada para exclusao.")
        else:
            st.success(
                f"{removed} lead(s) removidos da lista e {excluded_added} adicionados em excluidos.csv."
            )
            generate_prospect_list(LEADS_PATH, PROSPECT_PATH, PROSPECT_XLSX_PATH)
            generate_ig_locator_prospect(load_leads())
            st.rerun()


def render_kpis(leads_df: pd.DataFrame) -> None:
    norm_status = leads_df["status"].map(_normalize)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total leads", len(leads_df))
    c2.metric("Novos", int((norm_status == "novo").sum()))
    c3.metric("Enviados", int((norm_status == "enviado").sum()))
    c4.metric("Respondeu", int((norm_status == "respondeu").sum()))
    c5.metric("Fechou", int((norm_status == "fechou").sum()))


def render_outreach_analytics(leads_df: pd.DataFrame) -> None:
    st.subheader("Analise de Envios")
    st.caption("Contagem baseada em status='enviado' e campo last_contact_at.")

    df = ensure_columns(leads_df.copy(), LEADS_REQUIRED_COLUMNS)
    status_norm = df["status"].map(_normalize)
    sent_df = df[status_norm == "enviado"].copy()

    if sent_df.empty:
        st.info("Ainda nao ha leads com status 'enviado'.")
        return

    sent_df["_sent_at"] = pd.to_datetime(sent_df["last_contact_at"], errors="coerce", utc=True, format="mixed")
    missing_ts = int(sent_df["_sent_at"].isna().sum())
    sent_with_ts = sent_df[sent_df["_sent_at"].notna()].copy()

    if sent_with_ts.empty:
        st.warning("Ha leads enviados, mas sem data em last_contact_at para analisar por periodo.")
        return

    local_tz = datetime.now().astimezone().tzinfo
    sent_with_ts["_sent_local"] = sent_with_ts["_sent_at"].dt.tz_convert(local_tz)
    sent_with_ts["_sent_date"] = sent_with_ts["_sent_local"].dt.date
    today = datetime.now().astimezone().date()
    yesterday = today - pd.Timedelta(days=1)

    window_days = st.selectbox(
        "Janela (ultimos dias)",
        options=[7, 14, 30, 60],
        index=0,
        key="analytics_window_days",
    )
    window_start = today - pd.Timedelta(days=int(window_days) - 1)

    sent_today = int((sent_with_ts["_sent_date"] == today).sum())
    sent_yesterday = int((sent_with_ts["_sent_date"] == yesterday).sum())
    sent_window = int((sent_with_ts["_sent_date"] >= window_start).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enviados hoje", sent_today)
    c2.metric("Enviados ontem", sent_yesterday)
    c3.metric(f"Ultimos {int(window_days)} dias", sent_window)
    c4.metric("Sem data de envio", missing_ts)

    daily = (
        sent_with_ts.groupby("_sent_date")
        .size()
        .reset_index(name="enviados")
        .rename(columns={"_sent_date": "data"})
        .sort_values("data", ascending=False)
    )
    recent_daily = daily[daily["data"] >= window_start].copy()
    st.dataframe(recent_daily, use_container_width=True, hide_index=True, height=260)


def render_contact_settings() -> None:
    st.subheader("Mensagens WhatsApp")
    st.caption("Esses textos sao usados pelos botoes 'Enviar mensagem 1' e 'Enviar mensagem 2' no menu de acoes do lead.")
    settings = load_contact_settings()
    msg_1_br = st.text_area(
        "Mensagem 1 - Brasil (use {nome}, {cidade}, {bairro})",
        value=settings["msg_1_br"],
        key="cfg_msg_1_br",
        height=140,
    )
    msg_1_pt = st.text_area(
        "Mensagem 1 - Portugal (use {nome}, {cidade}, {bairro})",
        value=settings["msg_1_pt"],
        key="cfg_msg_1_pt",
        height=140,
    )
    msg_2 = st.text_area(
        "Mensagem 2 (use {nome}, {cidade}, {bairro})",
        value=settings["msg_2"],
        key="cfg_msg_2",
        height=140,
    )
    c1, c2 = st.columns([1, 1])
    if c1.button("Salvar mensagens", key="cfg_save_messages"):
        save_contact_settings({"msg_1_br": msg_1_br, "msg_1_pt": msg_1_pt, "msg_2": msg_2})
        st.success("Mensagens salvas.")
    if c2.button("Restaurar padrao", key="cfg_reset_messages"):
        save_contact_settings(_default_contact_settings())
        st.success("Mensagens padrao restauradas.")
        st.rerun()


def render_latest_prospect_tab() -> None:
    st.subheader("Ultima Prospeccao")
    st.caption("Exibe a ultima rodada de leads coletados (campanha mais recente).")

    qty = st.selectbox(
        "Quantidade para exibir",
        options=[10, 25, 50, 100],
        index=2,
        key="latest_prospect_qty",
    )

    df = ensure_columns(load_leads().copy(), LEADS_REQUIRED_COLUMNS)
    df["_data_coleta_dt"] = pd.to_datetime(df["data_coleta"], errors="coerce", utc=True, format="mixed")
    df = df[df["_data_coleta_dt"].notna()].copy()
    if df.empty:
        st.info("Sem data_coleta valida para montar a ultima prospeccao.")
        return

    latest_dt = df["_data_coleta_dt"].max()
    # Considera a ultima rodada como uma janela curta antes do ultimo registro.
    recent_window = latest_dt - pd.Timedelta(minutes=20)
    latest_batch = df[df["_data_coleta_dt"] >= recent_window].copy()

    if latest_batch.empty:
        st.info("Nao foi possivel identificar a ultima rodada.")
        return

    latest_batch["score_num"] = pd.to_numeric(latest_batch["score"], errors="coerce").fillna(0).astype(int)
    latest_batch = latest_batch.sort_values(
        by=["_data_coleta_dt", "score_num", "nome"],
        ascending=[False, False, True],
    )
    st.caption(
        f"Janela da ultima rodada: {recent_window.isoformat()} ate {latest_dt.isoformat()} "
        f"({len(latest_batch)} lead(s))."
    )

    view_cols = [
        "fonte",
        "nome",
        "nicho",
        "pais",
        "status",
        "instagram",
        "telefone",
        "whatsapp_link",
        "cidade",
        "bairro",
        "score",
        "data_coleta",
    ]
    view_cols = [c for c in view_cols if c in latest_batch.columns]
    recent = latest_batch.head(int(qty))[view_cols].copy()

    st.dataframe(recent, use_container_width=True, hide_index=True, height=420)


def render_collection_tab(leads_df: pd.DataFrame) -> None:
    st.subheader("Coleta/Busca")

    api_key_exists = bool(get_google_places_api_key())
    st.caption("Configure overrides sem alterar config.py e execute a coleta daqui.")

    with st.form("collect_form"):
        city_override = st.text_input("CITY_OVERRIDE", value=CITY)

        base_terms = [str(t) for t in SEARCH_TERMS]
        selected_terms = st.multiselect(
            "SEARCH_TERMS_OVERRIDE",
            options=base_terms,
            default=base_terms,
        )
        extra_terms_raw = st.text_input("Termos adicionais (separados por virgula)")
        max_results_override = st.number_input(
            "MAX_RESULTS_PER_TERM",
            min_value=1,
            max_value=200,
            value=int(MAX_RESULTS_PER_TERM),
            step=1,
        )

        c1, c2, c3 = st.columns(3)
        run_maps = c1.checkbox("Rodar Maps (Google Places)", value=api_key_exists)
        run_ig_seeds = c2.checkbox("Rodar Seeds Instagram", value=False)
        run_ig_enrichment = c3.checkbox("Rodar Enrichment Instagram (DDG)", value=True)
        only_without_site = st.checkbox(
            "Apenas paginas sem site",
            value=False,
            help="Quando marcado, salva somente leads com site vazio/tem_site = nao.",
        )

        ig_limit = st.number_input("Limite IG por execucao", min_value=1, max_value=500, value=30, step=1)

        st.write("Seeds Instagram (opcional)")
        uploaded = st.file_uploader("Upload seeds_ig.txt", type=["txt"])
        seeds_text = st.text_area("Ou cole seeds (uma por linha)", height=120)

        submitted = st.form_submit_button("Executar Coleta", use_container_width=True, type="primary")

    if submitted:
        extra_terms = [t.strip() for t in extra_terms_raw.split(",") if t.strip()]
        terms = []
        seen_terms = set()
        for term in selected_terms + extra_terms:
            key = term.strip().lower()
            if key and key not in seen_terms:
                seen_terms.add(key)
                terms.append(term.strip())

        seeds = load_seeds(uploaded, seeds_text)

        summary = run_collection(
            city=city_override.strip() or CITY,
            terms=terms,
            max_results_per_term=int(max_results_override),
            run_maps=bool(run_maps),
            run_ig_seeds=bool(run_ig_seeds),
            run_ig_enrichment=bool(run_ig_enrichment),
            ig_limit=int(ig_limit),
            seeds=seeds,
            only_without_site=bool(only_without_site),
        )

        if summary["errors"]:
            for err in summary["errors"]:
                st.error(err)

        st.success(
            "Resumo da coleta: "
            f"Maps adicionados={summary['maps_added']} | IG Seeds adicionados={summary['ig_seed_added']} | "
            f"IG atualizados={summary['ig_updated']} | Filtrados sem site={summary.get('filtered_without_site', 0)} | "
            f"Total CSV={summary['total_csv']}"
        )

    last = st.session_state.get("last_collection") or read_last_collection()
    if last:
        st.caption(f"Ultima coleta: {last}")

    summary = st.session_state.get("last_collection_summary")
    if summary:
        st.json(summary)

    c1, c2 = st.columns(2)
    if c1.button("Gerar Prospeccao (Top 50)", key="collect_gen_prosp"):
        result = generate_prospect_list(LEADS_PATH, PROSPECT_PATH, PROSPECT_XLSX_PATH)
        st.success(f"Prospeccao gerada: {result['selected']} leads")

    if c2.button("Recarregar dados", key="collect_reload"):
        st.rerun()


def render_instagram_locator_tab(leads_df: pd.DataFrame) -> None:
    st.subheader("Instagram Nicho/Pais")
    st.caption("Localizador de paginas publicas no Instagram por nicho e pais (sem Maps).")

    with st.form("ig_locator_form"):
        c1, c2, c3 = st.columns(3)
        niche = c1.text_input("Nicho", value="designer de sobrancelhas")
        country = c2.text_input("Pais", value="Brasil")
        city_hint = c3.text_input("Cidade (opcional)", value="")

        c4, c5, c6 = st.columns(3)
        limit = c4.number_input("Limite de perfis", min_value=5, max_value=100, value=25, step=1)
        top_n = c5.number_input("Top para prospeccao", min_value=10, max_value=200, value=50, step=5)
        check_bio = c6.checkbox("Verificar link na bio (mais lento)", value=False)

        submitted = st.form_submit_button("Buscar paginas e salvar", use_container_width=True, type="primary")

    if submitted:
        if not niche.strip() or not country.strip():
            st.error("Informe pelo menos nicho e pais.")
        else:
            with st.spinner("Buscando perfis no Instagram..."):
                rows = collect_profiles_by_niche_country(
                    niche=niche.strip(),
                    country=country.strip(),
                    sleep_seconds=REQUEST_SLEEP_SECONDS,
                    limit=int(limit),
                    city_hint=city_hint.strip(),
                    check_bio=bool(check_bio),
                )
            st.session_state["ig_locator_last_search_rows"] = rows
            st.session_state["ig_locator_last_search_meta"] = {
                "niche": niche.strip(),
                "country": country.strip(),
                "city_hint": city_hint.strip(),
                "limit": int(limit),
                "check_bio": bool(check_bio),
                "searched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            added = append_dedup_rows(rows)
            duplicates = max(len(rows) - added, 0)

            refreshed = load_leads()
            prospect_result = generate_ig_locator_prospect(
                refreshed,
                niche=niche.strip(),
                country=country.strip(),
                top_n=int(top_n),
            )
            st.session_state["ig_locator_last_filters"] = {
                "niche": niche.strip(),
                "country": country.strip(),
                "top_n": int(top_n),
            }
            st.success(
                f"IG Locator: encontrados={len(rows)} | adicionados={added} | duplicados={duplicates} | "
                f"prospeccao={prospect_result['selected']}"
            )

    last_rows = st.session_state.get("ig_locator_last_search_rows", [])
    last_meta = st.session_state.get("ig_locator_last_search_meta", {})
    if last_rows:
        st.caption(
            "Busca atual em memoria: "
            f"{len(last_rows)} perfil(is) | "
            f"nicho={last_meta.get('niche', '')} | pais={last_meta.get('country', '')} | "
            f"executada em {last_meta.get('searched_at', '')}"
        )
        c_now_1, c_now_2 = st.columns([1.4, 2.6])
        if c_now_1.button("Gerar prospeccao desta busca", key="ig_locator_generate_current_search"):
            result = generate_ig_locator_prospect(
                load_leads(),
                niche=str(last_meta.get("niche", "")).strip(),
                country=str(last_meta.get("country", "")).strip(),
                top_n=int(last_meta.get("top_n", 50)),
            )
            st.success(f"Prospecao IG gerada: {result['selected']} lead(s)")

        current_search_df = pd.DataFrame(last_rows)
        export_cols = [c for c in LEADS_REQUIRED_COLUMNS if c in current_search_df.columns]
        if export_cols:
            current_search_df = current_search_df[export_cols]
        csv_bytes = current_search_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        c_now_2.download_button(
            "Baixar busca atual (CSV)",
            csv_bytes,
            file_name="ig_locator_busca_atual.csv",
            key="ig_locator_export_current_search",
        )

    st.divider()
    st.write("Prospecao separada (IG Nicho/Pais)")

    last_filters = st.session_state.get("ig_locator_last_filters", {})
    default_niche = str(last_filters.get("niche", "")).strip()
    default_country = str(last_filters.get("country", "")).strip()
    default_top_n = int(last_filters.get("top_n", 50))

    c1, c2, c3, c4 = st.columns([2, 2, 1.2, 1.2])
    filter_niche = c1.text_input("Filtro nicho", value=default_niche, key="ig_locator_filter_niche")
    filter_country = c2.text_input("Filtro pais", value=default_country, key="ig_locator_filter_country")
    filter_top_n = c3.number_input("Top N", min_value=10, max_value=200, value=default_top_n, step=5, key="ig_locator_filter_topn")

    if c4.button("Gerar prospeccao IG", key="ig_locator_generate"):
        result = generate_ig_locator_prospect(
            load_leads(),
            niche=filter_niche.strip(),
            country=filter_country.strip(),
            top_n=int(filter_top_n),
        )
        st.success(f"Prospeccao IG atualizada: {result['selected']} lead(s)")

    locator_df = load_csv_optional(IG_LOCATOR_PROSPECT_PATH, OUTPUT_COLUMNS)
    if locator_df is None:
        st.info("prospeccao_instagram_locator.csv ainda nao existe.")
        return
    if locator_df.empty:
        st.info("Prospeccao IG vazia para os filtros atuais.")
        return

    filtered = apply_filters(locator_df, "ig_locator", has_priority=True)
    view_cols = OUTPUT_COLUMNS.copy()
    filtered = ensure_columns(filtered, view_cols)
    table_df = filtered[view_cols].copy()
    table_df = add_status_badge_column(table_df)
    preferred_cols = [
        "priority_score",
        "fonte",
        "nome",
        "nicho",
        "pais",
        "status_visual",
        "status",
        "instagram",
        "telefone",
        "whatsapp_link",
        "site",
        "cidade",
        "bairro",
        "tem_link_na_bio",
        "tem_site",
        "tem_whatsapp_visivel",
        "activity_score",
        "score",
        "observacao",
        "link_origem",
        "data_coleta",
    ]
    table_df = table_df[[c for c in preferred_cols if c in table_df.columns] + [c for c in table_df.columns if c not in preferred_cols]]

    grid_key = "grid_ig_locator"
    selected_before = selected_from_grid(filtered, grid_key)
    with st.container(border=True):
        if not selected_before.empty:
            render_selection_menu("ig_locator", selected_before)
        st.dataframe(
            table_df,
            use_container_width=True,
            height=420,
            key=grid_key,
            on_select="rerun",
            selection_mode=["single-row", "single-cell"],
            column_config={
                "status_visual": st.column_config.TextColumn("status", width="medium"),
                "observacao": st.column_config.TextColumn("observacao", width="large"),
            },
        )


def main() -> None:
    apply_responsive_styles()
    st.title("Painel Local de Leads")

    leads_exists_before = LEADS_PATH.exists()
    leads_df = load_leads()

    if not leads_exists_before:
        st.warning("Arquivo leads.csv nao encontrado. Use a aba Coleta/Busca para criar a base.")

    render_kpis(leads_df)

    tab_collect, tab_ig_locator, tab_recent, tab_prosp, tab_all, tab_utils = st.tabs(
        [
            "Coleta/Busca",
            "IG Nicho/Pais",
            "Ultima Prospeccao",
            "Prospecao (Top 50)",
            "Todos os Leads",
            "Config/Utils",
        ]
    )

    with tab_collect:
        render_collection_tab(leads_df)

    with tab_ig_locator:
        render_instagram_locator_tab(leads_df)

    with tab_recent:
        render_latest_prospect_tab()

    with tab_prosp:
        st.subheader("Prospecao (Top 50)")
        st.caption("Status: 🟨 enviado | 🟩 fechou | 🟥 ignorar | 🟦 respondeu | ⬜ novo")
        if not PROSPECT_PATH.exists():
            st.info("prospeccao.csv nao existe ainda.")
            if st.button("Gerar prospeccao agora", key="gen_missing"):
                result = generate_prospect_list(LEADS_PATH, PROSPECT_PATH, PROSPECT_XLSX_PATH)
                st.success(f"prospeccao.csv gerado com {result['selected']} lead(s).")
                st.rerun()
        else:
            prosp_df = load_csv_optional(PROSPECT_PATH, OUTPUT_COLUMNS)
            if prosp_df is None:
                st.warning("Nao foi possivel carregar prospeccao.csv")
            else:
                filtered = apply_filters(prosp_df, "prosp", has_priority=True)
                view_cols = OUTPUT_COLUMNS.copy()
                filtered = ensure_columns(filtered, view_cols)
                table_df = filtered[view_cols].copy()
                table_df = add_status_badge_column(table_df)
                preferred_cols = [
                    "priority_score",
                    "fonte",
                    "nome",
                    "nicho",
                    "pais",
                    "status_visual",
                    "status",
                    "instagram",
                    "telefone",
                    "whatsapp_link",
                    "site",
                    "cidade",
                    "bairro",
                    "tem_link_na_bio",
                    "tem_site",
                    "tem_whatsapp_visivel",
                    "activity_score",
                    "score",
                    "observacao",
                    "link_origem",
                    "data_coleta",
                ]
                table_df = table_df[[c for c in preferred_cols if c in table_df.columns] + [c for c in table_df.columns if c not in preferred_cols]]

                grid_key = "grid_prosp"
                selected_before = selected_from_grid(filtered, grid_key)
                with st.container(border=True):
                    if not selected_before.empty:
                        render_selection_menu("prosp", selected_before)
                    st.dataframe(
                        table_df,
                        use_container_width=True,
                        height=420,
                        key=grid_key,
                        on_select="rerun",
                        selection_mode=["single-row", "single-cell"],
                        column_config={
                            "status_visual": st.column_config.TextColumn("status", width="medium"),
                            "observacao": st.column_config.TextColumn("observacao", width="large"),
                        },
                    )

    with tab_all:
        st.subheader("Todos os Leads")
        st.caption("Status: 🟨 enviado | 🟩 fechou | 🟥 ignorar | 🟦 respondeu | ⬜ novo")

        all_df = ensure_columns(leads_df, LEADS_REQUIRED_COLUMNS)
        filtered = apply_filters(all_df, "all", has_priority=False)
        table_df = filtered.copy()
        table_df = add_status_badge_column(table_df)
        preferred_cols = [
            "fonte",
            "nome",
            "nicho",
            "pais",
            "status_visual",
            "status",
            "instagram",
            "telefone",
            "whatsapp_link",
            "site",
            "cidade",
            "bairro",
            "tem_link_na_bio",
            "tem_site",
            "tem_whatsapp_visivel",
            "activity_score",
            "score",
            "observacao",
            "last_contact_note",
            "last_contact_at",
            "link_origem",
            "data_coleta",
        ]
        table_df = table_df[[c for c in preferred_cols if c in table_df.columns] + [c for c in table_df.columns if c not in preferred_cols]]

        grid_key = "grid_all"
        selected_before = selected_from_grid(filtered, grid_key)
        with st.container(border=True):
            if not selected_before.empty:
                render_selection_menu("all", selected_before)
            st.dataframe(
                table_df,
                use_container_width=True,
                height=420,
                key=grid_key,
                on_select="rerun",
                selection_mode=["single-row", "single-cell"],
                column_config={
                    "status_visual": st.column_config.TextColumn("status", width="medium"),
                    "observacao": st.column_config.TextColumn("observacao", width="large"),
                    "last_contact_note": st.column_config.TextColumn("last_contact_note", width="large"),
                },
            )

    with tab_utils:
        st.subheader("Config/Utils")
        render_outreach_analytics(leads_df)
        st.divider()
        render_contact_settings()

        c1, c2, c3 = st.columns(3)

        if c1.button("Recarregar tela"):
            st.rerun()

        if c2.button("Gerar prospeccao"):
            result = generate_prospect_list(LEADS_PATH, PROSPECT_PATH, PROSPECT_XLSX_PATH)
            st.success(f"Prospeccao atualizada: {result['selected']} leads")

        if c3.button("Salvar leads.csv agora"):
            save_leads(load_leads())
            st.success("leads.csv salvo.")

        st.divider()
        st.write("Arquivos")
        if LEADS_PATH.exists():
            st.caption(f"Leads: {LEADS_PATH}")
            leads_bytes = _safe_read_bytes(LEADS_PATH)
            if leads_bytes is None:
                # Fallback when file is temporarily locked (e.g. sync software).
                leads_bytes = ensure_columns(leads_df.copy(), LEADS_REQUIRED_COLUMNS).to_csv(
                    index=False, encoding="utf-8-sig"
                ).encode("utf-8-sig")
                st.warning("Leads bloqueado no disco no momento. Download gerado da memoria.")
            st.download_button("Baixar leads.csv", leads_bytes, file_name="leads.csv")

        if PROSPECT_PATH.exists():
            st.caption(f"Prospeccao: {PROSPECT_PATH}")
            prosp_bytes = _safe_read_bytes(PROSPECT_PATH)
            if prosp_bytes is None:
                st.warning("prospeccao.csv bloqueado no momento. Tente novamente em alguns segundos.")
            else:
                st.download_button("Baixar prospeccao.csv", prosp_bytes, file_name="prospeccao.csv")
        else:
            st.info("prospeccao.csv ainda nao existe.")

        if IG_LOCATOR_PROSPECT_PATH.exists():
            st.caption(f"Prospeccao IG Locator: {IG_LOCATOR_PROSPECT_PATH}")
            ig_prosp_bytes = _safe_read_bytes(IG_LOCATOR_PROSPECT_PATH)
            if ig_prosp_bytes is None:
                st.warning("prospeccao_instagram_locator.csv bloqueado no momento. Tente novamente.")
            else:
                st.download_button(
                    "Baixar prospeccao_instagram_locator.csv",
                    ig_prosp_bytes,
                    file_name="prospeccao_instagram_locator.csv",
                )
        else:
            st.info("prospeccao_instagram_locator.csv ainda nao existe.")

        if IG_LOCATOR_PROSPECT_XLSX_PATH.exists():
            ig_prosp_xlsx = _safe_read_bytes(IG_LOCATOR_PROSPECT_XLSX_PATH)
            if ig_prosp_xlsx is not None:
                st.download_button(
                    "Baixar prospeccao_instagram_locator.xlsx",
                    ig_prosp_xlsx,
                    file_name="prospeccao_instagram_locator.xlsx",
                )

        if EXCLUDED_PATH.exists():
            st.caption(f"Excluidos: {EXCLUDED_PATH}")
            excluded_bytes = _safe_read_bytes(EXCLUDED_PATH)
            if excluded_bytes is None:
                st.warning("excluidos.csv bloqueado no momento. Tente novamente em alguns segundos.")
            else:
                st.download_button("Baixar excluidos.csv", excluded_bytes, file_name="excluidos.csv")


if __name__ == "__main__":
    main()








































