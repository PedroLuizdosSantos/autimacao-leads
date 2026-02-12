import time
from datetime import datetime, timezone

import requests


TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


def _only_digits(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def _phone_to_whatsapp_digits(phone: str) -> str:
    raw = str(phone or "").strip()
    digits = _only_digits(raw)
    if not digits:
        return ""

    # International numbers usually come with '+' or '00' prefix.
    if raw.startswith("+"):
        return digits
    if digits.startswith("00") and len(digits) > 2:
        return digits[2:]

    if 11 <= len(digits) <= 15:
        # Avoid false positive for BR mobile local format: DDD + 9XXXXXXXX.
        if len(digits) == 11 and digits.startswith("1") and digits[2] != "9":
            return digits
        # Most international formats without '+' already include country code.
        if len(digits) >= 12:
            return digits

    # Brazilian local format (DDD + numero): infer +55.
    if len(digits) in {10, 11} and not digits.startswith("55"):
        return f"55{digits}"

    # Keep as-is when it already looks international.
    return digits


def _build_whatsapp_link(phone: str) -> str:
    digits = _phone_to_whatsapp_digits(phone)
    if not digits:
        return ""
    return f"https://wa.me/{digits}"


def _calc_score(site: str, telefone: str, whatsapp_link: str, bairro: str) -> int:
    score = 0
    if not site:
        score += 4
    if telefone or whatsapp_link:
        score += 2
    if bairro:
        score += 1
    return score


def _extract_bairro(formatted_address: str, city: str) -> str:
    if not formatted_address:
        return ""

    parts = [part.strip() for part in formatted_address.split(",") if part.strip()]
    if len(parts) < 2:
        return ""

    city_token = city.split("-")[0].strip().lower()
    skip_tokens = {"brasil", "brazil", "sp"}

    for candidate in parts[1:]:
        lowered = candidate.lower()
        if any(ch.isdigit() for ch in lowered):
            continue
        if city_token in lowered:
            continue
        if lowered in skip_tokens:
            continue
        if "-" in lowered and "sp" in lowered:
            continue
        return candidate
    return ""


def _request_json(url: str, params: dict, sleep_seconds: float, retries: int = 2):
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            time.sleep(sleep_seconds)
            return payload
        except requests.RequestException as exc:
            if attempt == attempts:
                print(f"[Maps] Erro de request apos {attempts} tentativas: {exc}")
                return None
            wait_time = sleep_seconds * attempt
            print(f"[Maps] Falha de request ({attempt}/{attempts}). Retry em {wait_time:.1f}s.")
            time.sleep(wait_time)
    return None


def _fetch_place_details(place_id: str, api_key: str, sleep_seconds: float):
    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,international_phone_number,website,url,geometry,types",
        "key": api_key,
    }
    payload = _request_json(DETAILS_URL, params, sleep_seconds)
    if not payload:
        return None

    status = payload.get("status")
    if status != "OK":
        print(f"[Maps] Details ignorado para place_id={place_id}. status={status}")
        return None
    return payload.get("result", {})


def _fetch_place_ids(term: str, city: str, api_key: str, max_results: int, sleep_seconds: float):
    place_ids = []
    params = {"query": f"{term} {city}", "key": api_key}
    next_page_token = None

    while True:
        if next_page_token:
            time.sleep(2)
            params = {"pagetoken": next_page_token, "key": api_key}

        payload = _request_json(TEXT_SEARCH_URL, params, sleep_seconds)
        if not payload:
            break

        status = payload.get("status")
        if status not in {"OK", "ZERO_RESULTS"}:
            print(f"[Maps] Text Search com status={status} para termo='{term}'.")
            break
        if status == "ZERO_RESULTS":
            break

        for item in payload.get("results", []):
            place_id = item.get("place_id")
            if place_id:
                place_ids.append(place_id)
                if len(place_ids) >= max_results:
                    return place_ids

        next_page_token = payload.get("next_page_token")
        if not next_page_token:
            break

    return place_ids


def collect_maps_leads(term: str, city: str, api_key: str, max_results: int, sleep_seconds: float):
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    leads = []
    place_ids = _fetch_place_ids(term, city, api_key, max_results, sleep_seconds)

    for place_id in place_ids:
        details = _fetch_place_details(place_id, api_key, sleep_seconds)
        if not details:
            continue

        telefone = details.get("international_phone_number", "") or ""
        site = details.get("website", "") or ""
        link_origem = details.get("url", "") or ""
        bairro = _extract_bairro(details.get("formatted_address", "") or "", city)
        whatsapp_link = _build_whatsapp_link(telefone)

        row = {
            "fonte": "Maps",
            "nome": details.get("name", "") or "",
            "instagram": "",
            "telefone": telefone,
            "whatsapp_link": whatsapp_link,
            "site": site,
            "cidade": city,
            "bairro": bairro,
            "nicho": term,
            "tem_link_na_bio": "não",
            "tem_site": "sim" if site else "não",
            "tem_whatsapp_visivel": "sim" if (telefone or whatsapp_link) else "não",
            "score": _calc_score(site, telefone, whatsapp_link, bairro),
            "observacao": f"Encontrado por: {term}",
            "link_origem": link_origem,
            "status": "novo",
            "data_coleta": now_iso,
        }
        leads.append(row)

    return leads


def collect_maps_leads_batch(
    api_key: str,
    city: str,
    search_terms: list[str],
    max_results_per_term: int,
    sleep_seconds: float,
    logger=None,
):
    all_leads = []
    per_term_counts = {}

    for term in search_terms:
        clean_term = str(term).strip()
        if not clean_term:
            continue
        if logger:
            logger(f"[Maps] Coletando termo: {clean_term}")
        leads = collect_maps_leads(
            term=clean_term,
            city=city,
            api_key=api_key,
            max_results=max_results_per_term,
            sleep_seconds=sleep_seconds,
        )
        per_term_counts[clean_term] = len(leads)
        all_leads.extend(leads)
        if logger:
            logger(f"[Maps] Termo '{clean_term}' coletou {len(leads)} lead(s).")

    return all_leads, per_term_counts
