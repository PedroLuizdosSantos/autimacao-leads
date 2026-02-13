import time
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests


TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

TARGET_TYPES = {
    "beauty_salon",
    "hair_salon",
    "nail_salon",
    "spa",
    "health_and_beauty_business",
}
HARD_EXCLUDED_TYPES = {
    "school",
    "university",
    "hospital",
    "pharmacy",
    "bank",
    "atm",
    "restaurant",
    "lodging",
    "gas_station",
    "car_repair",
    "car_dealer",
}
TERM_HINTS = {
    "sobrancelh": {"beauty_salon", "hair_salon"},
    "brow": {"beauty_salon", "hair_salon"},
    "lash": {"beauty_salon", "hair_salon"},
    "micropigment": {"beauty_salon", "hair_salon"},
    "henna": {"beauty_salon", "hair_salon"},
}


def _norm(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _only_digits(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def _to_int(value) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _to_float(value) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


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

    if len(digits) in {10, 11} and not digits.startswith("55"):
        return f"55{digits}"

    return digits


def _build_whatsapp_link(phone: str) -> str:
    digits = _phone_to_whatsapp_digits(phone)
    if not digits:
        return ""
    return f"https://wa.me/{digits}"


def _extract_instagram_from_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    host = parsed.netloc.lower().replace("www.", "")
    if not host.endswith("instagram.com"):
        return ""

    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return ""

    root = parts[0].lower()
    blocked = {"p", "reel", "reels", "explore", "stories", "accounts", "about", "direct"}
    if root in blocked:
        return ""
    return f"https://www.instagram.com/{parts[0]}/"


def _calc_score(
    site: str,
    telefone: str,
    whatsapp_link: str,
    bairro: str,
    business_status: str,
    rating: str,
    user_ratings_total: str,
    relevance_hit: bool,
) -> int:
    score = 0
    if not site:
        score += 5
    if telefone or whatsapp_link:
        score += 3
    if bairro:
        score += 1

    status_norm = _norm(business_status)
    if status_norm == "operational":
        score += 2
    else:
        score -= 10

    rating_num = _to_float(rating)
    reviews = _to_int(user_ratings_total)
    if rating_num == 0:
        score += 1
    elif rating_num <= 4.2:
        score += 2
    elif rating_num >= 4.7:
        score -= 1

    if reviews == 0:
        score += 1
    elif reviews <= 30:
        score += 2
    elif reviews >= 300:
        score -= 2

    if relevance_hit:
        score += 2
    else:
        score -= 4
    return score


def _activity_score(
    business_status: str,
    open_now: bool | None,
    user_ratings_total: str,
) -> int:
    score = 0
    if _norm(business_status) == "operational":
        score += 3
    if open_now is True:
        score += 2
    reviews = _to_int(user_ratings_total)
    if reviews > 0:
        score += 2
    if reviews >= 20:
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
        "fields": (
            "place_id,name,formatted_address,international_phone_number,website,url,types,"
            "business_status,rating,user_ratings_total,opening_hours"
        ),
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


def _is_operational(business_status: str) -> bool:
    return str(business_status or "").strip().upper() == "OPERATIONAL"


def _term_types_match(term: str, types_set: set[str]) -> bool:
    term_norm = _norm(term)
    for token, expected_types in TERM_HINTS.items():
        if token in term_norm and types_set.intersection(expected_types):
            return True
    return False


def _term_text_match(term: str, name: str, types_set: set[str]) -> bool:
    term_tokens = [t for t in _norm(term).replace("_", " ").split() if len(t) >= 4]
    haystack = f"{_norm(name)} {' '.join(sorted(types_set))}"
    return any(tok in haystack for tok in term_tokens)


def _is_relevant_place(term: str, name: str, place_types: list[str]) -> tuple[bool, bool]:
    types_set = {str(t).strip().lower() for t in (place_types or []) if str(t).strip()}
    if not types_set:
        return False, False

    has_target_type = bool(types_set.intersection(TARGET_TYPES))
    has_hard_excluded = bool(types_set.intersection(HARD_EXCLUDED_TYPES))
    type_hint_ok = _term_types_match(term, types_set)
    text_hint_ok = _term_text_match(term, name, types_set)
    relevance_hit = has_target_type or type_hint_ok or text_hint_ok

    if has_hard_excluded and not relevance_hit:
        return False, False
    return relevance_hit, relevance_hit


def collect_maps_leads(term: str, city: str, api_key: str, max_results: int, sleep_seconds: float):
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    leads = []
    place_ids = _fetch_place_ids(term, city, api_key, max_results, sleep_seconds)

    for place_id in place_ids:
        details = _fetch_place_details(place_id, api_key, sleep_seconds)
        if not details:
            continue

        business_status_raw = str(details.get("business_status", "")).strip()
        if not _is_operational(business_status_raw):
            continue

        place_types = [str(t).strip().lower() for t in details.get("types", []) if str(t).strip()]
        name = details.get("name", "") or ""
        relevant, relevance_hit = _is_relevant_place(term, name, place_types)
        if not relevant:
            continue

        telefone = details.get("international_phone_number", "") or ""
        website_raw = details.get("website", "") or ""
        instagram = _extract_instagram_from_url(website_raw)
        site = "" if instagram else website_raw
        link_origem = details.get("url", "") or ""
        bairro = _extract_bairro(details.get("formatted_address", "") or "", city)
        whatsapp_link = _build_whatsapp_link(telefone)
        rating = str(details.get("rating", "")).strip()
        user_ratings_total = str(details.get("user_ratings_total", "")).strip()
        opening_hours = details.get("opening_hours", {}) or {}
        open_now = opening_hours.get("open_now")
        open_now_str = "sim" if open_now is True else ("nao" if open_now is False else "")

        row = {
            "fonte": "Maps",
            "place_id": details.get("place_id", "") or place_id,
            "nome": name,
            "instagram": instagram,
            "telefone": telefone,
            "whatsapp_link": whatsapp_link,
            "site": site,
            "cidade": city,
            "bairro": bairro,
            "nicho": term,
            "tem_link_na_bio": "nao",
            "tem_site": "sim" if site else "nao",
            "tem_whatsapp_visivel": "sim" if (telefone or whatsapp_link) else "nao",
            "business_status": business_status_raw,
            "rating": rating,
            "user_ratings_total": user_ratings_total,
            "place_types": ",".join(place_types),
            "open_now": open_now_str,
            "activity_score": _activity_score(
                business_status=business_status_raw,
                open_now=open_now if isinstance(open_now, bool) else None,
                user_ratings_total=user_ratings_total,
            ),
            "score": _calc_score(
                site=site,
                telefone=telefone,
                whatsapp_link=whatsapp_link,
                bairro=bairro,
                business_status=business_status_raw,
                rating=rating,
                user_ratings_total=user_ratings_total,
                relevance_hit=relevance_hit,
            ),
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
