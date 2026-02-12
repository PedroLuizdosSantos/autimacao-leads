import re
import time
import unicodedata
from datetime import datetime, timezone
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup


DUCKDUCKGO_HTML_URL = "https://duckduckgo.com/html/"
BING_SEARCH_URL = "https://www.bing.com/search"
BRAVE_SEARCH_URL = "https://search.brave.com/search"
NICH_KEYWORDS = [
    "sobrancelha",
    "henna",
    "brow",
    "designer",
    "micropigment",
    "lash",
    "beleza",
]
COUNTRY_SYNONYMS = {
    "brasil": ["brasil", "brazil", "br"],
    "portugal": ["portugal", "pt"],
    "estados unidos": ["estados unidos", "united states", "usa", "us"],
}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _norm_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _normalize_nao(value: str) -> str:
    return _norm_text(value)


def recalculate_score(row: dict) -> int:
    site = str(row.get("site", "")).strip()
    telefone = str(row.get("telefone", "")).strip()
    whatsapp_link = str(row.get("whatsapp_link", "")).strip()
    bairro = str(row.get("bairro", "")).strip()
    tem_link_na_bio = _normalize_nao(row.get("tem_link_na_bio", ""))

    score = 0
    if not site:
        score += 4
    if telefone or whatsapp_link:
        score += 2
    if bairro:
        score += 1
    if tem_link_na_bio == "nao":
        score += 3
    return score


def _seed_score(tem_link_na_bio: str) -> int:
    score = 4
    if _normalize_nao(tem_link_na_bio) == "nao":
        score += 3
    return score


def _request_text(url: str, params: dict | None, sleep_seconds: float, retries: int = 2):
    headers = {"User-Agent": USER_AGENT}
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            text = response.text or ""
            time.sleep(sleep_seconds)
            return text
        except requests.RequestException as exc:
            if attempt == attempts:
                print(f"[IG] Erro de request apos {attempts} tentativas: {exc}")
                return ""
            wait_time = sleep_seconds * attempt
            print(f"[IG] Falha de request ({attempt}/{attempts}). Retry em {wait_time:.1f}s.")
            time.sleep(wait_time)
    return ""


def _name_tokens(name: str):
    base = re.sub(r"[^a-z0-9\s]", " ", _norm_text(name))
    tokens = [t for t in base.split() if len(t) >= 3]
    return tokens[:6]


def _contains_name_hint(name: str, haystack: str) -> bool:
    tokens = _name_tokens(name)
    if not tokens:
        return False
    lowered = _norm_text(haystack)
    return any(token in lowered for token in tokens)


def _contains_niche_hint(haystack: str) -> bool:
    lowered = _norm_text(haystack)
    return any(keyword in lowered for keyword in NICH_KEYWORDS)


def _country_hints(country: str) -> list[str]:
    normalized = _norm_text(country)
    if normalized in COUNTRY_SYNONYMS:
        return COUNTRY_SYNONYMS[normalized]
    return [normalized] if normalized else []


def _contains_country_hint(country: str, haystack: str) -> bool:
    hints = _country_hints(country)
    if not hints:
        return True
    lowered = _norm_text(haystack)
    return any(hint in lowered for hint in hints)


def _extract_instagram_profile_url(raw_url: str) -> str:
    if not raw_url:
        return ""

    parsed = urlparse(raw_url)
    url = raw_url
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        uddg = query.get("uddg", [])
        if uddg:
            url = unquote(uddg[0])

    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    if host != "instagram.com":
        return ""

    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        return ""

    first = path_parts[0].lower()
    blocked_roots = {
        "p",
        "reel",
        "reels",
        "explore",
        "stories",
        "accounts",
        "about",
        "developer",
        "legal",
        "direct",
    }
    if first in blocked_roots:
        return ""

    username = path_parts[0]
    if not re.match(r"^[A-Za-z0-9._]{1,30}$", username):
        return ""
    return f"https://www.instagram.com/{username}/"


def _username_from_url(profile_url: str) -> str:
    return profile_url.rstrip("/").split("/")[-1].lower()


def _parse_duck_results(html: str, top_n: int = 10):
    soup = BeautifulSoup(html, "lxml")
    parsed = []
    for result in soup.select(".result")[:top_n]:
        link_tag = result.select_one("a.result__a")
        snippet_tag = result.select_one(".result__snippet")
        if not link_tag:
            continue
        parsed.append(
            {
                "title": link_tag.get_text(" ", strip=True),
                "snippet": snippet_tag.get_text(" ", strip=True) if snippet_tag else "",
                "url": link_tag.get("href", ""),
            }
        )
    return parsed


def _parse_bing_results(html: str, top_n: int = 10):
    soup = BeautifulSoup(html, "lxml")
    parsed = []
    for result in soup.select("li.b_algo")[:top_n]:
        link_tag = result.select_one("h2 a")
        snippet_tag = result.select_one(".b_caption p")
        if not link_tag:
            continue
        parsed.append(
            {
                "title": link_tag.get_text(" ", strip=True),
                "snippet": snippet_tag.get_text(" ", strip=True) if snippet_tag else "",
                "url": link_tag.get("href", ""),
            }
        )
    return parsed


def _parse_brave_results(html: str, top_n: int = 10):
    urls = re.findall(r"https?://(?:www\.)?instagram\.com/[A-Za-z0-9._]+/?", html or "", flags=re.IGNORECASE)
    seen = set()
    parsed = []
    for url in urls:
        normalized = _extract_instagram_profile_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        username = _username_from_url(normalized)
        parsed.append(
            {
                "title": username,
                "snippet": "Resultado Brave",
                "url": normalized,
            }
        )
        if len(parsed) >= top_n:
            break
    return parsed


def _duck_has_challenge(html: str) -> bool:
    lowered = (html or "").lower()
    return ("anomaly.js" in lowered) or ("challenge-form" in lowered)


def _search_web_results(query: str, sleep_seconds: float, top_n: int = 10):
    brave_html = _request_text(BRAVE_SEARCH_URL, {"q": query, "source": "web"}, sleep_seconds)
    if brave_html:
        parsed_brave = _parse_brave_results(brave_html, top_n=top_n)
        if parsed_brave:
            return parsed_brave

    html = _request_text(DUCKDUCKGO_HTML_URL, {"q": query}, sleep_seconds)
    if html and not _duck_has_challenge(html):
        parsed = _parse_duck_results(html, top_n=top_n)
        if parsed:
            return parsed

    # Fallback for when DDG blocks automated requests.
    bing_html = _request_text(BING_SEARCH_URL, {"q": query, "setlang": "pt-BR"}, sleep_seconds)
    if bing_html:
        parsed_bing = _parse_bing_results(bing_html, top_n=top_n)
        if parsed_bing:
            return parsed_bing

    return []


def _candidate_score(profile_url: str, nome: str, bairro: str, cidade: str, haystack: str) -> int:
    score = 0
    text = _norm_text(haystack)
    username = _norm_text(_username_from_url(profile_url).replace(".", " ").replace("_", " "))

    if _contains_niche_hint(haystack):
        score += 3

    if _contains_name_hint(nome, haystack):
        score += 4
    else:
        name_tokens = _name_tokens(nome)
        if any(token in username for token in name_tokens):
            score += 3

    if bairro and _norm_text(bairro) in text:
        score += 2

    city_token = _norm_text(cidade.split("-")[0].strip())
    if city_token and city_token in text:
        score += 1

    if "instagram" in text:
        score += 1

    return score


def _build_queries(nome: str, bairro: str, cidade: str):
    city_main = cidade.split("-")[0].strip()
    q1 = f"{nome} {bairro} {cidade} instagram"
    q2 = f"site:instagram.com {nome} {city_main}"
    q3 = f"{nome} instagram {city_main}"

    queries = []
    seen = set()
    for q in [q1, q2, q3]:
        clean = " ".join(q.split()).strip()
        key = _norm_text(clean)
        if clean and key not in seen:
            seen.add(key)
            queries.append(clean)
    return queries


def _build_locator_queries(niche: str, country: str, city_hint: str = ""):
    terms = [niche.strip(), country.strip(), city_hint.strip(), "instagram"]
    q1 = f"site:instagram.com {' '.join([t for t in terms if t])}"
    q2 = f"{niche} {country} perfil instagram"
    q3 = f"{niche} {country} insta"

    queries = []
    seen = set()
    for q in [q1, q2, q3]:
        clean = " ".join(q.split()).strip()
        key = _norm_text(clean)
        if clean and key not in seen:
            seen.add(key)
            queries.append(clean)
    return queries


def _find_instagram_profile(nome: str, bairro: str, cidade: str, sleep_seconds: float):
    best_url = ""
    best_score = -1

    for query in _build_queries(nome, bairro, cidade):
        results = _search_web_results(query, sleep_seconds, top_n=10)
        for item in results:
            profile_url = _extract_instagram_profile_url(item.get("url", ""))
            if not profile_url:
                continue

            haystack = f"{item.get('title', '')} {item.get('snippet', '')}"
            score = _candidate_score(profile_url, nome, bairro, cidade, haystack)
            if score > best_score:
                best_score = score
                best_url = profile_url

    if best_url and best_score >= 4:
        return best_url, f"IG encontrado por busca (score={best_score})."
    if best_url and best_score >= 2:
        return best_url, f"IG encontrado por busca (baixa confianca, score={best_score})."

    return "", "IG nao encontrado com confianca."


def _has_external_link_indicator(profile_html: str) -> bool:
    lowered = profile_html.lower()
    markers = ["link in bio", "l.instagram.com", "external_link", "bio_link"]
    if any(marker in lowered for marker in markers):
        return True

    for match in re.findall(r"https?://[^\s\"'<>]+", profile_html, flags=re.IGNORECASE):
        domain = urlparse(match).netloc.lower()
        if "instagram.com" not in domain:
            return True
    return False


def _check_link_in_bio(instagram_url: str, sleep_seconds: float):
    html = _request_text(instagram_url, None, sleep_seconds)
    if not html or len(html) < 250:
        return "não", "nao foi possivel verificar link na bio.", True, html

    if _has_external_link_indicator(html):
        return "sim", "Possui link na bio.", False, html
    return "não", "Sem link na bio.", False, html


def _extract_name_from_profile_html(html: str, username: str) -> str:
    if not html:
        return username

    soup = BeautifulSoup(html, "lxml")

    def _clean_candidate(candidate: str) -> str:
        cleaned = str(candidate).strip()
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        if lowered in {"instagram", "login • instagram", "login - instagram"}:
            return ""
        return cleaned

    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        content = og_title.get("content", "").strip()
        candidate = _clean_candidate(content.split("(")[0].strip())
        if candidate:
            return candidate

    title_tag = soup.find("title")
    if title_tag:
        title_text = title_tag.get_text(" ", strip=True)
        candidate = _clean_candidate(title_text.split("•")[0].strip())
        if candidate:
            return candidate

    return username


def _name_from_search_title(title: str, profile_url: str) -> str:
    clean = str(title or "").strip()
    if not clean:
        return _username_from_url(profile_url)
    clean = clean.split("•")[0].strip()
    clean = clean.split("-")[0].strip()
    clean = clean.split("(")[0].strip()
    return clean or _username_from_url(profile_url)


def normalize_seed_to_profile_url(seed: str) -> str:
    raw = str(seed).strip()
    if not raw:
        return ""

    if raw.startswith("@"):
        raw = raw[1:]

    if raw.startswith("http://") or raw.startswith("https://"):
        return _extract_instagram_profile_url(raw)

    username = raw.split("/")[0].strip()
    if not re.match(r"^[A-Za-z0-9._]{1,30}$", username):
        return ""
    return f"https://www.instagram.com/{username}/"


def collect_from_seeds(seeds: list[str], city: str, sleep_seconds: float):
    leads = []
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for seed in seeds:
        profile_url = normalize_seed_to_profile_url(seed)
        if not profile_url:
            print(f"[IG Seeds] Seed ignorada: {seed}")
            continue

        username = profile_url.rstrip("/").split("/")[-1]
        tem_link_na_bio, bio_obs, _, html = _check_link_in_bio(profile_url, sleep_seconds)
        nome = _extract_name_from_profile_html(html, username)

        leads.append(
            {
                "fonte": "Instagram",
                "nome": nome,
                "instagram": profile_url,
                "telefone": "",
                "whatsapp_link": "",
                "site": "",
                "cidade": city,
                "bairro": "",
                "nicho": "Instagram (seed)",
                "tem_link_na_bio": tem_link_na_bio,
                "tem_site": "não",
                "tem_whatsapp_visivel": "não",
                "score": _seed_score(tem_link_na_bio),
                "observacao": f"Seed IG. {bio_obs}",
                "link_origem": profile_url,
                "status": "novo",
                "data_coleta": now_iso,
            }
        )

    return leads


def collect_profiles_by_niche_country(
    niche: str,
    country: str,
    sleep_seconds: float,
    limit: int = 30,
    city_hint: str = "",
    check_bio: bool = False,
):
    leads = []
    seen_profiles = set()
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    target_niche = str(niche).strip()
    target_country = str(country).strip()

    if not target_niche:
        return leads

    candidates = []
    for query in _build_locator_queries(target_niche, target_country, city_hint):
        results = _search_web_results(query, sleep_seconds, top_n=max(int(limit) * 2, 20))
        for item in results:
            profile_url = _extract_instagram_profile_url(item.get("url", ""))
            if not profile_url or profile_url in seen_profiles:
                continue

            haystack = f"{item.get('title', '')} {item.get('snippet', '')} {query}"
            relevance = 0
            if _contains_niche_hint(haystack) or _norm_text(target_niche) in _norm_text(haystack):
                relevance += 4
            if _contains_country_hint(target_country, haystack):
                relevance += 3
            if "instagram" in _norm_text(haystack):
                relevance += 1

            if relevance < 3:
                continue

            seen_profiles.add(profile_url)
            candidates.append(
                {
                    "profile_url": profile_url,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "relevance": relevance,
                    "query": query,
                }
            )

    candidates = sorted(
        candidates,
        key=lambda x: (int(x.get("relevance", 0)), str(x.get("title", "")).lower()),
        reverse=True,
    )[: int(limit)]

    for item in candidates:
        profile_url = item["profile_url"]
        username = _username_from_url(profile_url)
        title_name = _name_from_search_title(item.get("title", ""), profile_url)

        if check_bio:
            tem_link_na_bio, bio_obs, not_verified, html = _check_link_in_bio(profile_url, sleep_seconds)
            nome = _extract_name_from_profile_html(html, title_name or username)
            if not_verified:
                obs = "IG Locator | Bio nao verificada."
            else:
                obs = f"IG Locator | {bio_obs}"
        else:
            tem_link_na_bio = ""
            nome = title_name
            obs = "IG Locator | Bio nao verificada (modo rapido)."

        lead = {
            "fonte": "Instagram",
            "nome": nome or username,
            "instagram": profile_url,
            "telefone": "",
            "whatsapp_link": "",
            "site": "",
            "cidade": city_hint.strip(),
            "pais": target_country,
            "bairro": "",
            "nicho": target_niche,
            "tem_link_na_bio": tem_link_na_bio,
            "tem_site": "não",
            "tem_whatsapp_visivel": "não",
            "score": 0,
            "observacao": f"{obs} Encontrado por: {item.get('query', '')}",
            "link_origem": profile_url,
            "status": "novo",
            "data_coleta": now_iso,
        }
        lead["score"] = recalculate_score(lead)
        leads.append(lead)

    return leads


def enrich_instagram_lead(row: dict, sleep_seconds: float, check_bio: bool = False):
    nome = str(row.get("nome", "")).strip()
    bairro = str(row.get("bairro", "")).strip()
    cidade = str(row.get("cidade", "")).strip()

    instagram_url, search_obs = _find_instagram_profile(nome, bairro, cidade, sleep_seconds)
    if not instagram_url:
        return None

    existing_obs = str(row.get("observacao", "")).strip()

    updated_row = dict(row)
    updated_row["instagram"] = instagram_url

    # Fast mode: only fill Instagram without checking bio.
    if check_bio:
        tem_link_na_bio, bio_obs, not_verified, _ = _check_link_in_bio(instagram_url, sleep_seconds)
        updated_row["tem_link_na_bio"] = tem_link_na_bio
        new_obs = f"{search_obs} {bio_obs}".strip()
    else:
        not_verified = False
        new_obs = f"{search_obs} Bio nao verificada (modo rapido).".strip()

    final_obs = f"{existing_obs} | {new_obs}" if existing_obs else new_obs
    updated_row["observacao"] = final_obs
    updated_row["score"] = recalculate_score(updated_row)

    return {
        "updated_row": updated_row,
        "not_verified": not_verified,
    }

