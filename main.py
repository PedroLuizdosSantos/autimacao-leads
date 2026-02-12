import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from collectors.instagram_collector import collect_from_seeds, enrich_instagram_lead
from collectors.maps_collector import collect_maps_leads
from config import (
    COLUMNS,
    CITY,
    CSV_PATH,
    IG_SEEDS_FILE,
    MAX_IG_SEEDS_PER_RUN,
    MAX_RESULTS_PER_TERM,
    REQUEST_SLEEP_SECONDS,
    RUN_MODE,
    SEARCH_TERMS,
)
from pipeline.prospect import generate_prospect_list
from storage.sheet import (
    append_rows,
    dedupe_key,
    ensure_csv,
    load_existing,
    update_rows_by_key,
)


def _is_no(value: str) -> bool:
    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("ã", "a")
        .replace("á", "a")
        .replace("â", "a")
        .replace("à", "a")
    )
    return normalized == "nao"


def _append_new_rows(rows: list[dict]) -> int:
    existing_keys = load_existing(CSV_PATH)
    new_rows = []

    for row in rows:
        key = dedupe_key(row)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_rows.append(row)

    append_rows(CSV_PATH, new_rows)
    return len(new_rows)


def _run_maps_collection() -> int:
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        print("Erro: variavel de ambiente GOOGLE_PLACES_API_KEY nao encontrada.")
        print("Defina a chave da Google Places API e rode novamente.")
        sys.exit(1)

    collected = []
    for term in SEARCH_TERMS:
        print(f"[Maps] Coletando termo: {term}")
        leads = collect_maps_leads(
            term=term,
            city=CITY,
            api_key=api_key,
            max_results=MAX_RESULTS_PER_TERM,
            sleep_seconds=REQUEST_SLEEP_SECONDS,
        )
        collected.extend(leads)
        print(f"[Maps] Termo '{term}' coletou {len(leads)} leads.")

    added = _append_new_rows(collected)
    print(f"[Maps] Novos leads adicionados: {added}")
    return added


def _load_seeds(file_path: str, max_items: int) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        print(f"[IG Seeds] Arquivo nao encontrado: {file_path}")
        return []

    seeds = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            seeds.append(raw)
            if len(seeds) >= max_items:
                break

    return seeds


def _run_instagram_seed_collection() -> int:
    seeds = _load_seeds(IG_SEEDS_FILE, MAX_IG_SEEDS_PER_RUN)
    if not seeds:
        print("[IG Seeds] Nenhuma seed para processar.")
        return 0

    print(f"[IG Seeds] Processando {len(seeds)} seed(s).")
    rows = collect_from_seeds(seeds=seeds, city=CITY, sleep_seconds=REQUEST_SLEEP_SECONDS)
    added = _append_new_rows(rows)
    print(f"[IG Seeds] Novos leads adicionados: {added}")
    return added


def _run_instagram_enrichment() -> None:
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    pending_ig = df[
        (df["instagram"].str.strip() == "") & (df["status"].str.strip().str.lower() == "novo")
    ].head(30)

    ig_updates = []
    ig_found = 0
    no_link = 0
    not_verified = 0

    if not pending_ig.empty:
        print(f"[IG] Enriquecendo {len(pending_ig)} lead(s) com instagram vazio.")

    for _, row in pending_ig.iterrows():
        result = enrich_instagram_lead(row.to_dict(), REQUEST_SLEEP_SECONDS)
        if not result:
            continue

        updated_row = result["updated_row"]
        ig_updates.append(updated_row)
        ig_found += 1

        if result["not_verified"]:
            not_verified += 1
        elif _is_no(updated_row.get("tem_link_na_bio", "")):
            no_link += 1

    updated_count = update_rows_by_key(CSV_PATH, ig_updates)
    print(f"[IG] Perfis encontrados: {ig_found}")
    print(f"[IG] Sem link na bio: {no_link}")
    print(f"[IG] Nao verificados: {not_verified}")
    print(f"[IG] Linhas atualizadas no CSV: {updated_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prospect", action="store_true", help="Gera prospeccao.csv/xlsx e encerra")
    args = parser.parse_args()

    ensure_csv(CSV_PATH, COLUMNS)

    if args.prospect:
        result = generate_prospect_list(CSV_PATH)
        print(f"[Prospeccao] Leads elegiveis: {result['selected']} de {result['total']}")
        print(f"[Prospeccao] CSV gerado: {result['csv']}")
        print(f"[Prospeccao] XLSX gerado: {result['xlsx']}")
        return

    if RUN_MODE == "maps_only":
        _run_maps_collection()
    elif RUN_MODE == "instagram_only":
        _run_instagram_seed_collection()
    elif RUN_MODE == "maps_and_instagram":
        _run_maps_collection()
        _run_instagram_enrichment()
    else:
        print(f"Erro: RUN_MODE invalido: {RUN_MODE}")
        print("Use: maps_only | instagram_only | maps_and_instagram")
        sys.exit(1)

    total_rows = len(pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False))
    print(f"Total de leads no CSV: {total_rows}")


if __name__ == "__main__":
    main()
