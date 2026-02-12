from pathlib import Path

import pandas as pd


def dedupe_key(row) -> str:
    fonte = str(row.get("fonte", "")).strip().lower()
    link_origem = str(row.get("link_origem", "")).strip().lower()
    if link_origem:
        return f"{fonte}|{link_origem}"

    nome = str(row.get("nome", "")).strip().lower()
    telefone = str(row.get("telefone", "")).strip()
    return f"{fonte}|{nome}|{telefone}"


def ensure_csv(path, columns):
    csv_path = Path(path)
    if not csv_path.exists():
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False, encoding="utf-8-sig")
        return

    existing = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    normalized = existing.copy()
    for col in columns:
        if col not in normalized.columns:
            normalized[col] = ""
    normalized = normalized[columns]
    normalized.to_csv(csv_path, index=False, encoding="utf-8-sig")


def load_existing(path) -> set:
    csv_path = Path(path)
    if not csv_path.exists():
        return set()

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return {dedupe_key(row) for _, row in df.iterrows()}


def append_rows(path, rows):
    if not rows:
        return

    csv_path = Path(path)
    existing = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    new_rows = pd.DataFrame(rows)
    updated = pd.concat([existing, new_rows], ignore_index=True)
    updated.to_csv(csv_path, index=False, encoding="utf-8-sig")


def update_rows_by_key(path, updates, target_fields=None):
    if not updates:
        return 0

    csv_path = Path(path)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    key_to_index = {dedupe_key(row): idx for idx, row in df.iterrows()}

    if target_fields is None:
        target_fields = {"instagram", "tem_link_na_bio", "observacao", "score"}
    else:
        target_fields = set(target_fields)

    applied = 0

    for item in updates:
        key = dedupe_key(item)
        idx = key_to_index.get(key)
        if idx is None:
            continue

        changed = False
        for field in target_fields:
            if field not in item:
                continue
            if field not in df.columns:
                df[field] = ""
            new_value = str(item[field])
            if str(df.at[idx, field]) != new_value:
                df.at[idx, field] = new_value
                changed = True
        if changed:
            applied += 1

    if applied:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return applied
