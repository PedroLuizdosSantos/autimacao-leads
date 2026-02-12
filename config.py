from pathlib import Path


CSV_PATH = Path("leads.csv")
CITY = "São José dos Campos - SP"

RUN_MODE = "maps_and_instagram"  # "maps_only" | "instagram_only" | "maps_and_instagram"
MAX_IG_SEEDS_PER_RUN = 50
IG_SEEDS_FILE = "seeds_ig.txt"

SEARCH_TERMS = [
    "designer de sobrancelhas",
    "sobrancelhas henna",
    "brow designer",
    "micropigmentação",
    "estúdio de sobrancelhas",
    "lash designer",
]
MAX_RESULTS_PER_TERM = 40
REQUEST_SLEEP_SECONDS = 1.0

COLUMNS = [
    "fonte",
    "nome",
    "instagram",
    "telefone",
    "whatsapp_link",
    "site",
    "cidade",
    "pais",
    "bairro",
    "nicho",
    "tem_link_na_bio",
    "tem_site",
    "tem_whatsapp_visivel",
    "score",
    "observacao",
    "link_origem",
    "status",
    "data_coleta",
]
