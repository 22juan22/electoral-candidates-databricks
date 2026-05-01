# ─────────────────────────────────────────────────────────────────────────────
# Notebook 02 — Bronze + Silver: Ingesta y limpieza de noticias por candidato
# ─────────────────────────────────────────────────────────────────────────────
# Fuente  : Google News RSS (búsqueda por nombre de candidato)
# Origen  : jne.Bronze_candidatos (creado en notebook 01)
# Destino : jne.bronze_noticias (raw)
#           jne.silver_noticias (limpio, deduplicado, fechas parseadas)
# ─────────────────────────────────────────────────────────────────────────────

import requests
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timezone
from delta.tables import DeltaTable

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
SCHEMA_NAME  = "jne"
BRONZE_TABLE = f"{SCHEMA_NAME}.bronze_noticias"
SILVER_TABLE = f"{SCHEMA_NAME}.silver_noticias"

MAX_NEWS_PER_CANDIDATE = 10  # máximo de noticias por candidato desde Google News

# ─────────────────────────────────────────
# PASO 1: Leer candidatos desde Bronze
# ─────────────────────────────────────────
df_candidatos = spark.table("jne.bronze_candidatos").toPandas()

# Construir nombre completo desde campos separados del API JNE
df_candidatos["nombre_completo"] = (
    df_candidatos["txNom"].str.strip() + " " +
    df_candidatos["txApePat"].str.strip() + " " +
    df_candidatos["txApeMat"].str.strip()
).str.strip()

candidatos = df_candidatos[
    ["txDocId", "idOrgPol", "txOrgPol", "nombre_completo"]
].drop_duplicates()

print(f"✓ Candidatos a procesar: {len(candidatos)}")

# ─────────────────────────────────────────
# PASO 2: Scraping Google News RSS por candidato
# ─────────────────────────────────────────
def search_news(name_candidate: str) -> list:
    """
    Busca noticias de un candidato en Google News RSS filtrando por
    términos relacionados con denuncias y controversias legales.
    Retorna lista de dicts con: title, news_url, publish_date, source.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
        )
    }

    # Query: nombre exacto + términos de riesgo legal en español
    name_q  = urllib.parse.quote(f'"{name_candidate}"')
    terms_q = urllib.parse.quote(
        "Perú (denuncia OR corrupción OR investigación OR fiscalía OR lavado OR soborno)"
    )
    url = (
        f"https://news.google.com/rss/search"
        f"?q={name_q}+{terms_q}&hl=es-419&gl=PE&ceid=PE:es-419"
    )

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    tree        = ET.fromstring(response.content)
    source_news = []

    for item in tree.findall(".//item")[:MAX_NEWS_PER_CANDIDATE]:
        title    = item.find("title")
        link     = item.find("link")
        pub_date = item.find("pubDate")
        source   = item.find("source")

        if title is None or link is None:
            continue

        source_news.append({
            "title":        title.text,
            "news_url":     link.text,
            "publish_date": pub_date.text if pub_date is not None else None,
            "source":       source.text   if source  is not None else None,
        })

    return source_news


# Loop principal: iterar por candidato y recolectar noticias
ingested_at = datetime.now(timezone.utc)
bronze_rows = []

for _, row in candidatos.iterrows():
    nombre = row["nombre_completo"]
    try:
        noticias = search_news(nombre)
        for n in noticias:
            bronze_rows.append({
                "txDocId":          row["txDocId"],
                "idOrgPol":         row["idOrgPol"],
                "txOrgPol":         row["txOrgPol"],
                "nombre_candidato": nombre,
                "title":            n["title"],
                "news_url":         n["news_url"],
                "publish_date_raw": n["publish_date"],
                "source":           n["source"],
                "ingested_at":      ingested_at,
            })
        print(f"✓ {nombre}: {len(noticias)} noticias")
    except Exception as e:
        print(f"✗ Error procesando {nombre}: {e}")

print(f"\n✓ Total noticias recolectadas: {len(bronze_rows)}")

# ─────────────────────────────────────────
# PASO 3: Guardar bronze_noticias (raw, sin transformación)
# ─────────────────────────────────────────
df_bronze       = pd.DataFrame(bronze_rows)
df_bronze_spark = spark.createDataFrame(df_bronze)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")

if not spark.catalog.tableExists(BRONZE_TABLE):
    df_bronze_spark.write.format("delta").mode("overwrite").saveAsTable(BRONZE_TABLE)
    print(f"✓ Tabla '{BRONZE_TABLE}' creada")
else:
    df_bronze_spark.write.format("delta").mode("append").saveAsTable(BRONZE_TABLE)
    print(f"✓ Noticias agregadas a '{BRONZE_TABLE}'")

# ─────────────────────────────────────────
# PASO 4: Transformar → silver_noticias
# ─────────────────────────────────────────
df_silver = df_bronze.copy()

# Parsear fecha desde formato RSS: "Mon, 01 Jan 2026 12:00:00 GMT"
df_silver["publish_date"] = pd.to_datetime(
    df_silver["publish_date_raw"],
    format="%a, %d %b %Y %H:%M:%S %Z",
    errors="coerce"   # fechas inválidas → NaT
)

# Limpiar título: Google News agrega " - Nombre Fuente" al final
df_silver["title_clean"] = df_silver["title"].str.replace(
    r"\s+-\s+[^-]+$", "", regex=True
).str.strip()

# Deduplicar: mismo candidato + mismo título limpio
df_silver = df_silver.drop_duplicates(subset=["txDocId", "title_clean"])

# Eliminar registros sin fecha válida (no parseables)
df_silver = df_silver.dropna(subset=["publish_date"])

# Seleccionar y renombrar columnas finales
df_silver = df_silver[[
    "txDocId",
    "idOrgPol",
    "txOrgPol",
    "nombre_candidato",
    "title_clean",
    "news_url",
    "publish_date",
    "source",
    "ingested_at",
]].rename(columns={"title_clean": "title"})

print(f"✓ Noticias limpias para Silver: {len(df_silver)}")

# ─────────────────────────────────────────
# PASO 5: Guardar silver_noticias con upsert
# ─────────────────────────────────────────
df_silver_spark = spark.createDataFrame(df_silver)

if not spark.catalog.tableExists(SILVER_TABLE):
    df_silver_spark.write.format("delta").mode("overwrite").saveAsTable(SILVER_TABLE)
    print(f"✓ Tabla '{SILVER_TABLE}' creada")
else:
    # Upsert por (txDocId + title) para evitar duplicados en recargas
    delta_table = DeltaTable.forName(spark, SILVER_TABLE)
    delta_table.alias("target").merge(
        df_silver_spark.alias("source"),
        "target.txDocId = source.txDocId AND target.title = source.title"
    ).whenNotMatchedInsertAll().execute()
    print(f"✓ Upsert completado en '{SILVER_TABLE}'")

# ─────────────────────────────────────────
# VERIFICACIÓN
# ─────────────────────────────────────────
display(spark.table(SILVER_TABLE).limit(5))
spark.sql(f"SELECT nombre_candidato, COUNT(*) as noticias FROM {SILVER_TABLE} GROUP BY 1 ORDER BY 2 DESC").show(10)
