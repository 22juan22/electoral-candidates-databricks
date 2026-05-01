# ─────────────────────────────────────────────────────────────────────────────
# Notebook 01 — Bronze: Ingesta de Candidatos desde API JNE
# ─────────────────────────────────────────────────────────────────────────────
# Fuente : API REST oficial JNE — votoinformado.jne.gob.pe
# Destino: jne.Bronze_candidatos (managed table Delta, Unity Catalog)
# Filtro : nuPos = 1 (candidatos presidenciales, cabeza de lista)
# ─────────────────────────────────────────────────────────────────────────────

import requests
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp
from delta.tables import DeltaTable

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
SCHEMA_NAME = "jne"
TABLE_NAME  = "Bronze_candidatos"
FULL_NAME   = f"{SCHEMA_NAME}.{TABLE_NAME}"

# Proceso electoral 124 = Elecciones Generales 2026
API_URL = "https://web.jne.gob.pe/serviciovotoinformado/api/candidatos/listarcandidatos"

API_PAYLOAD = {
    "idProcesoElectoral": 124,
    "strUbiDepartamento": "",
    "idTipoEleccion": 1
}

API_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://votoinformado.jne.gob.pe",
    "referer": "https://votoinformado.jne.gob.pe/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    # Nota: el x-session-token expira. Renovar si la API devuelve 401/403.
    "x-session-token": "RENOVAR_SI_EXPIRA",
}

# ─────────────────────────────────────────
# FASE 1: Extraer data del API JNE
# ─────────────────────────────────────────
response = requests.post(API_URL, headers=API_HEADERS, json=API_PAYLOAD, timeout=30)

if response.status_code != 200:
    raise Exception(f"Error API JNE: {response.status_code} - {response.text[:200]}")

data = response.json()
print(f"✓ Registros extraídos de API JNE: {len(data)}")

# ─────────────────────────────────────────
# FASE 2: JSON → DataFrame Spark
# ─────────────────────────────────────────
# Se usa pandas.json_normalize para compatibilidad con Databricks Serverless
# (evita sparkContext.parallelize que no está disponible en Serverless)
df_pandas = pd.json_normalize(data)
df_new    = spark.createDataFrame(df_pandas)
df_new    = df_new.withColumn("ingested_at", current_timestamp())

# Filtrar solo candidatos presidenciales (posición 1 = cabeza de lista)
df_new = df_new.filter("nuPos = 1")
print(f"✓ Candidatos presidenciales (nuPos=1): {df_new.count()}")

df_new.printSchema()

# ─────────────────────────────────────────
# FASE 3: Escribir a Delta — managed table Unity Catalog
# ─────────────────────────────────────────
# Primera ejecución: crea la tabla
# Ejecuciones siguientes: upsert por txDocId (DNI del candidato)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")

table_exists = spark.catalog.tableExists(FULL_NAME)

if not table_exists:
    df_new.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(FULL_NAME)
    print(f"✓ Tabla '{FULL_NAME}' creada con {df_new.count()} registros")

else:
    # Upsert: actualiza si el DNI ya existe, inserta si es nuevo
    delta_table = DeltaTable.forName(spark, FULL_NAME)

    delta_table.alias("target").merge(
        df_new.alias("source"),
        "target.txDocId = source.txDocId"
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()

    print(f"✓ Upsert completado en '{FULL_NAME}'")

# ─────────────────────────────────────────
# VERIFICACIÓN
# ─────────────────────────────────────────
spark.sql(f"SELECT COUNT(*) as total FROM {FULL_NAME}").show()
spark.sql(f"SELECT * FROM {FULL_NAME}").show(5, truncate=False)
