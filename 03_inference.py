# ─────────────────────────────────────────────────────────────────────────────
# Notebook 03 — Gold: Inferencia de riesgo político con Llama 4 Maverick
# ─────────────────────────────────────────────────────────────────────────────
# Fuente  : jne.silver_noticias (creado en notebook 02)
# Destino : jne.dt_inference (managed table Delta, Unity Catalog)
# Modelo  : Llama 4 Maverick vía Databricks AI Gateway (SDK OpenAI-compatible)
#
# Estrategia de inferencia (dos pasos por candidato):
#   1. Análisis individual: un llamado a Llama 4 por cada noticia → risk parcial
#   2. Consolidación: un llamado que agrega todos los análisis → risk final
# ─────────────────────────────────────────────────────────────────────────────
#
# SETUP — configurar como Databricks Secrets antes de ejecutar:
#   databricks secrets put-secret --scope jne-scope --key databricks-token
#   databricks secrets put-secret --scope jne-scope --key endpoint-url
#
# O configurar como variables de entorno en el cluster:
#   DATABRICKS_TOKEN=dapi...
#   DATABRICKS_ENDPOINT_URL=https://<workspace>.ai-gateway.cloud.databricks.com/mlflow/v1
# ─────────────────────────────────────────────────────────────────────────────

import json
import time
import os
import pandas as pd
from datetime import datetime, timezone
from openai import OpenAI
from delta.tables import DeltaTable

# ─────────────────────────────────────────
# CONFIGURACIÓN — usar Secrets o variables de entorno, nunca hardcodear tokens
# ─────────────────────────────────────────
SCHEMA_NAME     = "jne"
SILVER_TABLE    = f"{SCHEMA_NAME}.silver_noticias"
INFERENCE_TABLE = f"{SCHEMA_NAME}.dt_inference"

SERVING_ENDPOINT = "databricks-llama-4-maverick"

# Opción A: Databricks Secrets (recomendado en producción)
# ENDPOINT_TOKEN = dbutils.secrets.get(scope="jne-scope", key="databricks-token")
# ENDPOINT_URL   = dbutils.secrets.get(scope="jne-scope", key="endpoint-url")

# Opción B: Variables de entorno (para desarrollo local o CI)
ENDPOINT_TOKEN = os.environ.get("DATABRICKS_TOKEN")
ENDPOINT_URL   = os.environ.get("DATABRICKS_ENDPOINT_URL")

if not ENDPOINT_TOKEN or not ENDPOINT_URL:
    raise ValueError(
        "Configura DATABRICKS_TOKEN y DATABRICKS_ENDPOINT_URL como variables de entorno "
        "o usa Databricks Secrets. Ver README para instrucciones."
    )

client = OpenAI(api_key=ENDPOINT_TOKEN, base_url=ENDPOINT_URL)

# ─────────────────────────────────────────
# PASO 1: Leer silver_noticias
# ─────────────────────────────────────────
df_silver = spark.table(SILVER_TABLE).toPandas()
print(f"✓ Noticias totales en Silver: {len(df_silver)}")
print(f"✓ Candidatos únicos: {df_silver['txDocId'].nunique()}")


# ─────────────────────────────────────────
# PASO 2: Funciones de prompt engineering
# ─────────────────────────────────────────
def build_risk_prompt(name_candidate: str, news_title: str) -> str:
    """
    Prompt para análisis individual de una noticia.
    Fuerza output JSON estricto con risk_level: green / orange / red.
    Instruye al modelo a no inventar información fuera del contexto dado.
    """
    return f"""
Eres un analista neutral que evalúa riesgos legales y controversias de candidatos políticos del Perú.
Debes analizar el siguiente candidato utilizando ÚNICAMENTE el titular proporcionado en el contexto.

Candidato: {name_candidate}
Contexto (titular de noticia): {news_title}

Analiza si el candidato tiene antecedentes o menciones relacionadas con:
- denuncias, investigaciones fiscales, corrupción, lavado de dinero,
  enriquecimiento ilícito, procesos judiciales, escándalos políticos relevantes

Reglas:
1. Usa SOLO la información del contexto proporcionado.
2. No inventes información ni agregues datos externos.
3. Si la noticia no es claramente sobre el candidato, ignórala.
4. Si no hay evidencia clara, marca los campos como null.
5. Sé objetivo y neutral.

Clasificación de riesgo:
- "green"  → No se detectan problemas legales ni controversias relevantes.
- "orange" → Controversias políticas, denuncias menores o situaciones discutidas públicamente.
- "red"    → Investigaciones, denuncias graves o procesos judiciales relevantes.

Responde SOLO en JSON con esta estructura exacta:
{{
  "risk_level": "green | orange | red",
  "clean": true/false/null,
  "legal_issues": true/false/null,
  "controversies": true/false/null,
  "events": ["Descripción breve del evento + fuente"],
  "summary": "explicación corta y neutral del análisis",
  "confidence": 0.0
}}
"""


def build_consolidation_prompt(name_candidate: str, analyses: list) -> str:
    """
    Prompt para consolidar múltiples análisis individuales en un perfil de riesgo final.
    Aplica reglas de aggregation: risk_level toma el nivel más alto (red > orange > green).
    """
    analyses_text = "\n\n".join([
        f"Noticia {i+1}:\n"
        f"- risk_level: {a.get('risk_level')}\n"
        f"- legal_issues: {a.get('legal_issues')}\n"
        f"- controversies: {a.get('controversies')}\n"
        f"- events: {a.get('events')}\n"
        f"- summary: {a.get('summary')}\n"
        f"- confidence: {a.get('confidence')}"
        for i, a in enumerate(analyses)
    ])

    return f"""
Eres un analista neutral experto en riesgo político del Perú.
Se han analizado múltiples noticias sobre el candidato {name_candidate}.
A continuación tienes los análisis individuales de cada noticia:

{analyses_text}

Tu tarea es consolidar TODOS estos análisis en UNA sola respuesta final.

Reglas de consolidación:
1. risk_level final: toma el nivel más alto detectado (red > orange > green).
2. clean: false si algún análisis tiene legal_issues o controversies en true.
3. legal_issues: true si al menos un análisis lo detectó.
4. controversies: true si al menos un análisis lo detectó.
5. events: une todos los eventos relevantes sin repetir, máximo 5.
6. summary: redacta un párrafo corto, coherente y neutral resumiendo el perfil de riesgo del candidato.
7. confidence: promedio ponderado de los confidence individuales.

Responde SOLO en JSON con esta estructura exacta:
{{
  "risk_level": "green | orange | red",
  "clean": true/false/null,
  "legal_issues": true/false/null,
  "controversies": true/false/null,
  "events": ["evento 1", "evento 2"],
  "summary": "resumen consolidado neutral",
  "confidence": 0.0
}}
"""


def call_llama(prompt: str) -> dict:
    """
    Llama a Llama 4 Maverick vía Databricks AI Gateway (SDK OpenAI-compatible).
    Maneja errores de parsing JSON y errores de red con fallback seguro.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Responde únicamente JSON válido, sin markdown ni triple backticks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=SERVING_ENDPOINT,
            temperature=0  # determinístico para análisis de riesgo
        )
        raw = response.choices[0].message.content.strip()
        # Limpiar markdown si el modelo lo incluye de todas formas
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError:
        return {
            "risk_level": None, "clean": None, "legal_issues": None,
            "controversies": None, "events": [],
            "summary": "Error al parsear respuesta del modelo",
            "confidence": 0.0
        }
    except Exception as e:
        return {
            "risk_level": None, "clean": None, "legal_issues": None,
            "controversies": None, "events": [],
            "summary": f"Error: {str(e)}",
            "confidence": 0.0
        }


# ─────────────────────────────────────────
# PASO 3: Inferencia + consolidación por candidato
# ─────────────────────────────────────────
ingested_at   = datetime.now(timezone.utc)
final_results = []
candidatos    = df_silver["txDocId"].unique()

print(f"✓ Candidatos a procesar: {len(candidatos)}")

for txDocId in candidatos:
    df_cand  = df_silver[df_silver["txDocId"] == txDocId]
    nombre   = df_cand["nombre_candidato"].iloc[0]
    noticias = df_cand["title"].tolist()

    print(f"\n→ {nombre} ({len(noticias)} noticias)")

    # Llamada 1: analizar cada noticia individualmente
    analyses = []
    for titulo in noticias:
        prompt   = build_risk_prompt(nombre, titulo)
        analysis = call_llama(prompt)
        analyses.append(analysis)
        print(f"  · {titulo[:60]}... → {analysis.get('risk_level')}")
        time.sleep(0.3)  # respetar rate limits del AI Gateway

    # Llamada 2: consolidar todos los análisis individuales en perfil final
    consolidation_prompt = build_consolidation_prompt(nombre, analyses)
    consolidated         = call_llama(consolidation_prompt)
    time.sleep(0.3)

    print(f"  ✓ Consolidado → risk: {consolidated.get('risk_level')} | confidence: {consolidated.get('confidence')}")

    final_results.append({
        "txDocId":            txDocId,
        "idOrgPol":           df_cand["idOrgPol"].iloc[0],
        "txOrgPol":           df_cand["txOrgPol"].iloc[0],
        "nombre_candidato":   nombre,
        "noticias_analizadas": len(noticias),
        "risk_level":         consolidated.get("risk_level"),
        "clean":              consolidated.get("clean"),
        "legal_issues":       consolidated.get("legal_issues"),
        "controversies":      consolidated.get("controversies"),
        "events":             json.dumps(consolidated.get("events", []), ensure_ascii=False),
        "summary":            consolidated.get("summary"),
        "confidence":         float(consolidated.get("confidence") or 0.0),
        "ingested_at":        ingested_at,
    })

print(f"\n✓ Total candidatos procesados: {len(final_results)}")

# ─────────────────────────────────────────
# PASO 4: Guardar dt_inference con upsert Delta
# ─────────────────────────────────────────
df_inference       = pd.DataFrame(final_results)
df_inference_spark = spark.createDataFrame(df_inference)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")

if not spark.catalog.tableExists(INFERENCE_TABLE):
    df_inference_spark.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(INFERENCE_TABLE)
    print(f"✓ Tabla '{INFERENCE_TABLE}' creada con {len(final_results)} registros")
else:
    # Upsert por txDocId para permitir recargas idempotentes
    delta_table = DeltaTable.forName(spark, INFERENCE_TABLE)
    delta_table.alias("target").merge(
        df_inference_spark.alias("source"),
        "target.txDocId = source.txDocId"
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()
    print(f"✓ Upsert completado en '{INFERENCE_TABLE}'")

# ─────────────────────────────────────────
# PASO 5: Verificar resultados
# ─────────────────────────────────────────
display(spark.table(INFERENCE_TABLE).select(
    "nombre_candidato", "noticias_analizadas",
    "risk_level", "confidence", "summary"
).orderBy("risk_level", "confidence").limit(15))
