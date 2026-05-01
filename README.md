# 🗳️ Análisis de Candidatos Electorales con LLM — Databricks + Llama 4

Pipeline de datos con arquitectura **Medallion** en Databricks para análisis de riesgo político de candidatos electorales peruanos. Usa **Llama 4 Maverick** vía Databricks AI Gateway para inferencia de riesgo por candidato a partir de noticias reales.

---

## 🏗️ Arquitectura

```
API JNE (REST)          →  Bronze: bronze_candidatos   (36 candidatos presidenciales)
                                         ↓
Google News RSS          →  Bronze: bronze_noticias     (hasta 10 noticias por candidato)
                                         ↓
                            Silver: silver_noticias     (limpieza, dedup, parseo de fechas)
                                         ↓
              Llama 4 Maverick (AI Gateway)
                                         ↓
                            Gold:   dt_inference        (riesgo: green / orange / red)
```

Todas las tablas son **managed tables en Unity Catalog** sobre **Delta Lake**, corriendo en **Databricks Serverless**.

---

## 📦 Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| Plataforma | Databricks Serverless + Unity Catalog |
| Procesamiento | PySpark + pandas |
| Almacenamiento | Delta Lake (managed tables) |
| LLM | Llama 4 Maverick vía Databricks AI Gateway (OpenAI-compatible SDK) |
| Fuente candidatos | API REST oficial JNE — `web.jne.gob.pe` |
| Fuente noticias | Google News RSS con filtros de búsqueda en español |
| Lenguaje | Python 3 |

---

## 📁 Estructura del proyecto

```
electoral-candidates-databricks/
│
├── notebooks/
│   ├── 01_candidatos.py      # Bronze: ingesta API JNE → Delta
│   ├── 02_noticias.py        # Bronze + Silver: Google News RSS → Delta
│   └── 03_inference.py       # Gold: inferencia de riesgo con Llama 4
│
└── README.md
```

---

## 🔄 Pipeline detallado

### Notebook 1 — `candidatos.py` → Bronze

- Llama la API REST del JNE (`/serviciovotoinformado/api/candidatos/listarcandidatos`) con `idProcesoElectoral=124`
- Convierte el JSON response a DataFrame Spark vía `pandas.json_normalize` (compatible con Databricks Serverless, evita `sparkContext`)
- Filtra candidatos con `nuPos = 1` (cabeza de lista, candidatos presidenciales) → **36 registros**
- Escribe como managed table Delta en Unity Catalog: `jne.Bronze_candidatos`
- Implementa **upsert con Delta Merge** usando `txDocId` como clave para recargas idempotentes

**Schema Bronze candidatos:**
```
idOrgPol     long       ID organización política
txNom        string     Nombres
txApePat     string     Apellido paterno
txApeMat     string     Apellido materno
txOrgPol     string     Nombre del partido
txDocId      string     DNI (clave de merge)
txEstCand    string     Estado (INSCRITO / FALLECIDO)
idHojaVida   long       ID hoja de vida JNE
ingested_at  timestamp  Timestamp de ingesta
```

---

### Notebook 2 — `noticias.py` → Bronze + Silver

**Bronze:**
- Lee `jne.bronze_candidatos` y construye nombre completo por candidato
- Llama Google News RSS con query estructurada por candidato:
  ```
  "<nombre>" Perú (denuncia OR corrupción OR investigación OR fiscalía OR lavado OR soborno)
  ```
- Extrae hasta 10 noticias por candidato (título, URL, fecha, fuente)
- Escribe raw a `jne.bronze_noticias`

**Silver:**
- Parsea fechas RSS (`%a, %d %b %Y %H:%M:%S %Z`)
- Limpia títulos: elimina el sufijo ` - Fuente` que agrega Google News con regex
- Deduplica por `txDocId + title_clean`
- Filtra registros sin fecha válida
- Upsert a `jne.silver_noticias` usando `txDocId + title` como clave compuesta

---

### Notebook 3 — `inference.py` → Gold

- Lee `jne.silver_noticias` y agrupa noticias por candidato
- Por cada noticia llama a **Llama 4 Maverick** vía Databricks AI Gateway (SDK compatible con OpenAI)
- Prompt de análisis individual → JSON estructurado con:
  - `risk_level`: `green` / `orange` / `red`
  - `legal_issues`: bool
  - `controversies`: bool
  - `events`: lista de eventos detectados
  - `summary`: análisis neutral
  - `confidence`: float 0.0–1.0
- Segunda llamada de **consolidación por candidato**: agrega todos los análisis individuales en un perfil de riesgo final
- Escribe resultados a `jne.dt_inference`

**Prompt engineering destacado:**
- Instrucción explícita de neutralidad: "Usa SOLO la información del contexto proporcionado"
- Regla anti-alucinación: "Si la noticia no es claramente sobre el candidato, ignórala"
- Output forzado a JSON con estructura exacta para parsing determinístico

---

## 🧠 Ejemplo de llamada a Llama 4

```python
from openai import OpenAI

client = OpenAI(api_key=ENDPOINT_TOKEN, base_url=ENDPOINT_URL)

prompt = f"""
Candidato: {nombre_candidato}
Contexto (titular de noticia): {news_title}

Clasifica el riesgo como green / orange / red.
Responde SOLO en JSON con esta estructura exacta:
{{
  "risk_level": "green | orange | red",
  "legal_issues": true/false/null,
  "controversies": true/false/null,
  "events": ["Descripción breve del evento"],
  "summary": "explicación corta y neutral",
  "confidence": 0.0
}}
"""

response = client.chat.completions.create(
    model="databricks-llama-4-maverick",
    messages=[{"role": "user", "content": prompt}]
)
```

---

## 📊 Resultado esperado — tabla `jne.dt_inference`

| candidato | partido | risk_level | legal_issues | summary |
|-----------|---------|-----------|--------------|---------|
| KEIKO SOFIA FUJIMORI | FUERZA POPULAR | red | true | Investigada por... |
| GEORGE FORSYTH | SOMOS PERU | orange | false | Controversias menores... |
| JOSE WILLIAMS | AVANZA PAIS | green | false | Sin antecedentes relevantes |

---

## 🛠️ Desafíos técnicos resueltos

| Problema | Solución |
|----------|---------|
| Schema inference en Serverless | Conversión vía `pandas.json_normalize` + `spark.createDataFrame` en lugar de `spark.read.json` |
| Compatibilidad con Databricks Serverless | Uso de managed tables en Unity Catalog sin paths DBFS explícitos |
| Upsert idempotente | Delta Merge con claves compuestas por notebook (txDocId, txDocId+title) |
| Parsing de fechas RSS | `pd.to_datetime` con formato explícito + `errors='coerce'` para fechas inválidas |
| Output estructurado del LLM | Prompt con JSON schema explícito + parsing con `json.loads` + retry en caso de respuesta malformada |

---

## ⚙️ Setup

### Requisitos
- Workspace Databricks con Serverless habilitado
- Unity Catalog configurado — crear schema: `jne`
- Acceso a Databricks AI Gateway con modelo `databricks-llama-4-maverick`

### Ejecución en orden
```
01_candidatos.py   →  crea jne.Bronze_candidatos
02_noticias.py     →  crea jne.bronze_noticias + jne.silver_noticias
03_inference.py    →  crea jne.dt_inference
```

### Dependencias
```python
%pip install requests openai pandas delta-spark
dbutils.library.restartPython()
```
