# 🧠 E³ – Enterprise / Ecosystem Engine (Duke AI Hack)

> **E³ (Explainable Entity Expansion)** is an LLM-driven framework that autonomously extracts, expands, and connects knowledge from text into a dynamic **Neo4j knowledge graph**, complete with explainable relationship confidence and contextual grounding.

---

## 🚀 Overview  
E³ transforms unstructured enterprise data (project briefs, documents, research notes, and client profiles) into a **living knowledge graph**.  
Using **LLM-powered entity expansion**, it continuously discovers new entities and relationships, verifies them against the existing graph, and provides **conversational access** through a natural-language assistant built with Streamlit and LangChain.

Users can ask questions like:  
> “Which clients are linked to sustainability projects using AWS?”  
> “Who are the engineers connected to AlphaCorp?”  

…and E³ translates those queries into Cypher, runs them on Neo4j v5, and summarizes the findings in plain English.

---

## 🧩 Key Features  

### 1. 🔍 **LLM-Powered Entity Expansion**  
- Uses GPT-based models to **extract structured entities** (`Person`, `Client`, `Technology`, `Project`, etc.) and relationships from unstructured text.  
- Expands the graph intelligently — linking new entities to existing nodes using semantic similarity and schema grounding.  
- Incorporates **explainability** by attaching rationales and confidence scores to each new relationship (foundation for future conformal prediction layer).  

### 2. 🧠 **Conversational Graph Assistant**  
- Built on **LangChain GraphCypherQAChain**, generating safe, schema-aware Cypher queries from natural language.  
- Automatically explains query results with human-friendly summaries.  
- Real-time schema retrieval ensures adaptive query generation.

### 3. 💬 **Streamlit UI**  
- Simple chat interface to ask enterprise questions.  
- **Color-coded chat bubbles** (green = user, white = assistant).  
- **Sidebar transparency**: executed Cypher query + raw database results.  

### 4. 🔗 **Neo4j-Backed Knowledge Graph**  
- Stores projects, technologies, clients, and people with rich metadata.  
- Enables semantic navigation through enterprise knowledge — not just keyword search.

---

## 🧱 Architecture  

```
            ┌──────────────────────────────┐
            │  Unstructured Enterprise Text │
            └────────────┬─────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  LLM Entity Extractor (E³)    │
         │  → Named Entity + Relation    │
         │  → Confidence + Rationale     │
         └───────────────┬───────────────┘
                         │
            ┌────────────▼────────────┐
            │   Neo4j Knowledge Graph │
            └────────────┬────────────┘
                         │
         ┌───────────────▼──────────────┐
         │  LangChain GraphCypher Chain │
         │  (English → Cypher → Answer) │
         └───────────────┬──────────────┘
                         │
               ┌─────────▼─────────┐
               │  Streamlit UI     │
               └───────────────────┘
```

---

## 📂 Repository Structure  
```
assets/                # Logos, icons  
data/                  # Sample text and graph data  
E3_Entity_Extractor.ipynb  # LLM-based entity & relation extraction
KG_generation.ipynb    # Graph construction notebook  
main.py                # Streamlit conversational app  
cyphers.txt            # Example Cypher templates  
requirements.txt       # Python dependencies  
```

---

## 🛠 Setup & Run  

1. **Clone the repo**  
   ```bash
   git clone https://github.com/arko-sys/E3-Duke-AIHack.git
   cd E3-Duke-AIHack
   ```

2. **Add environment variables (`.env`)**  
   ```ini
   OPENAI_API_KEY=your_openai_key
   NEO4J_CONNECTION_URL=bolt://host:port
   NEO4J_USER=username
   NEO4J_PASSWORD=password
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**  
   ```bash
   streamlit run main.py
   ```

---

## ⚙️ How It Works  

1. **Entity Extraction & Expansion**  
   - The LLM parses text to identify entities + relationships.  
   - Newly found entities are linked to graph nodes using fuzzy-matching + semantic similarity.  
   - Each relationship can be stored with evidence + confidence (E³ signature).

2. **Conversational Graph Querying**  
   - User enters a question.  
   - LLM translates it into Cypher via a schema-aware prompt.  
   - The query runs on Neo4j, and the assistant summarizes the result.

3. **Transparent Interaction**  
   - Sidebar logs the executed query + result set for auditing.  
   - All transformations (NL → Cypher → Answer) remain explainable.

---

## ✅ Highlights  

- Natural-language interface to enterprise data.  
- Continuous **entity expansion** that keeps the graph up-to-date.  
- **Explainable reasoning** for new links (evidence + confidence).  
- Modular and easily extensible — can connect to other data sources (APIs, CSVs, documents).  

---

## 🧭 Future Directions  
- Add **Conformal Prediction** for statistically-bounded confidence on new relationships.  
- Integrate **Snowflake Cortex / BigQuery MCP** connectors for large-scale ingestion.  
- Visualize graph substructures and relationship provenance in-app.  
- Reinforcement loop for user feedback to refine entity mappings.

---

## 👥 Contributors  
- **Arko Bhattacharya** — Project Lead & Developer  
- Duke AI Hackathon 2025 team  
- Open contributions welcome via pull requests and issues.

---

## 📄 License  
MIT License (or your preferred license)

---

### 🌟 *E³ turns enterprise text into an evolving, explainable knowledge network — connecting people, projects, and technologies you didn’t even know were related.*
