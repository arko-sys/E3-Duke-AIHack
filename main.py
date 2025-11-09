import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import dotenv, os, re, json

# ==============================================================
# Load environment variables
# ==============================================================
dotenv.load_dotenv()

# ==============================================================
# LLM Configuration
# ==============================================================
llm = ChatOpenAI(
    model="gpt-5",  # or "gpt-4o"
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

# ==============================================================
# Neo4j Configuration
# ==============================================================
neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# ==============================================================
# Cypher Prompt Template
# ==============================================================
cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided.

Follow these rules:
1. Generate Cypher queries compatible ONLY for Neo4j Version 5.
2. Do not use EXISTS, SIZE, or HAVING keywords.
3. Use only Nodes and relationships mentioned in the schema.
4. Always use case-insensitive fuzzy matching for property searches.
5. Never use relationships not in the schema.
6. When matching projects, use OR between name/summary with case-insensitive matching.
7. Never use Cypher parameters (like $name or $clientName). Always inline all values as strings.
8. Output ONLY the Cypher statement — no prose, no Markdown, no code fences.
9. The first token must be MATCH/OPTIONAL MATCH/CALL/CREATE/MERGE/UNWIND/WITH/RETURN.
10. Use straight ASCII quotes (") in string literals. No curly quotes or backticks.
11. If the question is ambiguous, make the most reasonable assumption and proceed — never ask clarifying questions.

schema: {schema}

Question: {question}
"""

cypher_prompt = PromptTemplate(
    template=cypher_generation_template,
    input_variables=["schema", "question"],
)

# ==============================================================
# Enhanced QA Prompt
# ==============================================================
CYPHER_QA_TEMPLATE = """
You are an assistant that summarizes query results from a Neo4j graph database in a clear, human-understandable way.

The database context is a structured list of results — each row is a dictionary of properties.
Interpret this data and summarize what it tells us in plain English.
If it lists entities (like Person, Project, or Client), list them clearly.
If relationships are shown, explain what links exist between them.
If the list is empty, respond that no matches were found.

Context (query results):
{context}

Question: {question}

Helpful, concise answer:
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

# ==============================================================
# Helper Utilities
# ==============================================================
STRICT_CYPHER_PROMPT = PromptTemplate.from_template("""
You translate the question to ONE Neo4j Cypher statement (Neo4j v5).
Rules:
- Output ONE line of Cypher ONLY. No prose, no Markdown.
- First token must be MATCH/OPTIONAL MATCH/CALL/CREATE/MERGE/UNWIND/WITH/RETURN.
- Use straight quotes (") only. No curly quotes or backticks.
- Inline all values as strings (no parameters).
- Use only the provided schema.

schema:
{schema}

question:
{question}
""")

CYHER_START = re.compile(r'\b(MATCH|OPTIONAL MATCH|CALL|CREATE|MERGE|UNWIND|WITH|RETURN)\b', re.IGNORECASE)

def normalize_quotes(s: str) -> str:
    """Replace curly quotes/backticks with ASCII equivalents."""
    return (
        s.replace("“", '"')
         .replace("”", '"')
         .replace("‘", "'")
         .replace("’", "'")
         .replace("`", "")
         .strip()
    )

def extract_cypher(text: str) -> str:
    """Extract clean Cypher from text containing markdown or prose."""
    text = normalize_quotes(text)
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            m = CYHER_START.search(p)
            if m:
                return p[m.start():].strip()
    m = CYHER_START.search(text)
    if m:
        return text[m.start():].strip()
    return text

# ==============================================================
# Query Function (Final stable version)
# ==============================================================
def query_graph(user_input):
    """Main LLM → Cypher → Neo4j → Answer flow with fallback safety."""
    try:
        schema_graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_user,
            password=neo4j_password,
        )
        # Some builds expose .get_schema as a string, others as a function.
        schema_text = schema_graph.get_schema if isinstance(schema_graph.get_schema, str) else schema_graph.get_schema()
    except Exception as e:
        st.warning(f"⚠️ Could not fetch live schema: {e}")
        schema_text = ""

    # New graph instance for querying
    graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        allow_dangerous_requests=True,
    )

    try:
        # Inject schema dynamically
        result = chain({"query": user_input, "schema": schema_text})

        intermediate_steps = result.get("intermediate_steps", [])
        if intermediate_steps:
            # Normalize quotes
            if "query" in intermediate_steps[0]:
                intermediate_steps[0]["query"] = normalize_quotes(intermediate_steps[0]["query"])

            # Pretty-format JSON-like context
            if len(intermediate_steps) > 1:
                raw_ctx = intermediate_steps[1].get("context", "")
                if isinstance(raw_ctx, (list, dict)):
                    intermediate_steps[1]["context"] = json.dumps(raw_ctx, indent=2)
                elif isinstance(raw_ctx, str) and raw_ctx.startswith("["):
                    try:
                        parsed = json.loads(raw_ctx.replace("'", '"'))
                        intermediate_steps[1]["context"] = json.dumps(parsed, indent=2)
                    except Exception:
                        pass

        return result

    except Exception as e:
        st.warning("⚠️ Retrying with strict Cypher mode due to syntax issue...")

        # Generate stricter Cypher using fallback prompt
        strict_text = llm.invoke(
            STRICT_CYPHER_PROMPT.format(schema=schema_text, question=user_input)
        ).content

        safe_cypher = extract_cypher(strict_text)
        safe_cypher = normalize_quotes(safe_cypher)

        # Execute directly
        try:
            rows = graph.query(safe_cypher)
        except Exception as inner_e:
            return {
                "result": f"Execution failed even after strict mode. Error: {inner_e}",
                "intermediate_steps": [{"query": safe_cypher}, {"context": "[]"}],
            }

        # Generate human summary
        ctx_str = json.dumps(rows, indent=2) if isinstance(rows, (list, dict)) else str(rows)
        answer = llm.invoke(qa_prompt.format(context=ctx_str, question=user_input)).content

        return {
            "result": answer,
            "intermediate_steps": [
                {"query": safe_cypher},
                {"context": ctx_str},
            ],
        }

# ==============================================================
# Streamlit UI Setup
# ==============================================================
st.set_page_config(layout="wide", page_title="Conversational Neo4j Assistant", page_icon="💬")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

# ==============================================================
# Sidebar placeholder (dynamic refresh)
# ==============================================================
sidebar_placeholder = st.sidebar.empty()

def render_sidebar():
    with sidebar_placeholder.container():
        st.image("assets/icon.png", width=200)
        st.title("🔍 Query Insights")
        st.markdown("Executed Cypher Query and database results appear here.")
        st.divider()
        if "cypher_query" in st.session_state:
            st.subheader("🧩 Executed Cypher Query")
            st.text_area("Cypher", st.session_state.cypher_query, height=220)
        if "database_results" in st.session_state:
            st.subheader("🗄️ Database Results")
            st.text_area("Results", st.session_state.database_results, height=220)

# initial render
render_sidebar()

# ==============================================================
# Header
# ==============================================================
st.title("💬 Conversational E3 Assistant")
st.caption("Ask any questions to get E3 information.")

# ==============================================================
# Chat Input
# ==============================================================
user_input = st.text_input("Enter your question", key="input")

if user_input:
    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        cypher_query = ""
        database_results = ""
        answer = ""

        try:
            result = query_graph(user_input)
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps and len(intermediate_steps) >= 2:
                cypher_query = intermediate_steps[0].get("query", "")
                database_results = intermediate_steps[1].get("context", "")
            answer = result.get("result", "")
            st.session_state.system_msgs.append(answer)
            st.session_state.cypher_query = cypher_query
            st.session_state.database_results = database_results
        except Exception as e:
            st.error("Failed to process question. Please try again.")
            st.exception(e)

    st.write(f"⏱ Time taken: {timer() - start:.2f}s")

    # ✅ re-render sidebar immediately with latest query results
    render_sidebar()

# ==============================================================
# Chat Conversation (Top → Bottom)
# ==============================================================
st.subheader("💬 Conversation")

for i in range(len(st.session_state.user_msgs)):
    message(st.session_state.user_msgs[i], is_user=True, key=f"user_{i}")
    if i < len(st.session_state.system_msgs):
        message(st.session_state.system_msgs[i], key=f"assistant_{i}")
