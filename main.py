import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import dotenv, os

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
dotenv.load_dotenv()

# -------------------------------------------------------------------
# LLM Configuration (Standard OpenAI API)
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-4o"
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

# -------------------------------------------------------------------
# Neo4j Configuration
# -------------------------------------------------------------------
neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# -------------------------------------------------------------------
# Cypher Generation Prompt
# -------------------------------------------------------------------
cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided.
Follow these rules:
1. Generate Cypher queries compatible ONLY for Neo4j Version 5.
2. Do not use EXISTS, SIZE, or HAVING keywords.
3. Use only Nodes and relationships mentioned in the schema.
4. Always use case-insensitive fuzzy matching for property searches.
5. Never use relationships not in the schema.
6. When matching projects, use OR between name/summary with case-insensitive matching.

schema: {schema}

Examples:
Question: Which client's projects use most of our people?
Answer:
```MATCH (c:CLIENT)<-[:HAS_CLIENT]-(p:Project)-[:HAS_PEOPLE]->(person:Person)
RETURN c.name AS Client, COUNT(DISTINCT person) AS NumberOfPeople
ORDER BY NumberOfPeople DESC```

Question: {question}
"""

cypher_prompt = PromptTemplate(
    template=cypher_generation_template,
    input_variables=["schema", "question"],
)

# -------------------------------------------------------------------
# Human-readable Answer Prompt
# -------------------------------------------------------------------
CYPHER_QA_TEMPLATE = """You are an assistant that creates clear and human-understandable answers.
Use the provided information directly. Do not guess or add extra knowledge.
If no information is provided, respond that you don't know the answer.

Information:
{context}

Question: {question}
Helpful Answer:
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)


# -------------------------------------------------------------------
# Query Function
# -------------------------------------------------------------------
def query_graph(user_input):
    graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        allow_dangerous_requests=True,  # Required for Cypher execution
    )
    result = chain(user_input)
    return result


# -------------------------------------------------------------------
# Streamlit UI Setup
# -------------------------------------------------------------------
st.set_page_config(layout="wide")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

title_col, _, img_col = st.columns([2, 1, 2])
with title_col:
    st.title("Conversational Neo4j Assistant")
with img_col:
    st.image(
        "https://dist.neo4j.com/wp-content/uploads/20210423062553/neo4j-social-share-21.png",
        width=200,
    )

# -------------------------------------------------------------------
# Chat Input and Processing
# -------------------------------------------------------------------
user_input = st.text_input("Enter your question", key="input")

if user_input:
    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        # Initialize variables to avoid NameError
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
        except Exception as e:
            st.error("Failed to process question. Please try again.")
            st.exception(e)

    st.write(f"⏱ Time taken: {timer() - start:.2f}s")

    # -------------------------------------------------------------------
    # Display Results
    # -------------------------------------------------------------------
    col1, col2, col3 = st.columns([1, 1, 1])

    # Chat history
    with col1:
        if st.session_state.system_msgs:
            for i in range(len(st.session_state.system_msgs) - 1, -1, -1):
                message(st.session_state.system_msgs[i], key=f"{i}_assistant")
                message(st.session_state.user_msgs[i], is_user=True, key=f"{i}_user")

    # Last Cypher Query
    with col2:
        if cypher_query:
            st.text_area("Last Cypher Query", cypher_query, key="_cypher", height=240)

    # Database Results
    with col3:
        if database_results:
            st.text_area(
                "Last Database Results",
                database_results,
                key="_database",
                height=240,
            )
