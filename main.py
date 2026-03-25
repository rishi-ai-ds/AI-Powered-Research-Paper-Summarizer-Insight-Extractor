# requirements


# neo4j
# pyvis
# openpyxl

import streamlit as st

st.set_page_config(layout="wide")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from gemini_file import ask_gemini, ask_groq

from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd
import io

# page title 

# Title
# color:#00FFFF;
st.markdown(
    "<h1 style='text-align:center; '>AI-Powered Research Paper Summarizer & Insight Extractor</h1>",
    unsafe_allow_html=True
)

# Small description
st.markdown(
    "<p style='text-align:center; color:white;'>Ask questions and get AI-powered insights from research papers</p>",
    unsafe_allow_html=True
)

# Tabs

tab1, tab2 = st.tabs(['Research Paper QA', "Knowledge Graph Explorer"])


def ask_llm(content, query):
    try:
        
        response=ask_gemini(content, query)

    except Exception as gemini_error:
        # log internally only
        print("Gemini failed:", gemini_error)
        
        try:
            response=ask_groq(content, query)
        except Exception as grok_error:
            print("Grok failed:", grok_error)
            return "Failed to get response."
    return response
        
# Tab 1-> RAG Research Paper Question Answering 

with tab1:
    st.markdown("""
    <style>
    .stTextInput input {
        background-color: #F5F5F5;
        color: #000000;
        border-radius: 10px;
        border: 2px solid #00FFFF;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    @st.cache_resource
    def load_vector_db():

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.load_local(
            "research_papers_faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_db


    vector_db = load_vector_db()

    # Show total papers
    st.write("📚 Total Papers in Database:", vector_db.index.ntotal)

    # User input
    user_query = st.text_input("🔎 Ask a question about research papers:")

    # Search button
    if st.button("Search"):

        results = vector_db.similarity_search(user_query, k=3)

        content = ""

        for idx, doc in enumerate(results, 1):

            title = doc.metadata.get("title", f"Paper {idx}")

            content += f"""
            Paper Title: {title}

            Paper Content:
            {doc.page_content}

            """

        print("whole content",content )
        # Call Gemini
        with st.spinner("🤖 Analyzing research papers and generating insights..."):
            response = ask_llm(content, user_query)
        print("response", response)
        # -------------------------
        # Extract Answer and Paper
        # -------------------------

        answer = ""
        paper_titles = []

        if "Research Paper:" in response:
            parts = response.split("Research Paper:")
            
            answer = parts[0].replace("Answer:", "").strip()

            papers_text = parts[1].strip()

            # Split multiple papers by comma
            paper_titles = [p.strip() for p in papers_text.split(",")]

        else:
            answer = response.strip()

        # Show AI answer
        st.subheader("🤖 AI Generated Insight")
        st.write(answer)

        # -------------------------
        # Show Only Relevant Paper
        # -------------------------

        if paper_titles and "none" not in [p.lower() for p in paper_titles]:

            st.subheader("📄 Relevant Research Papers")

            for doc in results:

                title = doc.metadata.get("title", "")

                for p in paper_titles:

                    if title.lower() == p.lower():

                        with st.expander(f"📄 {title}"):

                            st.write(doc.page_content)

        else:
            st.warning("No relevant research paper found.")
            
            
            
with tab2:
    st.subheader("Knowledge Graph Explorer")            
    
    
    # Neo4j connection
    @st.cache_resource
    def get_driver():
        return GraphDatabase.driver(
            "neo4j://127.0.0.1:7687",
            auth=('neo4j','neo4j123')
        )
    
    driver = get_driver()
    
    # get domain
    
    @st.cache_data
    def get_domain():
        
        query=""" 
        MATCH (d:Domain)
        RETURN d.name as domain
        """
        
        with driver.session() as session:
            result = session.run(query)
            domains = [r["domain"] for r in result]
        
        normalized={}
        
        for d in domains:
            normalized[d.lower()]=d.title()
            
        return sorted(normalized.values())
        
        
    # domain selector
    domain = st.selectbox(
        "Select Research Domain",
        get_domain()
    )
    
    # Fetch Graph Data
    def get_graph_data(domain):
        
        query= """ 
        
        MATCH (p:Paper)-[:BELONGS_TO]->(d:Domain)
        WHERE toLower(d.name) = toLower($domain)
        
        OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)
        OPTIONAL MATCH (p)-[:USES]-(m:Method)
        
        RETURN p.title AS paper,
        a.name AS author,
        m.name AS method,
        d.name AS domain
        """
        
        with driver.session() as session:
            result = session.run(query, domain=domain)
            return [r.data() for r in result]
        
    
    # Draw Graph
    
    def draw_graph(data):
        
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#111111",
            font_color="white"
        )
        
        for row in data:
            paper = row['paper']
            author = row['author']
            method = row['method']
            domain = row['domain']
            
            net.add_node(paper, label=paper, color="orange")
            
            if author:
                net.add_node(author, label=author, color="skyblue")
                net.add_edge(author, paper)
                
            if method:
                net.add_node(method, label=method, color="green")
                net.add_edge(paper, method)
                
            if domain:
                net.add_node(domain, label=domain, color="purple")
                net.add_edge(paper, domain)
                
        net.save_graph("graph.html")
        
        with open("graph.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=600)
    
    
    # display data
    if domain:
      st.subheader(f"Knowledge Graph for Domain: {domain}")  
      
      data = get_graph_data(domain)
      
      if len(data)==0:
          st.warning("No papers found")
          
      else:
        df= pd.DataFrame(data)
          
        papers = df['paper'].nunique()
        authors = df['author'].nunique()
        methods = df['method'].nunique()
          
        col1, col2, col3 =st.columns(3)
          
        col1.metric("Total Papers", papers)
        col2.metric("Total Authors", authors)
        col3.metric("Total Methods", methods)
          
        st.divider()
          
          
        st.subheader("Filtered Research Data")
        st.dataframe(df, use_container_width=True)
          
          
        #Export Excel Button
        
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        
        st.download_button(
            "Export Excel",
            excel_buffer,
            file_name=f"{domain}_research_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
          
        st.divider()
        
        st.subheader("Knowledge Graph Visualization")
        draw_graph(data)
          
          
        
            