import streamlit as st
import sqlite3
import hashlib
import io
import os
import datetime
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import docx2txt
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def init_db():
    conn = sqlite3.connect('salesanalyzer.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            company_name TEXT,
            company_type TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            tab TEXT,
            file_name TEXT,
            query TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            file_name TEXT,
            file_content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            chatbot_type TEXT,
            file_name TEXT,
            conversation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    update_table_schema(c)
    conn.commit()
    return conn, c

def update_table_schema(cursor):
    
    cursor.execute("PRAGMA table_info(history)")
    columns = [info[1] for info in cursor.fetchall()]
    if "file_name" not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN file_name TEXT")
    
    cursor.execute("PRAGMA table_info(chat_history)")
    columns = [info[1] for info in cursor.fetchall()]
    if "file_name" not in columns:
        cursor.execute("ALTER TABLE chat_history ADD COLUMN file_name TEXT")

def add_history(conn, cursor, username, tab, file_name, query, response):
    cursor.execute(
        "INSERT INTO history (username, tab, file_name, query, response) VALUES (?, ?, ?, ?, ?)",
        (username, tab, file_name, query, response)
    )
    conn.commit()

def add_chat_history(conn, cursor, username, chatbot_type, file_name, conversation):
    cursor.execute(
        "INSERT INTO chat_history (username, chatbot_type, file_name, conversation) VALUES (?, ?, ?, ?)",
        (username, chatbot_type, file_name, conversation)
    )
    conn.commit()

def load_latest_chat_history(cursor, username, chatbot_type, file_name):
    cursor.execute(
        "SELECT conversation FROM chat_history WHERE username=? AND chatbot_type=? AND file_name=? ORDER BY timestamp DESC LIMIT 1",
        (username, chatbot_type, file_name)
    )
    row = cursor.fetchone()
    if row:
        try:
            conversation = json.loads(row[0])
            return conversation
        except:
            return []
    return []

extraction_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    # system_instruction=
    
)

analysis_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    # system_instruction= 
)

recommendation_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    # system_instruction= 

)

recom_prompt = """
Analyze the following sales report text and generate actionable recommendations that help businesses increase future sales, reduce losses, and ultimately maximize profits. If certain data points are missing, use your general knowledge of business practices and market trends to formulate educated recommendations. Always clearly indicate when an assumption is made and briefly explain the reasoning behind it. Under no circumstances should you invent or fabricate information. If a recommendation cannot be reasonably inferred or is genuinely absent, explicitly state "Recommendation not available." Also pay close attention to the company type.
The response should consist of minimum 500+ words
"""

analysis_prompt = """
Analyze the following sales report text and extract key insights to help businesses understand their performance, identify areas for improvement, and make data-driven decisions. If specific data points are missing, use your general knowledge of business and sales to make educated assumptions. Crucially, always explicitly state that you are making an assumption and briefly explain the reasoning behind it. Under no circumstances should you invent or fabricate information. If a data point cannot be reasonably inferred or is genuinely absent, explicitly state "Information not available." Pay close attention to the company type.
The analysis should consist of minimum 500+ words.
"""

dv_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction= """
    "You are a sophisticated data visualization AI. Your primary function is to analyze sales reports and generate code for visualizations that effectively represent the data contained within these reports. You will be provided with a sales report in various formats. Your task is to extract relevant values from the report and provide complete, runnable Python code using Plotly. The code should define a variable named 'fig' (for a single plot) or 'figs' (for multiple plots), and it should be enclosed in triple backticks.
"""
)

def call_gemini_api(model, prompt: str, input_text: str) -> str:
    combined_prompt = f"{prompt}\nHere is the sales report\n------Start of the report-----\n{input_text}\n------End of the report---------\nBot Response: "
    try:
        response = model.generate_content(combined_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

def extract_text_from_file(uploaded_file) -> str:
    file_type = uploaded_file.type.lower()
    try:
        if "pdf" in file_type:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        elif "word" in file_type or "officedocument.wordprocessingml" in file_type:
            return docx2txt.process(uploaded_file)
        elif "csv" in file_type:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            return df.to_csv(index=False)
        else:
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error extracting file: {e}"

def extract_sales_data(file_text: str) -> str:
    prompt = "Extract all the necessary data in a very detailed manner from the provided text"
    extracted_info = call_gemini_api(extraction_model, prompt, file_text)
    return extracted_info

def analyze_data(extracted_data: str, company_type: str) -> str:
    analysis = call_gemini_api(analysis_model, analysis_prompt, extracted_data)
    return analysis

def development_recommendations(extracted_data: str, company_type: str) -> str:
    recommendations = call_gemini_api(recommendation_model, recom_prompt, extracted_data)
    return recommendations

def generate_plotly_visualizations(extracted_data: str, custom_query: str = None) -> str:
    if custom_query:
        prompt = f"""
Based on the following sales report data, generate Python code using Plotly to create a visualization that meets the following custom requirement: {custom_query}.
Only provide the code snippet (enclosed in triple backticks) that defines a variable 'fig' (for a single plot) or 'figs' (for multiple plots).
Sales report data:
{extracted_data}
"""
    else:
        prompt = f"""
Analyze the following sales report data and generate Python code using Plotly to create visualizations that best represent the key insights.
Only provide the code snippet (enclosed in triple backticks) that defines a variable 'fig' (for a single plot) or 'figs' (for multiple plots).
Sales report data:
{extracted_data}
"""
    response = call_gemini_api(dv_model, prompt, extracted_data)
    return response

def extract_code_from_response(response: str) -> str:
    match = re.search(r"```(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if 'python' in code:
            code = code[6:]
        return code
    else:
        return response

def run_plotly_code(code_str: str):
    local_vars = {}
    attempt = 0
    valid = False
    while attempt < 3:
        try:
            exec(code_str, {}, local_vars)
            if 'fig' in local_vars:
                f = local_vars['fig']
                if isinstance(f, (go.Figure, dict)) or (isinstance(f, list) and all(isinstance(x, (go.Figure, dict)) for x in f)):
                    st.plotly_chart(f)
                    valid = True
                else:
                    st.error(f"'fig' is not a valid Plotly figure: {f}")

            elif 'figs' in local_vars:
                figs = local_vars['figs']
                if isinstance(figs, list) and all(isinstance(x, (go.Figure, dict)) for x in figs):
                    for f in figs:
                        st.plotly_chart(f)
                    valid = True

                else:
                    st.error("One of the objects in 'figs' is not a valid Plotly figure.")
                    valid = False

            else:
                st.error("No Plotly figure found in the executed code.")
                valid = False

            if valid:

                return

        except Exception as e:
            attempt += 1
            if attempt < 3:
                st.info("Retrying ...")
            else:
                st.warning(f"Plot is not generated, Try by generating using customized indicators")
                return

def chat_with_file_conversation(query: str, vectorstore, conversation_history: list) -> str:
    docs = vectorstore.similarity_search(query, k=1)
    context = docs[0].page_content if docs else ""
    conversation_str = "\n".join([f"{msg['role']}: {msg['message']}" for msg in conversation_history])
    prompt = f"""
You are a very experienced Sales Analyst.
Context from the document: {context}

Conversation so far:
{conversation_str}

User: {query}
Bot:"""
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction="You are a very proficient and experienced sales analyst, able to extract insights from sales reports."
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.65)
    )
    return response.text

def chat_with_recommendations(query: str, conversation_history: list, extracted_data: str, company_type: str) -> str:
    conversation_str = "\n".join([f"{msg['role']}: {msg['message']}" for msg in conversation_history])
    prompt = f"""
You are an expert in providing development recommendations for sales growth.
Company Type: {company_type}
Extracted Sales Data: {extracted_data}

Conversation so far:
{conversation_str}

User: {query}
Bot:"""
    response = recommendation_model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.65)
    )
    return response.text

def initialize_vectorstore(file_text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(file_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    index_name = "sales-chat-db"
    test_embed = embeddings.embed_query('test')

    if index_name not in [i['name'] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=len(test_embed),
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1'),
        )

    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    vector_store.add_texts(texts)
    return vector_store

def main():
    st.set_page_config(page_title= "SalesIntel AI", page_icon= ":chart_with_upwards_trend:", layout="wide")
    st.title("SalesIntel AI")
    st.sidebar.title("SalesIntel AI")
    conn, cursor = init_db()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "chat_history_assistant" not in st.session_state:
        st.session_state.chat_history_assistant = []  
    if "chat_history_recommendation" not in st.session_state:
        st.session_state.chat_history_recommendation = []  

    if not st.session_state.logged_in:
        auth_choice = st.sidebar.selectbox("Choose Authentication Option", ["Login", "Signup"])
        if auth_choice == "Signup":
            st.header("Signup")
            with st.form("signup_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                company_name = st.text_input("Company Name")
                company_type = st.text_input("Company Type (e.g., ecommerce, tech, software)")
                submitted = st.form_submit_button("Signup")
                if submitted:
                    if username and password and company_name and company_type:
                        hashed_password = hashlib.sha256(password.encode()).hexdigest()
                        try:
                            cursor.execute(
                                "INSERT INTO users (username, password, company_name, company_type) VALUES (?, ?, ?, ?)",
                                (username, hashed_password, company_name, company_type)
                            )
                            conn.commit()
                            st.success("Signup successful. Please login from the sidebar.")
                        except sqlite3.IntegrityError:
                            st.error("Username already exists. Please choose another.")
                    else:
                        st.error("Please fill in all the fields.")

        else:
            st.header("Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    if username and password:
                        hashed_password = hashlib.sha256(password.encode()).hexdigest()
                        cursor.execute(
                            "SELECT * FROM users WHERE username=? AND password=?",
                            (username, hashed_password)
                        )
                        user = cursor.fetchone()
                        if user:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.company_name = user[2]
                            st.session_state.company_type = user[3]
                            st.success("Logged in successfully!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Please try again.")
                    else:
                        st.error("Please enter both username and password.")
        return

    
    st.sidebar.write(f"**Logged in as:** {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.sidebar.markdown('---')

    uploaded_file = st.sidebar.file_uploader("Upload a Sales Report File", type=["pdf", "docx", "csv", "txt"])
    if uploaded_file is not None:
    
        if st.session_state.get("uploaded_filename") != uploaded_file.name:
            file_text = extract_text_from_file(uploaded_file)
            st.session_state.file_text = file_text
            st.session_state.uploaded_filename = uploaded_file.name

            with st.spinner("Extracting the Data.."):
                extracted_data = extract_sales_data(file_text)
            st.session_state.extracted_data = extracted_data

            with st.spinner("Initializing Storage.."):
                vs = initialize_vectorstore(file_text)
            st.session_state.vectorstore = vs

            st.session_state.chat_history_assistant = []
            st.session_state.chat_history_recommendation = []    
            st.session_state.last_visualization_file = None

            cursor.execute(
                "INSERT INTO uploaded_files (username, file_name, file_content) VALUES (?, ?, ?)",
                (st.session_state.username, uploaded_file.name, file_text)
            )
            conn.commit()
            st.sidebar.success("File processed and data extracted.")
    else:
        if "file_text" not in st.session_state:
            st.sidebar.info("Please upload a file to begin.")

    cursor.execute(
        "SELECT file_name, file_content FROM uploaded_files WHERE username=? ORDER BY timestamp DESC",
        (st.session_state.username,)
    )
    uploaded_files = cursor.fetchall()

    if st.session_state.get("uploaded_filename"):
        st.info(f"Current file being analysed: {st.session_state.uploaded_filename}")

    st.sidebar.subheader("History")
    history_tab = st.sidebar.radio("Select Tab History", ["Analyzer", "Data Visualizer"])
    cursor.execute(
        "SELECT file_name, query, response, timestamp FROM history WHERE username=? AND tab=? ORDER BY timestamp DESC",
        (st.session_state.username, history_tab)
    )

    history_entries = cursor.fetchall()
    with st.sidebar:
        if history_entries:
            for entry in history_entries:
                exp_title = f"{entry[3]} – {entry[0]} – {history_tab} – Query: {entry[1]}"
                with st.expander(exp_title):
                    if history_tab == "Data Visualizer":
                        st.markdown("**Plot:**")
                        run_plotly_code(entry[2])
                    else:
                        st.write(entry[2])
        else:
            st.info("No history yet for this tab.")

    tabs = st.tabs(["Analyzer", "Data Visualizer", "Chatbot Assistant", "Development Recommendations", "Uploaded Files"])

    with tabs[0]:
        st.header("Sales Analysis")
        if "extracted_data" in st.session_state:
            if st.button("Run Analysis"):
                with st.spinner("Getting Analysis.."):
                    analysis_result = analyze_data(st.session_state.extracted_data, st.session_state.company_type)
                st.write(analysis_result)
                add_history(conn, cursor, st.session_state.username, "Analyzer", st.session_state.uploaded_filename, "Run Analysis", analysis_result)
        else:
            st.info("Please upload a file to begin analysis.")

    with tabs[1]:
        st.header("Data Visualizations")
        if "extracted_data" in st.session_state:
            st.info("In some cases the graphs generated may not align with the report, due to unavailability of data.")
            
            if st.session_state.get("last_visualization_file") != st.session_state.uploaded_filename:
                st.subheader("Automatic Visualization (Updated)")

                response = generate_plotly_visualizations(st.session_state.extracted_data)
                code_snippet = extract_code_from_response(response)
                if 'fig.show()' in code_snippet:
                    code_snippet = code_snippet[:-10]

                run_plotly_code(code_snippet)
                st.session_state.last_visualization_file = st.session_state.uploaded_filename

            st.markdown("---")
            st.subheader("Manual Trigger")
            if st.button("Generate Visualizations"):
                with st.spinner("Generating graphs..."):
                    response = generate_plotly_visualizations(st.session_state.extracted_data)
                
                code_snippet = extract_code_from_response(response)
                if 'fig.show()' in code_snippet:
                    code_snippet = code_snippet[:-10]

                run_plotly_code(code_snippet)
                add_history(conn, cursor, st.session_state.username, "Data Visualizer", st.session_state.uploaded_filename, "Generate Visualizations", code_snippet)

            st.markdown("---")
            st.subheader("Custom Visualization")

            custom_query = st.text_input("Enter your custom visualization query", key="custom_vis_input")
            if st.button("Generate Custom Visualization"):
                with st.spinner("Generating Graphs.."):
                    response = generate_plotly_visualizations(st.session_state.extracted_data, custom_query)
                code_snippet = extract_code_from_response(response)

                if 'fig.show()' in code_snippet:
                    code_snippet = code_snippet[:-10]
                
                run_plotly_code(code_snippet)
                add_history(conn, cursor, st.session_state.username, "Data Visualizer", st.session_state.uploaded_filename, custom_query, code_snippet)
        else:
            st.info("Please upload a file from the sidebar.")


    with tabs[2]:
        st.header("Chatbot Assistant")
        if "vectorstore" in st.session_state:

            if st.session_state.get("uploaded_filename"):
                if not st.session_state.chat_history_assistant:
                    st.session_state.chat_history_assistant = load_latest_chat_history(cursor, st.session_state.username, "assistant", st.session_state.uploaded_filename) or []

            for msg in st.session_state.chat_history_assistant:
                st.chat_message(msg["role"]).write(msg["message"])
            user_input = st.chat_input("Your message")

            if user_input:
                st.session_state.chat_history_assistant.append({"role": "user", "message": user_input})
                with st.spinner("Thinking..."):
                    bot_response = chat_with_file_conversation(user_input, st.session_state.vectorstore, st.session_state.chat_history_assistant)
                st.session_state.chat_history_assistant.append({"role": "assistant", "message": bot_response})
                st.rerun()

            if st.button("Save Chat Conversation (Assistant)"):
                conversation_json = json.dumps(st.session_state.chat_history_assistant, indent=2)
                current_file = st.session_state.get("uploaded_filename", "No file")
                add_chat_history(conn, cursor, st.session_state.username, "assistant", current_file, conversation_json)
                st.success("Chat conversation saved!")
        else:
            st.info("Please upload a file to initialize the chatbot.")

    with tabs[3]:
        st.header("Development Recommendations")
        if "extracted_data" in st.session_state:
            st.subheader("Choose your Recommender Type")

            rec_mode = st.radio("Recommendation Mode", ["Get Recommendations", "Chat with Recommendation Bot"])
            st.markdown('---')

            if rec_mode == "Get Recommendations":
                if st.button("Get Recommendations"):
                    with st.spinner("Getting Recommendations.."):
                        result = development_recommendations(st.session_state.extracted_data, st.session_state.company_type)            

                    if not st.session_state.chat_history_recommendation:
                        st.session_state.chat_history_recommendation = load_latest_chat_history(cursor, st.session_state.username, "recommendation", st.session_state.uploaded_filename) or []

                    st.session_state.chat_history_recommendation.append({"role": "system", "message": "Direct Recommendation Content"})
                    st.session_state.chat_history_recommendation.append({"role": "assistant", "message": result})
                    st.markdown(result)                    

            elif rec_mode == "Chat with Recommendation Bot":
                if st.session_state.get("uploaded_filename"):
                    if not st.session_state.chat_history_recommendation:
                        st.session_state.chat_history_recommendation = load_latest_chat_history(cursor, st.session_state.username, "recommendation", st.session_state.uploaded_filename) or []

                for msg in st.session_state.chat_history_recommendation:
                    st.chat_message(msg["role"]).write(msg["message"])

                rec_input = st.chat_input("Your message", key="rec_input")

                if rec_input:
                    st.session_state.chat_history_recommendation.append({"role": "user", "message": rec_input})

                    with st.spinner("Thinking.."):
                        bot_response = chat_with_recommendations(rec_input, st.session_state.chat_history_recommendation, st.session_state.extracted_data, st.session_state.company_type)
                    
                    st.session_state.chat_history_recommendation.append({"role": "assistant", "message": bot_response})
                    st.rerun()

            if st.button("Save Chat Conversation (Recommendation)"):
                conversation_json = json.dumps(st.session_state.chat_history_recommendation, indent=2)
                current_file = st.session_state.get("uploaded_filename", "No file")
                add_chat_history(conn, cursor, st.session_state.username, "recommendation", current_file, conversation_json)
                st.success("Recommendation chat conversation saved!")

        else:
            st.info("Please upload a file from the sidebar.")

    with tabs[4]:
        st.header("Uploaded Files")
        if uploaded_files:
            file_names = [file[0] for file in uploaded_files]
            selected_file = st.selectbox("Select a file to view its content", file_names)

            for fname, content in uploaded_files:
                if fname == selected_file:
                    st.subheader(f"Content of {fname}")
                    st.markdown(content)

                    if st.button(f"Load {fname} for analysis"):
                        with st.spinner("Reloading the file..."):
                            st.session_state.file_text = content
                            st.session_state.uploaded_filename = fname
                            st.session_state.extracted_data = extract_sales_data(content)
                            st.session_state.vectorstore = initialize_vectorstore(content)
                            
                            st.session_state.chat_history_assistant = []
                            st.session_state.chat_history_recommendation = []
                            
                            st.session_state.last_visualization_file = None
                            st.rerun()

                        st.success(f"{fname} loaded for further processing.")

                    break
        else:
            st.info("No files uploaded yet.")

if __name__ == '__main__':
    main()
