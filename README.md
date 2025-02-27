# SalesAnalyzer AI

SalesAnalyzer AI is an advanced, AI-powered sales report analysis application built with Streamlit. It allows users to securely upload and analyze sales reports in various formats (PDF, DOCX, CSV, TXT). The app extracts key data from the reports, generates actionable insights and recommendations, and creates data visualizations using Plotly. It also provides interactive chat functionality for both analysis and recommendation, and it stores user interactions and file uploads in a SQLite database for history tracking.

## Features

- **User Management & History:**

  - Secure user login/signup and session management using SQLite.
  - Stores uploaded files, analysis history, and chat conversations.

- **File Upload & Text Extraction:**

  - Supports multiple file formats: PDF, DOCX, CSV, and TXT.
  - Extracts text content using PyPDF2, docx2txt, and Pandas.

- **Data Extraction & Analysis:**

  - Uses Google Generative AI (Gemini) to extract detailed sales data from reports.
  - Performs comprehensive analysis with prompts designed for sales insights.
  - Generates actionable development recommendations for sales growth.

- **Visualizations:**

  - Automatically generates Plotly visualization code based on the sales report.
  - Provides options for automatic, manual, and custom visualizations.
  - Executes and displays interactive graphs directly in the app.

- **Chat Functionality:**

  - Chat with an AI-powered assistant to ask questions about the report.
  - Separate chatbot for development recommendations.
  - Maintains conversation history per file for seamless interactions.

- **Semantic Search Integration:**
  - Uses LangChain and Pinecone to split and index text for similarity search.
  - Enables retrieval of relevant document sections for context-aware responses.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Jnan-py/salesanalyzer-ai.git
   cd salesanalyzer-ai
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**

   - Create a `.env` file in the project root and add your API keys:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     PINECONE_API_KEY=your_pinecone_api_key_here
     ```

5. **Initialize the Database:**
   - The app will automatically create and update the SQLite database (`salesanalyzer.db`) on first run.

## Usage

1. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

2. **Upload a Sales Report:**

   - Use the sidebar to upload a sales report file (PDF, DOCX, CSV, or TXT).
   - The app will extract the text content, process the data with Google Generative AI, and index it using Pinecone.

3. **Navigate the Features:**
   - **Analyzer:** Run a detailed analysis of the sales report to extract key insights.
   - **Data Visualizer:** Automatically or manually generate Plotly-based visualizations from the extracted data.
   - **Chatbot Assistant:** Engage in a conversation with an AI assistant to ask questions about the report.
   - **Development Recommendations:** Get actionable recommendations for business development and sales growth.
   - **Uploaded Files:** View and manage your previously uploaded sales reports.

## Project Structure

```
salesanalyzer-ai/
├── prompts.py                      #Prompts file
├── main.py                         # Main Streamlit application
├── salesanalyzer.db               # SQLite database file (auto-created)
├── datasets/                      # (Optional) Folder for sample datasets
├── downloads/                     # Folder for downloaded files (if applicable)
├── .env                           # Environment variables file (create this file)
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## Technologies Used

- **Streamlit:** For building the interactive web interface.
- **SQLite:** To store user data, file uploads, analysis history, and chat conversations.
- **Google Generative AI (Gemini):** For generating data extraction, analysis, and recommendation outputs.
- **Pinecone:** For vectorizing and indexing document text for semantic search.
- **LangChain:** For text splitting and integration with Google Generative AI.
- **Plotly & Matplotlib:** For interactive data visualizations.
- **PyPDF2 & docx2txt:** For file text extraction.
- **Pandas & NumPy:** For data manipulation and numerical processing.
- **Python-Dotenv:** For environment variable management.

---

Save these files in your project directory. To launch the SalesAnalyzer AI app, activate your virtual environment (if using one) and run:

```bash
streamlit run main.py
```

Feel free to modify the documentation as needed. Enjoy analyzing your sales reports with SalesAnalyzer AI!
