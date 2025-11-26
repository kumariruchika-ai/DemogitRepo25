import os
import pandas as pd
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

load_dotenv()
# --- Configuration ---
# IMPORTANT: Replace 'your_data.xlsx' with your actual file path.
# IMPORTANT: Ensure your Excel file has columns named 'Title', 'Description', and 'Features'.
EXCEL_FILE_PATH = "Ruchika-FusionNewAIFeatures.xlsx"
COHERE_MODEL = "text-embedding-3-small" # Use 'claude-3-opus-20240229' for highest quality
#USER_PROMPT = "tell description for -Financial reporting narratives and explanations- feature from finance sheet?"
USER_PROMPT = input("Enter your query for the Excel data: ")

# Define the columns to combine
COLUMN_1 = 'Feature'
COLUMN_2 = 'Category'
COLUMN_3 = 'Description'


# --- New Data Loading and Context Creation ---

print(f"Loading all sheets from: {EXCEL_FILE_PATH}")
# Load ALL sheets into a dictionary of DataFrames
all_sheets_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None) 

# This list will hold all LangChain Document objects from all sheets
all_documents = []
master_row_index = 0

for sheet_name, df in all_sheets_data.items():
    print(f"Processing sheet: **{sheet_name}**")
    
    # 1. Add the sheet name as a new column for context
    df['Worksheet_Name'] = sheet_name
    
    # Filter out rows with missing data in the key columns
    df = df.dropna(subset=[COLUMN_1, COLUMN_2, COLUMN_3])

    # 2. Create the combined text column, now including the sheet name
    df['full_context'] = df.apply(
        lambda row: (
            f"Source Sheet: {row['Worksheet_Name']}. "  # ⬅️ **NEW CONTEXT ADDED**
            f"Product Title: {row[COLUMN_1]}. "
            f"Key Features: {row[COLUMN_3]}. "
            f"Description: {row[COLUMN_2]}."
        ),
        axis=1
    )

    # 3. Create LangChain Document objects for this sheet
    sheet_documents = [
        Document(
            # The page_content now includes the sheet name
            page_content=str(row['full_context']),
            # The metadata can store the sheet name and the original index
            metadata={
                "source_sheet": sheet_name,
                "original_row_index": index,
                # Use a master index if you need a unique ID across the entire file
                "master_index": master_row_index + i
            }
        )
        for i, (index, row) in enumerate(df.iterrows())
    ]
    
    all_documents.extend(sheet_documents)
    master_row_index += len(df)


print(f"Successfully processed {len(all_documents)} total data entries from all sheets.")

# --- 2. Vector Index Creation (FAISS) ---

print("2. Initializing local embedding model...")
# Use a fast, open-source local embedding model
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print(f"FATAL ERROR: Could not load HuggingFaceEmbeddings. Did you run 'pip install sentence-transformers' ? Error: {e}")
    exit()

# Optional: Split the documents into smaller chunks if they are very long
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(all_documents)

# Create the FAISS Vector Store
print(f"   -> Creating FAISS index for {len(split_docs)} chunks...")
vector_store = FAISS.from_documents(split_docs, embeddings)
print("   -> FAISS Indexing complete.")

# --- 3. Claude LLM and RAG Chain Setup ---

# if not os.environ.get("COHERE_API_KEY"):
#     print("\nFATAL ERROR: COHERE_API_KEY environment variable is not set. Please set it to use Cohere.")
#     exit()

# Initialize the Cohere LLM
print(f"\n3. Initializing Cohere LLM ({COHERE_MODEL})...")
# llm = ChatCohere(
#     model=COHERE_MODEL,
#     temperature=0.1,   # Lower temperature for factual Q&A
#     cohere_api_key=os.environ.get("COHERE_API_KEY")
# )

# llm = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0.1,
#     api_key=os.environ.get("OPENAI_API_KEY")
# )

print("\n3. Initializing Ollama LLM (neural-chat)...")
llm = Ollama(model="neural-chat", temperature=0.1)


# Set up the RetrievalQA chain
print("4. Setting up RetrievalQA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)
# --- 4. Run the RAG Query ---

print(f"\n5. Running Query: '{USER_PROMPT}'")
try:
    result = qa_chain.invoke({"query": USER_PROMPT})
    
    # --- 5. Display Results ---
    print("\n" + "="*70)
    print("🤖 CLAUDE AGENT RESPONSE")
    print("="*70)
    print(result['result'])
    print("="*70)
    print("🔍 SOURCES (Retrieved Excel Row Indices)")
    
    # Extract unique source row indices from the retrieved documents
    # sources = sorted(list(set(doc.metadata.get("source_row") for doc in result['source_documents'])))
    sources = sorted(list(set(doc.metadata.get("original_row_index") for doc in result['source_documents'] if doc.metadata.get("original_row_index") is not None)))    
    print(f"Sources Used (Original DataFrame Index): {sources}")
    print("="*70)
    
except Exception as e:
    print(f"\nAn error occurred : {e}")
 