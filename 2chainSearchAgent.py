import os
import pandas as pd
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

load_dotenv()

# --- Configuration ---
EXCEL_FILE_PATH = "Ruchika-FusionNewAIFeatures.xlsx"
COLUMN_1 = 'Feature'
COLUMN_2 = 'Category'
COLUMN_3 = 'Description'

# --- Load all sheets from Excel ---
print(f"Loading all sheets from: {EXCEL_FILE_PATH}")
all_sheets_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None)
all_documents = []
master_row_index = 0

for sheet_name, df in all_sheets_data.items():
    print(f"Processing sheet: **{sheet_name}**")
    df['Worksheet_Name'] = sheet_name
    df = df.dropna(subset=[COLUMN_1, COLUMN_2, COLUMN_3])

    df['full_context'] = df.apply(
        lambda row: (
            f"Source Sheet: {row['Worksheet_Name']}. "
            f"Product Title: {row[COLUMN_1]}. "
            f"Key Features: {row[COLUMN_3]}. "
            f"Description: {row[COLUMN_2]}."
        ),
        axis=1
    )

    sheet_documents = [
        Document(
            page_content=str(row['full_context']),
            metadata={
                "source_sheet": sheet_name,
                "original_row_index": index,
                "master_index": master_row_index + i
            }
        )
        for i, (index, row) in enumerate(df.iterrows())
    ]
    
    all_documents.extend(sheet_documents)
    master_row_index += len(df)

print(f"Successfully processed {len(all_documents)} total data entries from all sheets.")

# --- Embeddings and FAISS Vector Store ---
print("Initializing local embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(all_documents)
vector_store = FAISS.from_documents(split_docs, embeddings)
print(f"FAISS Indexing complete for {len(split_docs)} chunks.")

# --- Initialize Ollama LLM ---
print("Initializing Ollama LLM (neural-chat)...")
llm = Ollama(model="neural-chat", temperature=0.1)

# --- Step 1: Retrieval Chain ---
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# --- Step 2: Refinement Chain ---
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "user_question"],
    template="""
You are an expert assistant. 
Given the initial answer: "{existing_answer}", 
refine and enhance it to fully answer the user question: "{user_question}".
Provide a concise, clear, and professional answer.
"""
)
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

# --- Interactive Loop for User Queries ---
while True:
    user_prompt = input("\nEnter your query for the Excel data (or type 'exit' to quit): ")
    if user_prompt.lower() in ['exit', 'quit']:
        print("Exiting...")
        break

    try:
        # Step 1: Retrieve context
        initial_result = retrieval_qa.invoke({"query": user_prompt})
        initial_answer = initial_result['result']
        print("\n--- Initial RAG Answer ---")
        print(initial_answer)

        # Step 2: Promote / Refine the answer
        refined_answer = refine_chain.run(existing_answer=initial_answer, user_question=user_prompt)
        print("\n--- Refined Answer (Chain Promoted) ---")
        print(refined_answer)

        # Step 3: Show source rows used
        sources = sorted(list(set(
            doc.metadata.get("original_row_index")
            for doc in initial_result['source_documents']
            if doc.metadata.get("original_row_index") is not None
        )))
        print("\n🔍 Sources Used (Original DataFrame Index):", sources)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
