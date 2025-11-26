import os
import pandas as pd
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- Configuration ---
EXCEL_FILE_PATH = "Ruchika-FusionNewAIFeatures.xlsx"
COLUMN_1 = 'Feature'
COLUMN_2 = 'Category'
COLUMN_3 = 'Description'

# --- Load Excel and create row-by-row Documents ---
all_sheets_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None)
all_documents = []
master_row_index = 0

for sheet_name, df in all_sheets_data.items():
    df['Worksheet_Name'] = sheet_name
    df = df.dropna(subset=[COLUMN_1, COLUMN_2, COLUMN_3])

    for i, (index, row) in enumerate(df.iterrows()):
        text = (
            f"Source Sheet: {sheet_name}. "
            f"Feature: {row[COLUMN_1]}. "
            f"Category: {row[COLUMN_2]}. "
            f"Description: {row[COLUMN_3]}."
        )
        doc = Document(
            page_content=text,
            metadata={
                "source_sheet": sheet_name,
                "original_row_index": index,
                "master_index": master_row_index + i
            }
        )
        all_documents.append(doc)

    master_row_index += len(df)

# --- FAISS Vector Store ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(all_documents, embeddings)

# --- OpenAI GPT-3.5 LLM ---
# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo",
#     temperature=0.1,
#     api_key=os.environ.get("OPENAI_API_KEY")
# )

# --- Ollama LLM (Free, Local) ---
llm = OllamaLLM(
    model="neural-chat",
    temperature=0.1
)

# --- Create retrieval chain with chat history ---
system_prompt = (
    "You are a helpful assistant answering questions about Excel data. "
    "Use the provided context to answer accurately. "
    "If you don't know, say so.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    vector_store.as_retriever(search_kwargs={"k": 3}),
    question_answer_chain
)

print("\n🤖 Chat with your Excel data! Type your question and press Enter.")
print("Type 'exit' to end the chat.\n")

# --- Chat-like interaction ---
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting chat...")
        break

    try:
        response = rag_chain.invoke({"input": user_input})
        answer = response['answer']
        
        # Extract sources from context
        sources = []
        if 'context' in response:
            for doc in response['context']:
                idx = doc.metadata.get("original_row_index")
                if idx is not None:
                    sources.append(idx)
        sources = sorted(list(set(sources)))

        print(f"\n🤖 Assistant: {answer}")
        print(f"🔍 Sources Used (Original DataFrame Index): {sources}\n")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
