import os
import random
import time
import gradio as gr

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# -------------------- CONFIG --------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY. Set it in Hugging Face ‘Settings → Variables and secrets’.")

# -------------------- DATA PIPELINE --------------------
docs_dir = "documents"
loader = PyPDFDirectoryLoader(docs_dir) if os.path.isdir(docs_dir) else None
docs = loader.load() if loader else []

chunks = []
if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Safe collection name
if chunks:
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory="chroma_db", collection_name="knowledge_base"
    )
else:
    vectorstore = Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory="chroma_db",
    )

# LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

# Conversational retrieval chain
retriever = vectorstore.as_retriever()
conv_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

# -------------------- RESPONSE HELPERS --------------------
def clean_and_shorten(answer: str) -> str:
    """Shorten long answers but keep key points."""
    if len(answer.split()) > 120:
        sentences = answer.split(". ")
        return ". ".join(sentences[:4]).strip() + "..."
    return answer

def chunk_long_paragraphs(text: str, max_words: int = 40) -> str:
    """Break long custom responses into ~2–3 sentence chunks."""
    sentences = text.split(". ")
    chunks, buf, wc = [], [], 0
    for s in sentences:
        buf.append(s)
        wc += len(s.split())
        if wc >= max_words:
            chunks.append(". ".join(buf).strip())
            buf, wc = [], 0
    if buf:
        chunks.append(". ".join(buf).strip())
    return "\n\n".join(chunks)

def stream_history_update(user_msg: str, bot_text: str, history):
    """Stream bot_text into the last assistant bubble."""
    if len(bot_text) > 120:
        history = history + [(user_msg, "🤔 Thinking...")]
        yield history
        time.sleep(0.8)
        history[-1] = (user_msg, "")
    else:
        history = history + [(user_msg, "")]

    acc = ""
    for ch in bot_text:
        acc += ch
        history[-1] = (user_msg, acc)
        yield history
        time.sleep(0.018)

# -------------------- CHATBOT LOGIC --------------------
chat_memory = []  # keeps (user, assistant)

def generate_answer(user_question: str) -> str:
    uq = user_question.lower()

    # --- Custom: founder ---
    if "founder" in uq or "builder" in uq or "owner" in uq:
        response = (
            "This assistant was built and is maintained by **Joel Tamakloe** — a data scientist and AI & Cybersecurity enthusiast. 🚀\n\n"
            "Joel has hands-on experience in creating AI-powered applications and solving real-world problems with data.\n\n"
            "The assistant was created to make information about artificial intelligence more accessible and interactive. "
            "It is powered by advanced AI models like LLaMA, Hugging Face embeddings, and Chroma for fast retrieval.\n\n"
            "Currently, it's in the testing phase with a focus on AI-related topics, aiming to expand into education and business applications."
        )
        return chunk_long_paragraphs(response)

    # --- Custom: self-intro ---
    if "who are you" in uq or "myself" in uq or "yourself" in uq:
        base = (
            "I’m your friendly AI chat assistant 🤖 — think of me as a curious buddy who loves talking about artificial intelligence.\n\n"
            "I use tools like **LLaMA** (for understanding language) and **Chroma** (to fetch info fast).\n\n"
            "My job is to make AI concepts easy to explore and fun to learn. "
            "Over time, I’ll get better at conversations, and who knows — maybe even add voice or video interactions! 🎤📹\n\n"
            "So, feel free to ask me anything about AI. I’m here to chat!"
        )
        return chunk_long_paragraphs(base)

    # --- Greetings ---
    if any(g in uq for g in ["hello", "hi", "hey"]):
        return "👋 Hello! How can I help you today?"

    # --- Normal RAG ---
    result = conv_chain.invoke({"question": user_question, "chat_history": chat_memory})
    answer = result["answer"] if isinstance(result, dict) and "answer" in result else str(result)

    if random.random() < 0.3:
        closers = [
            "😃 What else would you like to explore?",
            "Pretty cool, right?",
            "Does that make sense?",
            "Keep the questions coming!",
        ]
        answer += "\n\n" + random.choice(closers)

    return clean_and_shorten(answer)

def chat_fn(user_message, history):
    bot_text = generate_answer(user_message)

    global chat_memory
    chat_memory.append((user_message, bot_text))

    for h in stream_history_update(user_message, bot_text, history):
        yield h

# -------------------- GRADIO UI --------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    <div style="text-align: center;">
        <h1 style="font-size: 2.5em; color: #4CAF50;">
            🤖 Joel AI Chat Assistant <br>
            <span style="font-size:0.8em;">(RAG-powered)</span>
        </h1>
        <p style="color: gray;">Ask me anything about Artificial Intelligence!</p>
    </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=480,
                label="Chat with Joel AI",
                bubble_full_width=False,
                show_copy_button=True,
            )
            msg = gr.Textbox(placeholder="Type your question here...", autofocus=True)
            clear = gr.Button("🧹 Clear Chat")

    gr.Markdown(
        """
        <div style="text-align: center; margin-top: 20px; font-size:0.9em; color: gray;">
            Built with ❤️ by <b>Joel Tamakloe</b>
        </div>
        """
    )

    msg.submit(chat_fn, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
    clear.click(lambda: [], None, chatbot, queue=False)

# -------------------- RUN --------------------
if __name__ == "__main__":
    demo.launch()
