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

DOCS_DIR = "documents"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"

# -------------------- DATA PIPELINE --------------------
loader = PyPDFDirectoryLoader(DOCS_DIR) if os.path.isdir(DOCS_DIR) else None
docs = loader.load() if loader else []

chunks = []
if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if chunks:
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
else:
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
retriever = vectorstore.as_retriever()
conv_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

# -------------------- RESPONSE HELPERS --------------------
def clean_and_shorten(answer: str) -> str:
    """Shorten long answers but keep key points."""
    if len(answer.split()) > 160:
        sentences = answer.split(". ")
        return ". ".join(sentences[:4]).strip() + "..."
    return answer

def chunk_long_paragraphs(text: str, max_words: int = 1000) -> str:
    """Break long responses into ~16–18 sentence chunks."""
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
    if len(bot_text) > 160:
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
chat_memory = []
pending_chunks = []

def generate_answer(user_question: str) -> str:
    global pending_chunks
    uq = user_question.lower().strip()

    # --- Check if user asked to continue ---
    if uq in ["yes", "continue", "more", "go on", "tell me more", "next", "keep going"]:
        if pending_chunks:
            nxt = pending_chunks.pop(0)
            if pending_chunks:  # only hint again if there’s more left
                cont_prompts = [
                    "👉 Want me to go on?",
                    "👉 Should I keep going?",
                    "👉 Would you like me to continue?",
                    "👉 Shall I tell you the rest?",
                ]
                return nxt + "\n\n" + random.choice(cont_prompts)
            return nxt
        else:
            return "There’s nothing more to continue right now, what next would you love to know 🙂?"

    # ❗ Reset pending chunks if user asked a new question
    pending_chunks = []

    # --- Custom response: founder / builder / owner ---
    if "founder" in uq or "builder" in uq or "owner" in uq:
        return (
            "This assistant was built and is maintained by **Joel Tamakloe** — a data scientist and AI & Cybersecurity enthusiast. 🚀\n\n"
            "Joel has hands-on experience in creating AI-powered applications and solving real-world problems with data.\n\n"
            "The assistant was created to make information about artificial intelligence more accessible and interactive. "
            "It is powered by advanced AI models like LLaMA, Hugging Face embeddings, and Chroma for fast retrieval.\n\n"
            "Currently, it's in the testing phase with a focus on AI-related topics, aiming to expand into education and business applications."
        )

    # --- Custom response: self-intro / identity ---
    if "who are you" in uq or "myself" in uq or "yourself" in uq:
        return (
            "I’m your friendly AI chat assistant 🤖 — think of me as a curious buddy who loves talking about artificial intelligence.\n\n"
            "I use tools like **LLaMA** (for understanding language) and **Chroma** (to fetch the right info fast).\n\n"
            "My job is to make AI concepts easy to explore and fun to learn. "
            "Over time, I’ll get better at conversations, and who knows maybe add voice or video interactions too! 🎤📹\n\n"
            "So, feel free to ask me anything about AI. I’m here to chat!"
        )

    # --- Custom response: greetings ---
    if any(greet in uq for greet in ["hello", "hi", "hey"]):
        return "👋 Hello! How can I help you today?"

    # --- Normal RAG answer ---
    result = conv_chain.invoke({"question": user_question, "chat_history": chat_memory})
    full_answer = result["answer"] if isinstance(result, dict) and "answer" in result else str(result)

    chunks_list = chunk_long_paragraphs(full_answer).split("\n\n")
    pending_chunks = chunks_list[1:]
    answer = chunks_list[0]

    if pending_chunks:
        cont_prompts = [
            "👉 Want me to go on?",
            "👉 Should I keep going?",
            "👉 Would you like me to continue?",
            "👉 Shall I tell you the rest?",
        ]
        answer += "\n\n" + random.choice(cont_prompts)
    else:
        answer = clean_and_shorten(answer)
        if random.random() < 0.3:
            closers = [
                "😃 What else would you like to explore?",
                "Pretty cool, right?",
                "Does that make sense?",
                "Keep the questions coming!",
            ]
            answer += "\n\n" + random.choice(closers)

    return answer

def chat_fn(user_message, history):
    bot_text = generate_answer(user_message)

    global chat_memory
    chat_memory.append((user_message, bot_text))

    for h in stream_history_update(user_message, bot_text, history):
        yield h

def clear_chat():
    global chat_memory, pending_chunks
    chat_memory = []
    pending_chunks = []
    return []

# -------------------- GRADIO UI --------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1 style="font-size: 2.5em; color: #4CAF50; margin-bottom: 0.2em;">
                🤖 Joel AI Chat Assistant <br>
                <span style="font-size:0.8em;">(RAG-powered)</span>
            </h1>
            <p style="color: gray; margin-top: 0;">
                Ask me anything about Artificial Intelligence!
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=360,
                label="Chat with Joel AI",
                bubble_full_width=False,
                show_copy_button=True,
            )
            msg = gr.Textbox(placeholder="Type your question here...", autofocus=True)
            send_btn = gr.Button("📤 Send")
            clear = gr.Button("🧹 Clear Chat")

            msg.submit(chat_fn, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
            send_btn.click(chat_fn, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
            clear.click(clear_chat, None, chatbot, queue=False)

            

    gr.Markdown(
        """
        <div style="text-align: center; margin-top: 20px; font-size:0.9em; color: gray;">
            Built with ❤️ by <b>Joel Tamakloe</b>
        </div>
        """
    )

    msg.submit(chat_fn, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
    clear.click(clear_chat, None, chatbot, queue=False)

# -------------------- RUN --------------------
if __name__ == "__main__":
    demo.launch()
