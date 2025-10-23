import os
import time
import random
import warnings

import gradio as gr

from config import settings

# Langchain / Chroma / embeddings / LLM imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

warnings.filterwarnings("ignore")

# -------------------- SETTINGS / VECTORSTORE LOAD --------------------
persist_directory = "chroma_db"
groq_api_key = settings.groq_api_key  

# Instantiate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the existing Chroma vectorstore from disk
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# Create a retriever (tweak search_kwargs if you want more/fewer results)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create LLM (ChatGroq). Tweak model/temperature as needed
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    groq_api_key=groq_api_key
)

# Building the conversational retrieval chain
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

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
    # history is list of tuples (user, bot)
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
    """Generate (or stream) an answer using RAG, with fallback to general reasoning and a friendly tone."""
    global pending_chunks
    uq = user_question.lower().strip()

    # --- CONTINUE CHUNK HANDLING ---
    if uq in ["yes", "continue", "more", "go on", "tell me more", "next", "keep going"]:
        if pending_chunks:
            nxt = pending_chunks.pop(0)
            if pending_chunks:
                cont_prompts = [
                    "👉 Want me to go on?",
                    "👉 Should I keep going?",
                    "👉 Would you like me to continue?",
                    "👉 Shall I tell you the rest?",
                ]
                return nxt + "\n\n" + random.choice(cont_prompts)
            return nxt
        else:
            return "There’s nothing more to continue right now — what would you like to explore next? 🙂"

    pending_chunks = []  # reset chunks for new topic

    # --- CUSTOM PERSONALITY ANSWERS ---
    if "founder" in uq or "builder" in uq or "owner" in uq:
        return (
           
                "This chat assistant was built and is maintained by Mr. Joel Tamakloe, a data scientist and an AI and Cybersecurity enthusiast. "
                "Joel's background includes extensive experience in building AI-powered applications and solving real-world problems using data. "
                "This assistant was created to make information about artificial intelligence more accessible and to assist users in exploring AI concepts interactively. "
                "It is powered by advanced AI models like LLaMA and uses cutting-edge tools such as Hugging Face for embedding and Chroma for vector storage. "
                "Currently, it's in the testing phase with a focus on AI-related topics, aiming to improve its capabilities and expand into educational and business applications in the future."
            )

    if "who are you" in uq or "yourself" in uq or "myself" in uq:
        return (
            "I am an AI-powered chat assistant🤖 designed to assist users with exploring and learning about artificial intelligence and related topics. "
                "My purpose is to provide an intuitive way for users to interact with AI and gain insights on topics related to artificial intelligence. "
                "I use advanced tools and technologies like the LLaMA model, a powerful large language model, to process natural language queries, and Chroma, a vector database management system, to efficiently store and retrieve information. "
                "In the future, I plan to expand my abilities to cover more topics, improve response accuracy, and possibly integrate video and voice interaction for a more dynamic user experience. "
                "Ask me anything about A.I. I am happy to help! 😊"
        )

    if any(greet in uq for greet in ["hello", "hi", "hey"]):
        return random.choice([
            "👋 Hey there! How can I help you today?",
            "Hi! 😊 What would you like to talk about?",
            "Hey hey! Got a question for me?"
        ])

    # --- STEP 1: TRY CONTEXTUAL (RAG) ANSWER ---
    try:
        result = conv_chain.invoke({"question": user_question, "chat_history": chat_memory})
        full_answer = result.get("answer", "").strip()
    except Exception as e:
        full_answer = ""

    # --- STEP 2: DETECT IF CONTEXT WAS USEFUL ---
    if not full_answer or any(
        phrase in full_answer.lower()
        for phrase in [
            "don't know",
            "not sure",
            "no information",
            "no relevant context",
            "not mentioned in the provided context",
        ]
    ):
        # --- STEP 3: FALLBACK TO GENERAL KNOWLEDGE ---
        general_response = llm.invoke(
            f"Answer this question conversationally and clearly: {user_question}"
        )
        full_answer = general_response.content if hasattr(general_response, "content") else str(general_response)

        # Make tone friendly
        friendly_starters = [
            "Sure! 😊",
            "Of course! Here’s a quick answer:",
            "Good question! Let’s talk about that —",
            "Here’s what I can tell you 👇",
            "Absolutely — here’s a simple explanation:"
        ]
        full_answer = f"{random.choice(friendly_starters)} {full_answer}"

    # --- STEP 4: SPLIT INTO CHUNKS + ADD CLOSERS ---
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
        if random.random() < 0.35:
            closers = [
                "😃 What else would you like to explore?",
                "Pretty cool, right?",
                "Does that make sense?",
                "Want to dive into another topic?",
                "Keep the questions coming!"
            ]
            answer += "\n\n" + random.choice(closers)

    return answer


def chat_fn(user_message, history):
    """Gradio stream-compatible chat function. Yields updated history states."""
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
                show_copy_button=True,
            )
            msg = gr.Textbox(placeholder="Type your question here...", autofocus=True)
            send_btn = gr.Button("📤 Send")
            clear = gr.Button("🧹 Clear Chat")

            # Wire up interactions: streaming via generator
            msg.submit(chat_fn, [msg, chatbot], [chatbot])
            send_btn.click(chat_fn, [msg, chatbot], [chatbot])
            clear.click(clear_chat, None, chatbot, queue=False)

    gr.Markdown(
        """
        <div style="text-align: center; margin-top: 20px; font-size:0.9em; color: gray;">
            Built with ❤️ by <b>Joel Tamakloe</b>
        </div>
        """
    )

# -------------------- RUN --------------------
if __name__ == "__main__":
    demo.launch()

