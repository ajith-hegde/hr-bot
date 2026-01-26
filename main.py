import os
import telebot
from flask import Flask, request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# --- CONFIGURATION ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL") 
WEBHOOK_URL = f"{RENDER_URL}/{TELEGRAM_TOKEN}"

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# --- LOAD AI ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- TOOLS ---
@tool
def hr_policy_search(query: str) -> str:
    """Answers HR policy questions."""
    # Simple RAG: Fetch docs and join them
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

tools = [hr_policy_search]

# --- AGENT ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful HR Assistant. Use your tools to answer user questions."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- TELEGRAM HANDLER ---
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    try:
        response = agent_executor.invoke({"input": message.text})
        bot.reply_to(message, response['output'])
    except Exception as e:
        bot.reply_to(message, "I am updating my brain. Please try again in 1 minute.")

# --- WEBHOOK ---
@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        bot.process_new_updates([telebot.types.Update.de_json(request.get_data().decode('utf-8'))])
        return '', 200
    return 'Forbidden', 403

@app.route('/set_webhook')
def set_webhook():
    bot.remove_webhook()
    bot.set_webhook(url=WEBHOOK_URL)
    return f"✅ Connected to {WEBHOOK_URL}!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
