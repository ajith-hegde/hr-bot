import os
import time
import telebot
from flask import Flask, request
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
# RESTORED YOUR IMPORT:
from langchain_classic.chains import RetrievalQA 
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

# 1. SETUP VARIABLES
load_dotenv() 

# SECURE: Get keys from Render Environment Variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL") 

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# 2. LOAD BRAIN
print("⏳ Loading AI Brain...")

# Fix: Pass key explicitly
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=GOOGLE_API_KEY
)

try:
    # Load the vectorstore folder
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # Fix: Pass key explicitly
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0, 
        google_api_key=GOOGLE_API_KEY
    )
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    print("✅ Brain loaded successfully!")
except Exception as e:
    print(f"⚠️ BRAIN ERROR: {e}")
    qa_chain = None

# 3. MESSAGE HANDLER (With Auto-Retry Fix)
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if not qa_chain:
        bot.reply_to(message, "My brain is sleeping. Check logs.")
        return

    try:
        print(f"User: {message.text}")
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Retry Logic for Rate Limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = qa_chain.invoke({"query": message.text})
                bot.reply_to(message, response['result'])
                break 
            except ResourceExhausted:
                print(f"⚠️ Quota hit! Waiting 10s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(10) 
                if attempt == max_retries - 1:
                    bot.reply_to(message, "I am overwhelmed. Try again later.")

    except Exception as e:
        bot.reply_to(message, "Error processing request.")
        print(f"Error: {e}")

# 4. SERVER (Webhook Only - Local Logic Removed)
@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return '', 200
    return 'Forbidden', 403

@app.route('/')
def index():
    return "Bot is running on Cloud!", 200

if __name__ == "__main__":
    # Simplified for Render
    print("☁️ Starting Render Webhook Server...")
    bot.remove_webhook()
    time.sleep(1)
    
    if RENDER_URL:
        bot.set_webhook(url=f"{RENDER_URL}/{TELEGRAM_TOKEN}")
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

