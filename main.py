import os
import time
import telebot
from flask import Flask, request
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA 
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

# 1. SETUP VARIABLES
load_dotenv() 

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL") 

# CRITICAL FIX: threaded=False forces the bot to finish work before hanging up
bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=False)
app = Flask(__name__)

# 2. LOAD BRAIN
print("⏳ Loading AI Brain...")
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )

    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
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

# 3. MESSAGE HANDLER (With Verbose Logging)
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    print(f"--- DEBUG: Received message: {message.text} ---") # LOG 1

    if not qa_chain:
        print("--- DEBUG: Brain is dead. Sending error. ---")
        bot.reply_to(message, "My brain is sleeping. Check logs.")
        return

    try:
        bot.send_chat_action(message.chat.id, 'typing')
        print("--- DEBUG: Action sent. Calling Gemini... ---") # LOG 2
        
        # Retry Logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = qa_chain.invoke({"query": message.text})
                print(f"--- DEBUG: Gemini Replied! Result: {response['result'][:50]}... ---") # LOG 3
                
                bot.reply_to(message, response['result'])
                print("--- DEBUG: Reply sent to Telegram. ---") # LOG 4
                break 
                
            except ResourceExhausted:
                print(f"⚠️ Quota hit! Waiting 10s... (Attempt {attempt+1})")
                time.sleep(10)
                if attempt == max_retries - 1:
                    bot.reply_to(message, "I am overwhelmed. Try again later.")
            except Exception as inner_e:
                print(f"⚠️ INNER ERROR: {inner_e}")
                # Don't break, let it retry or fail gracefully

    except Exception as e:
        print(f"❌ CRITICAL ERROR in handler: {e}")
        bot.reply_to(message, "Error processing request.")

# 4. SERVER
@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        
        # This function will now BLOCK until handle_message finishes
        bot.process_new_updates([update])
        
        print("--- DEBUG: Flask returning 200 OK ---") # LOG 5
        return '', 200
    return 'Forbidden', 403

@app.route('/')
def index():
    return "Bot is running on Cloud!", 200

if __name__ == "__main__":
    # Local testing only (Render ignores this)
    bot.remove_webhook()
    bot.infinity_polling()
