import os
import time
from pyngrok import ngrok
import subprocess
from dotenv import load_dotenv

def deploy_app():
    """Deploy Streamlit app with ngrok tunnel"""

    load_dotenv()
    
    # set ngrok token
    NGROK_AUTH_TOKEN = os.getenv('ngrok_token')
    if not NGROK_AUTH_TOKEN:
        print(" Error: ngrok_token not found in environment variables")
        return
    
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    print("Ngrok authentication token set!")
    
    # kill any existing ngrok tunnels
    ngrok.kill()
    
    # start Streamlit app
    print("Starting Streamlit app...")
    process = subprocess.Popen([
        'streamlit', 'run', 'app.py', 
        '--server.port=8501',
        '--server.address=0.0.0.0'
    ])
    
    # wait a moment for Streamlit to start
    time.sleep(3)
    
    # start ngrok tunnel
    print("Starting ngrok tunnel...")
    public_url = ngrok.connect(8501, "http")
    
    print("\n" + "="*50)
    print("Gaming Market Research AI is running!")
    print(f"Public URL: {public_url}")
    print("="*50)
    
    # keep the process running
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n Shutting down...")
        process.terminate()

if __name__ == "__main__":
    deploy_app()
