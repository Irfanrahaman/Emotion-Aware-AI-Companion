from flask import Flask, render_template, Response, jsonify, request
import threading
import google.generativeai as genai
from googletrans import Translator
from gtts import gTTS
from io import BytesIO

# Import our custom modules
from modules.camera import VideoCamera
from modules.voice_emotion import AudioProcessor

app = Flask(__name__)

# --- PASTE YOUR API KEY HERE ---
from config import GOOGLE_API_KEY
# -----------------------------

# Configure the Gemini API
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini API configured successfully.")
except Exception as e:
    model = None
    print(f"❌ Error configuring Gemini API: {e}")

# For translation
translator = Translator()

# For sharing data between threads
app_state = {
    'face_emotion': 'Neutral',
    'voice_emotion': 'Neutral'
}

# Initialize the audio processor
audio_processor = AudioProcessor(app_state)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(app_state)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    data = request.get_json()
    user_text = data.get('text')
    lang = data.get('lang', 'en-US')
    
    detected_emotion = app_state.get('face_emotion', 'Neutral')
    
    user_text_en = translator.translate(user_text, dest='en').text if lang.startswith('bn') else user_text

    ai_response_en = "Sorry, I couldn't process that request."
    if model:
        # --- ✅ নতুন, উন্নত Prompt ---
        prompt = f"""
        You are 'Aura', an empathetic AI mood coach. Your goal is to help the user feel better.
        The user is currently feeling: '{detected_emotion}'.
        Their spoken message is: "{user_text_en}".

        Follow these steps in your response:
        1.  Acknowledge and validate their feeling in a warm and caring tone.
        2.  Based on their detected emotion, provide one or two simple, actionable suggestions. For example, if they are sad, suggest listening to calming music or stepping outside for fresh air. If they are angry, suggest a simple breathing exercise.
        3.  If they ask a direct question, answer it concisely while keeping their emotional state in mind.
        4.  Keep your entire response short, encouraging, and easy to understand.
        
        Your response must be in simple English.
        """
        try:
            ai_response_en = model.generate_content(prompt).text
        except Exception as e:
            print(f"Error generating content: {e}")
            ai_response_en = "I'm having a little trouble thinking right now."
    
    final_response = translator.translate(ai_response_en, dest=lang.split('-')[0]).text

    app_state['voice_emotion'] = 'Neutral'
    
    return jsonify(response_text=final_response)

# --- ✅ New Route for High-Quality Text-to-Speech ---
@app.route('/synthesize_speech')
def synthesize_speech():
    text = request.args.get('text')
    lang = request.args.get('lang', 'en')

    if not text:
        return "No text provided", 400

    # Generate speech into an in-memory file
    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang=lang, slow=False) # Set slow=True for a slower voice
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0) # Rewind the buffer

    return Response(mp3_fp, mimetype='audio/mpeg')


if __name__ == '__main__':
    audio_processor.start_listening()
    app.run(debug=True, threaded=True, use_reloader=False)