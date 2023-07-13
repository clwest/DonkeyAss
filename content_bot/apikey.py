import os 
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv('OPENAI_API_KEY')
serper = os.getenv('SERPER_API_KEY')
hugging_face = os.getenv('HUGGING_FACE_API_KEY')