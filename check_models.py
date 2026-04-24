from google import genai
import sys

client = genai.Client(api_key='AIzaSyAkCt1vuENw7s5Lfg8zOqkCUW9huAenKAI')
try:
    for m in client.models.list():
        if "flash" in m.name:
            print(m.name)
except Exception as e:
    print("ERROR:", e)
