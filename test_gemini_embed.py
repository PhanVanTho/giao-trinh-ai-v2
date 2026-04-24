import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from cau_hinh import CauHinh

def test():
    key = CauHinh.GEMINI_API_KEYS[0]
    client = genai.Client(api_key=key)
    res = client.models.embed_content(
        model='models/gemini-embedding-2',
        contents=["Hello world", "This is a test"]
    )
    print("Type of res:", type(res))
    if isinstance(res, list):
        print("List length:", len(res))
        if len(res) > 0:
            print("Item type:", type(res[0]))
            print("Dir item:", dir(res[0]))
            print("Has embeddings?", hasattr(res[0], 'embeddings'))
            print("Has values?", hasattr(res[0], 'values'))
    else:
        print("Dir res:", dir(res))
        if hasattr(res, 'embeddings'):
            print("Has embeddings. type:", type(res.embeddings))
            print("len embeddings:", len(res.embeddings))
            print("item type:", type(res.embeddings[0]))
            print("item dir:", dir(res.embeddings[0]))
            if hasattr(res.embeddings[0], 'values'):
                print("Len of values:", len(res.embeddings[0].values))

test()
