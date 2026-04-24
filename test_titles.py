import json
from openai import OpenAI
from cau_hinh import CauHinh

def test_identify_titles(topic):
    client = OpenAI(api_key=CauHinh.OPENAI_API_KEY)
    count = 10
    system_prompt = "You are an Academic Curriculum Architect. Identify core concepts and search queries to build a comprehensive textbook."
    user_prompt = f"Generate {count} specific Wikipedia search keywords/queries (approx 50% en, 50% vi) for the overarching topic: \"{topic}\"."
    
    print(f"Testing Topic: {topic}")
    try:
        resp = client.chat.completions.create(
            model=CauHinh.WRITER_MODEL, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" } # Simple test
        )
        print("Response received:")
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_identify_titles("Địa lí")
