import google.generativeai as genai

genai.configure(api_key="AIzaSyCQvjTEg8ksNuejD-OGPL8yFXqQFFqt70U")

print("🔍 Listing available models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"   - {m.name}")
