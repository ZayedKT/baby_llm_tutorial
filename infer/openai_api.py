import requests
import os
from openai import OpenAI



def fetch_weather():
    try:
        # Fetch weather information for Abu Dhabi
        response = requests.get('https://wttr.in/Abu+Dhabi?format=3')
        if response.status_code == 200:
            return response.text.strip()
        else:
            print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def ask_gpt(client, today_weather=None):
    # Prepare the prompt
    if today_weather:
        prompt = f"The weather in Abu Dhabi today is: {today_weather}. Provide a brief description of this weather condition."
    else:
        prompt = "What is the weather like in Abu Dhabi, in general?"
    
    # Create a chat completion
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=256
    )
    
    # Extract and return the assistant's reply
    return response.choices[0].message.content



def main():
    ### Step 0: Retrieve the api key and initialize the client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    ### Step 1: Ask about the weather in general, simply querying GPT
    gpt_response = ask_gpt(client)
    print(f"General weather response:\n{gpt_response}\n\n")

    ### Step 2: Ask about the weather TODAY, using an external API call
    weather_info = fetch_weather()
    if weather_info:
        gpt_response = ask_gpt(client, weather_info)
        print(gpt_response)
    else:
        print("Weather data is not available at the moment.")


if __name__ == "__main__":
    main()
