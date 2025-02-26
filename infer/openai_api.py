import requests
from openai import OpenAI


def get_api_key():
    # Load the OpenAI API key from a file
    with open("../../keys.txt", "r") as file:
        for line in file:
            if line.startswith("OpenAI: "):
                return line.split("OpenAI: ")[1].strip()
    return None


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


def ask_gpt4o(api_key, weather_info):
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Prepare the prompt
    prompt = f"The current weather in Abu Dhabi is: {weather_info}. Provide a brief description of this weather condition."
    
    # Create a chat completion
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    
    # Extract and return the assistant's reply
    return response.choices[0].message['content']


def main():
    openai_api_key = get_api_key()
    
    weather_info = fetch_weather()
    if weather_info:
        gpt_response = ask_gpt4o(openai_api_key, weather_info)
        print(gpt_response)
    else:
        print("Weather data is not available at the moment.")

if __name__ == "__main__":
    main()
