from mistralai import Mistral

API_KEY = 'ICFcAmqJuKgEpH0f7ZjNJdCvuda9TV90'



if __name__ == "__main__":

    # Initialize Mistral client
    client = Mistral(api_key=API_KEY)

    # Define the model to use
    model = "mistral-large-latest"

    # Create a chat completion request
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How can you assist me today?"},
        ],
        stream=False  # Set to True for streaming responses
    )

    # Print the assistant's reply
    print(response.choices[0].message.content)