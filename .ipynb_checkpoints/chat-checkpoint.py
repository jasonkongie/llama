from llama_cpp import Llama

def main():
    # Initialize the Llama model with the specified path to the .gguf model file
    model_path = "./TinyLlama-1.1B-Chat-v1.0/TinyLlama.gguf"
    llm = Llama(model_path=model_path)

    print("Welcome to the TinyLlama Chatbot. Type 'quit' to exit.")

    while True:
        question = input("You: ")
        print(question)
        if question.lower() == 'quit':
            break

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs in JSON.",
                },
                {"role": "user", "content": f"{question}"},
            ],
            response_format={
                "type": "json_object",
            },
            temperature=0.7,
        )

        print(response) 

if __name__ == "__main__":
    main()
