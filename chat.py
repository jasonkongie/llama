from llama_cpp import Llama


class Chat:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def run(self):
        while True:
            print("enter quit to quit")
            my_prompt = input("You: ")

            if (my_prompt.lower() == "quit"):
                break
            
            response = self.generate_text_from_prompt(my_prompt)
            final_result = response["choices"][0]["text"].strip()
            print(final_result)

        print("finished running..")
                

    
    # LOAD THE MODEL
    def load_model(self, model_path):    
        model = Llama(model_path=model_path,
                            n_ctx=512)
        return model
    
    # GENERATE TEXT FROM USER
    def generate_text_from_prompt(self, 
                                 user_prompt,
                                 max_tokens = 100,
                                 temperature = 0.3,
                                 top_p = 0.1,
                                 echo = True,
                                 stop = ["Q", "\n"]):
    
       model_output = self.model(
           user_prompt,
           max_tokens=max_tokens,
           temperature=temperature,
           top_p=top_p,
           echo=echo,
           stop=stop,
       )
    
    
       return model_output

def main():
    # Initialize the Llama model with the specified path to the .gguf model file
    model_path = "../model/model.gguf"
    chat = Chat(model_path)
    chat.run()

if __name__ == "__main__":
    main()
