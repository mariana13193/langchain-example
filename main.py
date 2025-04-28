import json
import boto3
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

class NovaConversation:
    def __init__(self, profile_name='mariana-account', region_name='us-east-1', max_history=10):
        session = boto3.Session(profile_name=profile_name)
        self.client = session.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )

        # Initialize chat history
        self.chat_history = ChatMessageHistory()
        self.max_history = max_history
        
    def add_message(self, role: str, content: str):
        if role == "user":
            self.chat_history.add_user_message(content)
        else:
            self.chat_history.add_ai_message(content)
            
        # Trim history if it exceeds max_history
        if len(self.chat_history.messages) > self.max_history * 2:
            self.chat_history.messages = self.chat_history.messages[-self.max_history * 2:]
    
    def chat(self, user_input: str) -> str:
        self.add_message("user", user_input)
        
        # Convert LangChain messages to the format expected by Nova
        messages = []
        for msg in self.chat_history.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            messages.append({
                "role": role,
                "content": [{"text": msg.content}]
            })
        
        payload = {
            "messages": messages
        }
        
        try:
            response = self.client.invoke_model(
                modelId="us.amazon.nova-micro-v1:0",
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response['body'].read())
            assistant_message = response_body['output']['message']['content'][0]['text']
            self.add_message("assistant", assistant_message)
            
            return assistant_message
            
        except Exception as e:
            print(f"Error during model invocation: {str(e)}")
            # Remove the last user message if there's an error
            if self.chat_history.messages:
                self.chat_history.messages.pop()
            raise
    
    def clear_history(self):
        self.chat_history.clear()
            
    def print_history(self):
        messages = self.chat_history.messages
        
        if not messages:
            print("No conversation history available.")
            return
            
        print("\n=== Conversation History ===")
        for message in messages:
            if isinstance(message, HumanMessage):
                print(f"\nYou: {message.content}")
            else:
                print(f"\nAssistant: {message.content}")
        print("\n==========================")

def main():
    conversation = NovaConversation(max_history=10)
    
    print("Chat started. Type 'exit' to quit, 'clear' to clear history, or 'load' to view history.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            conversation.clear_history()
            print("Conversation history cleared.")
            continue
        elif user_input.lower() == 'load':
            conversation.print_history()
            continue
            
        try:
            response = conversation.chat(user_input)
            print("\nAssistant:", response)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
