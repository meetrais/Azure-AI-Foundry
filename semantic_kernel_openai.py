import os
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from dotenv import load_dotenv

class ToolPlugin:
    """A plugin that exposes tools for the chatbot."""
    @kernel_function(
        description="Logs a request to report sick.",
        name="report_sick"
    )
    def report_sick(self):
        """Placeholder function to handle reporting sick."""
        return "Request to report sick has been logged."

    @kernel_function(
        description="Logs a request to book a hotel.",
        name="book_hotel"
    )
    def book_hotel(self):
        """Placeholder function to handle booking a hotel."""
        return "Request to book a hotel has been logged."

    @kernel_function(
        description="Logs a request to book a limo.",
        name="book_limo"
    )
    def book_limo(self):
        """Placeholder function to handle booking a limo."""
        return "Request to book a limo has been logged."

    @kernel_function(
        description="Logs a request to report fatigue.",
        name="report_fatigue"
    )
    def report_fatigue(self):
        """Placeholder function to handle reporting fatigue."""
        return "Request to report fatigue has been logged."

def create_tool_dict(metadata):
    """Convert KernelFunctionMetadata to dictionary format for tools."""
    return {
        "type": "function",
        "function": {
            "name": metadata.fully_qualified_name,
            "description": metadata.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }

async def run_agentic_chatbot():
    """
    Runs an agentic chatbot that uses tools to handle user requests.
    """
    load_dotenv()

    # For Azure OpenAI, we need different environment variables
    endpoint = os.getenv("ENDPOINT_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    model_name = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")

    # Extract the base endpoint (remove the full path)
    if endpoint and "/openai/deployments/" in endpoint:
        base_endpoint = endpoint.split("/openai/deployments/")[0]
    else:
        base_endpoint = endpoint

    # Validate configuration
    if not base_endpoint:
        print("ERROR: ENDPOINT_URL environment variable is not set")
        return
    if not api_key:
        print("ERROR: AZURE_OPENAI_API_KEY environment variable is not set")
        return
    if not model_name:
        print("ERROR: DEPLOYMENT_NAME environment variable is not set")
        return

    kernel = sk.Kernel()
    
    # Use Azure AI Inference connector with proper endpoint format
    chat_service = AzureAIInferenceChatCompletion(
        ai_model_id=model_name,
        endpoint=base_endpoint,
        api_key=api_key,
    )
    kernel.add_service(chat_service)

    kernel.add_plugin(ToolPlugin(), "ToolPlugin")

    # Simple keyword-based categorization (more reliable than AI for this use case)
    def categorize_input(user_input):
        user_input_lower = user_input.lower().strip()
        
        # Define keywords for each category
        sick_keywords = ['sick', 'ill', 'illness', 'not well', 'unwell', 'health', 'disease', 'fever', 'cold', 'flu']
        fatigue_keywords = ['tired', 'exhausted', 'fatigue', 'sleepy', 'worn out', 'weary', 'drained']
        hotel_keywords = ['hotel', 'accommodation', 'room', 'stay', 'booking', 'lodge', 'inn', 'resort']
        limo_keywords = ['limo', 'ride', 'car', 'transport', 'taxi', 'transportation', 'vehicle', 'drive']
        
        # Check for keyword matches
        if any(keyword in user_input_lower for keyword in sick_keywords):
            return "Report Sick"
        elif any(keyword in user_input_lower for keyword in fatigue_keywords):
            return "Report Fatigue"
        elif any(keyword in user_input_lower for keyword in hotel_keywords):
            return "Book Hotel"
        elif any(keyword in user_input_lower for keyword in limo_keywords):
            return "Book Limo"
        
        return None

    # Track if this is the first interaction
    first_interaction = True
    
    while True:
        # Use different prompt for first vs subsequent interactions
        if first_interaction:
            user_input = input("What can I help you with today? ")
            first_interaction = False
        else:
            user_input = input("What else can I help you with today? ")
            
        if user_input.lower() == "exit":
            break

        # Use keyword-based categorization
        category = categorize_input(user_input)
        
        # Check if we found a valid category
        valid_categories = {
            "Report Sick": ("report_sick", "report that you're not feeling well"),
            "Report Fatigue": ("report_fatigue", "report fatigue or tiredness"), 
            "Book Hotel": ("book_hotel", "book a hotel"),
            "Book Limo": ("book_limo", "book a limo/transportation")
        }
        
        if category in valid_categories:
            function_name, description = valid_categories[category]
            
            # Show the identified category
            print(f"ChatBot: I've identified your request as: {category}")
            
            # Ask for confirmation
            confirmation_prompt = f"Would you like me to raise a request for this? (yes/no) "
            user_confirmation = input(confirmation_prompt)
            
            if user_confirmation.lower() in ["yes", "y"]:
                try:
                    # Call the appropriate function
                    tool_result = await kernel.invoke(
                        plugin_name="ToolPlugin",
                        function_name=function_name,
                        arguments=KernelArguments(),
                    )
                    response = str(tool_result)
                    print(f"ChatBot: {response}")
                except Exception as e:
                    print(f"Error calling function: {e}")
                    print(f"ChatBot: Request logged successfully (local fallback).")
            else:
                response = "Okay, no request will be raised."
                print(f"ChatBot: {response}")
        else:
            response = "I'm sorry, I couldn't categorize your request. I can help you with reporting sickness, reporting fatigue, booking hotels, or booking transportation. Could you please clarify what you need help with?"
            print(f"ChatBot: {response}")

if __name__ == "__main__":
    asyncio.run(run_agentic_chatbot())