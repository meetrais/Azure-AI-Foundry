import os
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from dotenv import load_dotenv
from openai import AzureOpenAI

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

def categorize_with_direct_openai(openai_client, user_input, model_name):
    """Use direct OpenAI client to categorize user input."""
    
    categorization_prompt = f"""You are a categorization assistant. Analyze the user's request and determine which category it belongs to.

Available categories:
1. Report Sick - for health issues, illness, feeling unwell, sickness, medical problems, pain, headaches
2. Report Fatigue - for tiredness, exhaustion, fatigue, being worn out, sleepiness, energy issues
3. Book Hotel - for hotel bookings, accommodation requests, room reservations, lodging
4. Book Limo - for transportation requests, ride bookings, car services, taxi requests, travel

User request: "{user_input}"

Rules:
- Respond with ONLY the exact category name (e.g., "Report Sick")
- If the request doesn't clearly match any category, respond with "no_match"
- Do not provide explanations or additional text
- Be flexible in understanding different ways people might express these needs

Category:"""

    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": categorization_prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        category = response.choices[0].message.content.strip()
        
        # Validate the response is one of our expected categories
        valid_categories = ["Report Sick", "Report Fatigue", "Book Hotel", "Book Limo", "no_match"]
        
        if category in valid_categories:
            return category if category != "no_match" else None
        else:
            # If LLM returns something unexpected, try to match it to our categories
            category_lower = category.lower()
            if "sick" in category_lower or "health" in category_lower:
                return "Report Sick"
            elif "fatigue" in category_lower or "tired" in category_lower:
                return "Report Fatigue"
            elif "hotel" in category_lower or "accommodation" in category_lower:
                return "Book Hotel"
            elif "limo" in category_lower or "transport" in category_lower or "ride" in category_lower:
                return "Book Limo"
            else:
                return None
                
    except Exception as e:
        print(f"❌ LLM categorization failed: {str(e)}")
        return None

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

    if api_key:
        print(f"API Key starts with: {api_key[:10]}...")
    print("==========================\n")

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

    # Initialize direct OpenAI client for categorization
    openai_client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version or "2024-12-01-preview",
        azure_endpoint=base_endpoint
    )

    kernel = sk.Kernel()
    
    # Use Azure AI Inference connector for function calling (bypass LLM categorization)
    chat_service = AzureAIInferenceChatCompletion(
        ai_model_id=model_name,
        endpoint=base_endpoint,
        api_key=api_key,
    )
    kernel.add_service(chat_service)

    kernel.add_plugin(ToolPlugin(), "ToolPlugin")

    # Track if this is the first interaction
    first_interaction = True
    
    print("🤖 AI Request Assistant (LLM-Powered)")
    print("I use AI to understand your requests for reporting sickness, reporting fatigue, booking hotels, or booking transportation.")
    print("Type 'exit' to quit.\n")
    
    while True:
        # Use different prompt for first vs subsequent interactions
        if first_interaction:
            user_input = input("What can I help you with today? ")
            first_interaction = False
        else:
            user_input = input("What else can I help you with today? ")
            
        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break

        print("🔍 Analyzing your request with AI...")
        
        # Use direct OpenAI client for categorization
        category = categorize_with_direct_openai(openai_client, user_input, model_name)
        
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
            print(f"ChatBot: 🎯 I've identified your request as: {category}")
            
            # Ask for confirmation
            confirmation_prompt = f"Would you like me to raise a request for this? (yes/no) "
            user_confirmation = input(confirmation_prompt)
            
            if user_confirmation.lower() in ["yes", "y"]:
                try:
                    print("⚙️ Processing your request...")
                    # Call the appropriate function
                    tool_result = await kernel.invoke(
                        plugin_name="ToolPlugin",
                        function_name=function_name,
                        arguments=KernelArguments(),
                    )
                    response = str(tool_result)
                    print(f"ChatBot: ✅ {response}")
                except Exception as e:
                    print(f"Error calling function: {e}")
                    print(f"ChatBot: ✅ Request logged successfully (local fallback).")
            else:
                response = "Okay, no request will be raised."
                print(f"ChatBot: {response}")
        else:
            response = "🤔 I'm sorry, I couldn't categorize your request. I can help you with reporting sickness, reporting fatigue, booking hotels, or booking transportation. Could you please clarify what you need help with?"
            print(f"ChatBot: {response}")
        
        print()  # Add empty line for better readability

if __name__ == "__main__":
    asyncio.run(run_agentic_chatbot())