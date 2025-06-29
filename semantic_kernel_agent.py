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

    endpoint = os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT")
    api_key = os.getenv("API_KEY")
    model_name = "Ministral-3B"

    kernel = sk.Kernel()
    
    chat_service = AzureAIInferenceChatCompletion(
        ai_model_id=model_name,
        endpoint=endpoint,
        api_key=api_key,
    )
    kernel.add_service(chat_service)

    kernel.add_plugin(ToolPlugin(), "ToolPlugin")

    system_message = (
        "You are a categorization assistant. You must respond with ONLY the exact category name from the list below.\n\n"
        "Categories:\n"
        "- Report Sick (for: not well, sick, illness, health issues, unwell)\n"
        "- Report Fatigue (for: tired, exhausted, fatigue, sleepy, worn out)\n"
        "- Book Hotel (for: hotel, accommodation, room, stay, booking)\n"
        "- Book Limo (for: limo, ride, car, transport, taxi, transportation)\n\n"
        "Examples:\n"
        "Input: 'not well' → Output: 'Report Sick'\n"
        "Input: 'hotel' → Output: 'Book Hotel'\n"
        "Input: 'tired' → Output: 'Report Fatigue'\n"
        "Input: 'ride' → Output: 'Book Limo'\n"
        "Input: 'weather' → Output: 'no_match'\n\n"
        "Rules:\n"
        "1. Respond with ONLY the category name or 'no_match'\n"
        "2. Do not add explanations, punctuation, or extra text\n"
        "3. Match keywords to the closest category\n"
        "4. If unsure, respond with 'no_match'"
    )
    chat_history = ChatHistory(system_message=system_message)
    
    # Get function metadata and convert to proper format
    function_metadata_list = kernel.get_full_list_of_function_metadata()
    tools = [create_tool_dict(metadata) for metadata in function_metadata_list]
    
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

        chat_history.add_user_message(user_input)

        # Get the category identification from the AI (without tools first)
        execution_settings = PromptExecutionSettings(
            tool_choice="none",  # Don't use tools for categorization
            temperature=0.0,
        )
        
        try:
            result = (
                await chat_service.get_chat_message_contents(
                    chat_history,
                    settings=execution_settings,
                )
            )[0]

            category = str(result.content).strip()
            
            # Check if the AI identified a valid category
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
                    # Call the appropriate function
                    tool_result = await kernel.invoke(
                        plugin_name="ToolPlugin",
                        function_name=function_name,
                        arguments=KernelArguments(),
                    )
                    response = str(tool_result)
                    print(f"ChatBot: {response}")
                    chat_history.add_assistant_message(response)
                else:
                    response = "Okay, no request will be raised."
                    print(f"ChatBot: {response}")
                    chat_history.add_assistant_message(response)
            else:
                response = "I'm sorry, I couldn't categorize your request. I can help you with reporting sickness, reporting fatigue, booking hotels, or booking transportation. Could you please clarify what you need help with?"
                print(f"ChatBot: {response}")
                chat_history.add_assistant_message(response)

        except Exception as e:
            print(f"Error: {e}")
            response = "I'm sorry, I encountered an error processing your request."
            print(f"ChatBot: {response}")
            chat_history.add_assistant_message(response)

if __name__ == "__main__":
    asyncio.run(run_agentic_chatbot())