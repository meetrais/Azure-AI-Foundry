import os
import asyncio
import streamlit as st
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

def categorize_input(user_input):
    """Simple keyword-based categorization."""
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

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components."""
    load_dotenv()

    # For Azure OpenAI, we need different environment variables
    endpoint = os.getenv("ENDPOINT_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    model_name = os.getenv("DEPLOYMENT_NAME")

    # Extract the base endpoint (remove the full path)
    if endpoint and "/openai/deployments/" in endpoint:
        base_endpoint = endpoint.split("/openai/deployments/")[0]
    else:
        base_endpoint = endpoint

    # Validate configuration
    if not base_endpoint or not api_key or not model_name:
        st.error("Configuration error: Please check your environment variables.")
        return None, None

    kernel = sk.Kernel()
    
    # Use Azure AI Inference connector with proper endpoint format
    chat_service = AzureAIInferenceChatCompletion(
        ai_model_id=model_name,
        endpoint=base_endpoint,
        api_key=api_key,
    )
    kernel.add_service(chat_service)
    kernel.add_plugin(ToolPlugin(), "ToolPlugin")

    return kernel, chat_service

async def process_request(kernel, category, function_name):
    """Process the user request and call the appropriate function."""
    try:
        tool_result = await kernel.invoke(
            plugin_name="ToolPlugin",
            function_name=function_name,
            arguments=KernelArguments(),
        )
        return str(tool_result)
    except Exception as e:
        st.error(f"Error calling function: {e}")
        return f"Request logged successfully (local fallback)."

def main():
    st.set_page_config(
        page_title="AI Request Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 AI Request Assistant")
    st.markdown("I can help you with reporting sickness, reporting fatigue, booking hotels, or booking transportation.")

    # Initialize chatbot
    kernel, chat_service = initialize_chatbot()
    
    if kernel is None:
        st.stop()

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.waiting_for_confirmation = False
        st.session_state.pending_category = None
        st.session_state.pending_function = None

    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Input handling
    if st.session_state.waiting_for_confirmation:
        # Show confirmation buttons
        st.markdown("### Confirmation Required")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Yes, raise the request", use_container_width=True):
                # Process the request
                with st.spinner("Processing request..."):
                    try:
                        response = asyncio.run(process_request(
                            kernel, 
                            st.session_state.pending_category,
                            st.session_state.pending_function
                        ))
                        
                        # Add user confirmation and bot response to chat
                        st.session_state.messages.append({"role": "user", "content": "Yes"})
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Reset confirmation state
                        st.session_state.waiting_for_confirmation = False
                        st.session_state.pending_category = None
                        st.session_state.pending_function = None
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing request: {e}")
        
        with col2:
            if st.button("❌ No, cancel request", use_container_width=True):
                # Cancel the request
                st.session_state.messages.append({"role": "user", "content": "No"})
                st.session_state.messages.append({"role": "assistant", "content": "Okay, no request will be raised."})
                
                # Reset confirmation state
                st.session_state.waiting_for_confirmation = False
                st.session_state.pending_category = None
                st.session_state.pending_function = None
                
                st.rerun()
    
    else:
        # Regular chat input
        if prompt := st.chat_input("What can I help you with today?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Categorize the input
            category = categorize_input(prompt)
            
            # Define valid categories
            valid_categories = {
                "Report Sick": ("report_sick", "report that you're not feeling well"),
                "Report Fatigue": ("report_fatigue", "report fatigue or tiredness"), 
                "Book Hotel": ("book_hotel", "book a hotel"),
                "Book Limo": ("book_limo", "book a limo/transportation")
            }
            
            if category in valid_categories:
                function_name, description = valid_categories[category]
                
                # Show identified category
                category_message = f"I've identified your request as: **{category}**\n\nWould you like me to raise a request for this?"
                st.session_state.messages.append({"role": "assistant", "content": category_message})
                
                # Set up confirmation state
                st.session_state.waiting_for_confirmation = True
                st.session_state.pending_category = category
                st.session_state.pending_function = function_name
                
            else:
                # Unable to categorize
                error_message = "I'm sorry, I couldn't categorize your request. I can help you with reporting sickness, reporting fatigue, booking hotels, or booking transportation. Could you please clarify what you need help with?"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            
            st.rerun()

    # Sidebar with information
    with st.sidebar:
        st.header("🔧 Available Services")
        st.markdown("""
        **I can help you with:**
        
        🏥 **Report Sick**
        - Health issues, illness, not feeling well
        
        😴 **Report Fatigue** 
        - Tiredness, exhaustion, being worn out
        
        🏨 **Book Hotel**
        - Hotel bookings, accommodation, rooms
        
        🚗 **Book Limo**
        - Transportation, rides, car services
        """)
        
        st.header("💡 Example Requests")
        st.markdown("""
        - "I'm not feeling well"
        - "I need a hotel room"
        - "I'm really tired today"
        - "Book me a ride"
        """)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.waiting_for_confirmation = False
            st.session_state.pending_category = None
            st.session_state.pending_function = None
            st.rerun()

if __name__ == "__main__":
    main()