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
        st.error(f"LLM categorization failed: {str(e)}")
        return None

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components."""
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

    # Show configuration in sidebar for debugging
    with st.sidebar:
        with st.expander("🔧 Configuration Debug", expanded=False):
            st.write(f"**Endpoint:** {base_endpoint}")
            st.write(f"**Model:** {model_name}")
            st.write(f"**API Version:** {api_version}")
            st.write(f"**API Key:** {'✅ Present' if api_key else '❌ Missing'}")

    # Validate configuration
    if not base_endpoint or not api_key or not model_name:
        st.error("Configuration error: Please check your environment variables.")
        return None, None, None

    # Initialize direct OpenAI client for categorization
    openai_client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version or "2024-12-01-preview",
        azure_endpoint=base_endpoint
    )

    # Initialize Semantic Kernel for function calling
    kernel = sk.Kernel()
    
    # Use Azure AI Inference connector for function calling (we'll bypass LLM categorization)
    chat_service = AzureAIInferenceChatCompletion(
        ai_model_id=model_name,
        endpoint=base_endpoint,
        api_key=api_key,
    )
    kernel.add_service(chat_service)
    kernel.add_plugin(ToolPlugin(), "ToolPlugin")

    return kernel, chat_service, openai_client

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

    st.title("🤖 AI Request Assistant (LLM-Powered)")
    st.markdown("I use AI to understand your requests for reporting sickness, reporting fatigue, booking hotels, or booking transportation.")

    # Initialize chatbot
    kernel, chat_service, openai_client = initialize_chatbot()
    
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
                        st.session_state.messages.append({"role": "assistant", "content": f"✅ {response}"})
                        
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
            
            # Show analyzing message
            with st.spinner("🔍 Analyzing your request with AI..."):
                # Get model name from environment
                model_name = os.getenv("DEPLOYMENT_NAME")
                # Categorize using direct OpenAI client
                category = categorize_with_direct_openai(openai_client, prompt, model_name)
            
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
                category_message = f"🎯 I've identified your request as: **{category}**\n\nWould you like me to raise a request for this?"
                st.session_state.messages.append({"role": "assistant", "content": category_message})
                
                # Set up confirmation state
                st.session_state.waiting_for_confirmation = True
                st.session_state.pending_category = category
                st.session_state.pending_function = function_name
                
            else:
                # Unable to categorize
                error_message = "🤔 I'm sorry, I couldn't categorize your request. I can help you with reporting sickness, reporting fatigue, booking hotels, or booking transportation. Could you please clarify what you need help with?"
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
        - "I have a headache"
        - "Need somewhere to stay tonight"
        - "Feeling really drained"
        - "Can you get me a ride?"
        """)
        
        st.header("🤖 AI Features")
        st.markdown("""
        - **Intelligent categorization** using direct OpenAI
        - **Natural language understanding**
        - **Flexible request interpretation**
        - **Reliable AI-powered analysis**
        """)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.waiting_for_confirmation = False
            st.session_state.pending_category = None
            st.session_state.pending_function = None
            st.rerun()

if __name__ == "__main__":
    main()