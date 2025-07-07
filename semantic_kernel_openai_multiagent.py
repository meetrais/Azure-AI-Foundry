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

class BaseAgent:
    """Base class for all agents."""
    def __init__(self, openai_client, model_name, agent_name):
        self.openai_client = openai_client
        self.model_name = model_name
        self.agent_name = agent_name
        self.conversation_history = []
    
    def add_to_history(self, role, message):
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": message})
    
    async def get_ai_response(self, user_input, system_prompt):
        """Get AI response based on conversation context."""
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I'm having trouble processing your request. Error: {str(e)}"

class SickReportAgent(BaseAgent):
    """Independent agent for handling sick reports."""
    
    def __init__(self, openai_client, model_name):
        super().__init__(openai_client, model_name, "Specialized Agent")
        self.system_prompt = """You are a specialized agent for sick leave requests. Your role is to help users report their illness.

Your workflow:
1. First, ask for the "From Date" for their sick leave
2. Then, ask for the "To Date" for their sick leave  
3. Once you have both dates, say "SICK_REPORT_COMPLETE" and provide a summary

Be compassionate and professional. Only ask for the two dates - From Date and To Date.
If the user provides both dates in one message, acknowledge them and complete the report.
Always end with "SICK_REPORT_COMPLETE" when you have the required information."""

class FatigueReportAgent(BaseAgent):
    """Independent agent for handling fatigue reports."""
    
    def __init__(self, openai_client, model_name):
        super().__init__(openai_client, model_name, "Specialized Agent")
        self.system_prompt = """You are a specialized agent for fatigue reporting. Your role is to help users report fatigue.

Your workflow:
1. Immediately acknowledge their fatigue report
2. Say "FATIGUE_REPORT_COMPLETE" and confirm the report is logged

Be understanding and supportive. No additional questions needed - just acknowledge and complete.
Always end with "FATIGUE_REPORT_COMPLETE" in your first response."""

class HotelBookingAgent(BaseAgent):
    """Independent agent for handling hotel bookings."""
    
    def __init__(self, openai_client, model_name):
        super().__init__(openai_client, model_name, "Specialized Agent")
        self.system_prompt = """You are a specialized agent for hotel bookings. Your role is to help users book hotels.

Your workflow:
1. First, ask for the "From Date" for their hotel stay (check-in)
2. Then, ask for the "To Date" for their hotel stay (check-out)
3. Once you have both dates, say "HOTEL_BOOKING_COMPLETE" and provide a summary

Be helpful and professional. Only ask for the two dates - From Date and To Date.
If the user provides both dates in one message, acknowledge them and complete the booking.
Always end with "HOTEL_BOOKING_COMPLETE" when you have the required information."""

class LimoBookingAgent(BaseAgent):
    """Independent agent for handling limo bookings."""
    
    def __init__(self, openai_client, model_name):
        super().__init__(openai_client, model_name, "Specialized Agent")
        self.system_prompt = """You are a specialized agent for transportation bookings. Your role is to help users book transportation.

Your workflow:
1. Immediately acknowledge their transportation request
2. Say "LIMO_BOOKING_COMPLETE" and confirm the booking is logged

Be professional and efficient. No additional questions needed - just acknowledge and complete.
Always end with "LIMO_BOOKING_COMPLETE" in your first response."""

class MainAgent:
    """Main categorization agent that routes to sub-agents."""
    
    def __init__(self, openai_client, model_name):
        self.openai_client = openai_client
        self.model_name = model_name
        
        # Initialize sub-agents
        self.agents = {
            "Report Sick": SickReportAgent(openai_client, model_name),
            "Report Fatigue": FatigueReportAgent(openai_client, model_name),
            "Book Hotel": HotelBookingAgent(openai_client, model_name),
            "Book Limo": LimoBookingAgent(openai_client, model_name)
        }
    
    def categorize_with_direct_openai(self, user_input):
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
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
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
    
    # Use Azure AI Inference connector for function calling
    chat_service = AzureAIInferenceChatCompletion(
        ai_model_id=model_name,
        endpoint=base_endpoint,
        api_key=api_key,
    )
    kernel.add_service(chat_service)
    kernel.add_plugin(ToolPlugin(), "ToolPlugin")

    return kernel, chat_service, openai_client

async def handle_agent_interaction(agent, user_input):
    """Handle interaction with specialized agent."""
    
    # Add user input to agent's history
    agent.add_to_history("user", user_input)
    
    # Get AI response from the agent
    if hasattr(agent, 'system_prompt'):
        response = await agent.get_ai_response(user_input, agent.system_prompt)
    else:
        response = "Agent is processing your request..."
    
    # Add agent response to history
    agent.add_to_history("assistant", response)
    
    # Check for completion signals
    completion_signals = ["SICK_REPORT_COMPLETE", "FATIGUE_REPORT_COMPLETE", "HOTEL_BOOKING_COMPLETE", "LIMO_BOOKING_COMPLETE"]
    
    is_complete = any(signal in response for signal in completion_signals)
    
    return response, is_complete

def main():
    st.set_page_config(
        page_title="Multi-Agent Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Multi-Agent Request Assistant")
    st.markdown("I use specialized AI agents to handle your requests for reporting sickness, reporting fatigue, booking hotels, or booking transportation.")

    # Initialize chatbot
    kernel, chat_service, openai_client = initialize_chatbot()
    
    if kernel is None:
        st.stop()

    # Initialize main agent
    if "main_agent" not in st.session_state:
        st.session_state.main_agent = MainAgent(openai_client, os.getenv("DEPLOYMENT_NAME"))

    # Initialize session state for chat history and agent states
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.waiting_for_confirmation = False
        st.session_state.pending_category = None
        st.session_state.agent_active = False
        st.session_state.current_agent = None

    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Handle different states
    if st.session_state.agent_active and st.session_state.current_agent:
        # Agent is handling the conversation
        st.markdown("---")
        st.markdown(f"🤖 **{st.session_state.current_agent.agent_name}** is assisting you")
        
        # Agent conversation input
        if agent_input := st.chat_input("Continue conversation with the specialized agent..."):
            # Add user message to main chat
            st.session_state.messages.append({"role": "user", "content": agent_input})
            
            # Process with agent
            with st.spinner("Agent is processing..."):
                response, is_complete = asyncio.run(handle_agent_interaction(st.session_state.current_agent, agent_input))
            
            # Add agent response to main chat
            st.session_state.messages.append({"role": "assistant", "content": f"**Specialized Agent:** {response}"})
            
            # Check if agent completed the task
            if is_complete:
                st.session_state.messages.append({"role": "assistant", "content": "🔄 **Returning to Main Agent**"})
                st.session_state.agent_active = False
                st.session_state.current_agent = None
                st.session_state.waiting_for_confirmation = False
                st.session_state.pending_category = None
            
            st.rerun()
        
        # Add return button
        if st.button("🔄 Return to Main Agent"):
            st.session_state.messages.append({"role": "assistant", "content": "🔄 **Returned to Main Agent**"})
            st.session_state.agent_active = False
            st.session_state.current_agent = None
            st.session_state.waiting_for_confirmation = False
            st.session_state.pending_category = None
            st.rerun()
            
    elif st.session_state.waiting_for_confirmation:
        # Show confirmation buttons
        st.markdown("### Confirmation Required")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Connect with Specialized Agent", use_container_width=True):
                # Connect to specialized agent
                st.session_state.messages.append({"role": "user", "content": "Yes"})
                
                # Get the appropriate agent
                if st.session_state.pending_category in st.session_state.main_agent.agents:
                    st.session_state.current_agent = st.session_state.main_agent.agents[st.session_state.pending_category]
                    st.session_state.agent_active = True
                    
                    # Initial agent greeting
                    category_emojis = {
                        "Report Sick": "🏥",
                        "Report Fatigue": "😴", 
                        "Book Hotel": "🏨",
                        "Book Limo": "🚗"
                    }
                    emoji = category_emojis.get(st.session_state.pending_category, "🤖")
                    
                    # Get initial agent response
                    with st.spinner("Connecting to specialized agent..."):
                        initial_response, _ = asyncio.run(handle_agent_interaction(
                            st.session_state.current_agent, 
                            f"I need help with {st.session_state.pending_category}"
                        ))
                    
                    greeting = f"{emoji} **Specialized Agent:** {initial_response}"
                    st.session_state.messages.append({"role": "assistant", "content": greeting})
                
                st.rerun()
        
        with col2:
            if st.button("❌ No, cancel request", use_container_width=True):
                # Cancel the request
                st.session_state.messages.append({"role": "user", "content": "No"})
                st.session_state.messages.append({"role": "assistant", "content": "No problem! Feel free to ask about something else."})
                
                # Reset confirmation state
                st.session_state.waiting_for_confirmation = False
                st.session_state.pending_category = None
                
                st.rerun()
    
    else:
        # Regular chat input
        if prompt := st.chat_input("What can I help you with today?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Show analyzing message
            with st.spinner("🔍 Analyzing your request..."):
                # Categorize using main agent
                category = st.session_state.main_agent.categorize_with_direct_openai(prompt)
            
            if category:
                # Show identified category
                category_message = f"🎯 I've identified your request as: **{category}**\n\nWould you like me to connect you with our Specialized Agent?"
                st.session_state.messages.append({"role": "assistant", "content": category_message})
                
                # Set up confirmation state
                st.session_state.waiting_for_confirmation = True
                st.session_state.pending_category = category
                
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
        - *Agent will ask for dates*
        
        😴 **Report Fatigue** 
        - Tiredness, exhaustion, being worn out
        - *No additional info required*
        
        🏨 **Book Hotel**
        - Hotel bookings, accommodation, rooms
        - *Agent will ask for dates*
        
        🚗 **Book Limo**
        - Transportation, rides, car services
        - *No additional info required*
        """)
        
        st.header("💡 Example Requests")
        st.markdown("""
        - "I have a headache"
        - "Need somewhere to stay tonight"
        - "Feeling really drained"
        - "Can you get me a ride?"
        """)
        
        st.header("🤖 Multi-Agent Features")
        st.markdown("""
        - **AI-driven categorization**
        - **Specialized agents** for each request type
        - **Natural conversation** with agents
        - **LLM-powered workflows**
        """)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.waiting_for_confirmation = False
            st.session_state.pending_category = None
            st.session_state.agent_active = False
            st.session_state.current_agent = None
            # Reinitialize main agent to clear agent histories
            st.session_state.main_agent = MainAgent(openai_client, os.getenv("DEPLOYMENT_NAME"))
            st.rerun()

if __name__ == "__main__":
    main()