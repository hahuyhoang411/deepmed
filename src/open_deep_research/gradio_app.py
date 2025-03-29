import gradio as gr
import os
import uuid
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder
from IPython.display import Markdown
from langgraph.types import Command
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith configuration
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# Report structure to be used in the demo
REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""

# Initialize the graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Store thread state between interactions
thread_state = {}

async def generate_report(topic, tavily_api_key, google_api_key, feedback=None, thread_id=None):
    # Set API keys
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Configure the thread
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        
    thread = {
        "configurable": {
            "thread_id": thread_id,
            "search_api": "tavily",
            "planner_provider": "google_genai",
            "planner_model": "gemini-2.0-flash",
            "writer_provider": "google_genai", 
            "writer_model": "gemini-2.0-flash",
            "max_search_depth": 1,
            "report_structure": REPORT_STRUCTURE,
        }
    }
    
    result = {}
    
    if feedback is None:
        # First run - get the plan
        async for event in graph.astream({"topic": topic}, thread, stream_mode="updates"):
            if "__interrupt__" in event:
                result = {
                    "status": "plan_ready", 
                    "plan": event['__interrupt__'][0].value,
                    "thread_id": thread_id
                }
                break
    elif feedback is True:
        # User approved - resume execution with True
        async for event in graph.astream(Command(resume=True), thread, stream_mode="values"):
            if "final_report" in event:
                result = {
                    "status": "report_ready", 
                    "report": event["final_report"],
                    "thread_id": thread_id
                }
                break
    else:
        # User provided feedback - resume with feedback string
        async for event in graph.astream(Command(resume=feedback), thread, stream_mode="updates"):
            if "__interrupt__" in event:
                result = {
                    "status": "plan_ready", 
                    "plan": event['__interrupt__'][0].value,
                    "thread_id": thread_id
                }
                break
    
    return result

def run_report_generation(topic, tavily_api_key, google_api_key, feedback=None, thread_id=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            generate_report(topic, tavily_api_key, google_api_key, feedback, thread_id)
        )
        return result
    finally:
        loop.close()

with gr.Blocks() as demo:
    gr.Markdown("# Open Deep Research Assistant")
    
    # Hidden state for thread ID
    thread_id = gr.State(None)
    
    with gr.Row():
        with gr.Column():
            topic_input = gr.Textbox(label="Research Topic", placeholder="Enter a research topic...")
            tavily_key = gr.Textbox(label="Tavily API Key", type="password")
            google_key = gr.Textbox(label="Google API Key", type="password")
            
            with gr.Row():
                submit_btn = gr.Button("Generate Plan", variant="primary")
                approve_btn = gr.Button("Approve Plan", visible=False, variant="success")
                revise_btn = gr.Button("Revise Plan", visible=False, variant="secondary")
            
            feedback_input = gr.Textbox(label="Feedback for Plan Revision", visible=False, 
                                        placeholder="Provide specific feedback to improve the plan...")
            submit_feedback_btn = gr.Button("Submit Feedback", visible=False)
            
        with gr.Column():
            status_indicator = gr.Markdown("Ready to generate a research plan")
            plan_output = gr.Textbox(label="Generated Plan", visible=False, lines=10)
            progress_indicator = gr.Markdown(visible=False)
            output = gr.Markdown(label="Generated Report", visible=False)
    
    def handle_submit(topic, tavily_key, google_key):
        if not topic or not tavily_key or not google_key:
            return {
                status_indicator: gr.Markdown(value="⚠️ Please fill in all required fields"),
                plan_output: gr.Textbox(visible=False),
                approve_btn: gr.Button(visible=False),
                revise_btn: gr.Button(visible=False),
                feedback_input: gr.Textbox(visible=False),
                submit_feedback_btn: gr.Button(visible=False),
                output: gr.Markdown(visible=False)
            }
            
        status_indicator.value = "Generating research plan..."
        result = run_report_generation(topic, tavily_key, google_key)
        
        if result["status"] == "plan_ready":
            return {
                status_indicator: gr.Markdown(value="✅ Plan generated! Review and approve or revise."),
                plan_output: gr.Textbox(value=result["plan"], visible=True),
                approve_btn: gr.Button(visible=True),
                revise_btn: gr.Button(visible=True),
                thread_id: result["thread_id"],
                output: gr.Markdown(visible=False)
            }
    
    def handle_approve(topic, tavily_key, google_key, current_thread_id):
        status_indicator.value = "Generating report based on approved plan..."
        progress_indicator.value = "Starting research and writing process..."
        
        result = run_report_generation(topic, tavily_key, google_key, feedback=True, thread_id=current_thread_id)
        
        return {
            status_indicator: gr.Markdown(value="✅ Report generation complete!"),
            plan_output: gr.Textbox(visible=False),
            approve_btn: gr.Button(visible=False),
            revise_btn: gr.Button(visible=False),
            progress_indicator: gr.Markdown(visible=False),
            output: gr.Markdown(value=result["report"], visible=True)
        }
    
    def handle_revise_click():
        return {
            feedback_input: gr.Textbox(visible=True),
            submit_feedback_btn: gr.Button(visible=True),
            status_indicator: gr.Markdown(value="Please provide feedback to revise the plan")
        }
    
    def handle_feedback_submit(topic, tavily_key, google_key, feedback, current_thread_id):
        if not feedback:
            return {
                status_indicator: gr.Markdown(value="⚠️ Please provide feedback for revision")
            }
            
        status_indicator.value = "Revising plan based on feedback..."
        
        result = run_report_generation(topic, tavily_key, google_key, feedback=feedback, thread_id=current_thread_id)
        
        if result["status"] == "plan_ready":
            return {
                status_indicator: gr.Markdown(value="✅ Plan revised! Review and approve or revise again."),
                plan_output: gr.Textbox(value=result["plan"], visible=True),
                feedback_input: gr.Textbox(value="", visible=False),
                submit_feedback_btn: gr.Button(visible=False),
                thread_id: result["thread_id"]
            }
    
    submit_btn.click(
        fn=handle_submit,
        inputs=[topic_input, tavily_key, google_key],
        outputs=[status_indicator, plan_output, approve_btn, revise_btn, thread_id, output]
    )
    
    approve_btn.click(
        fn=handle_approve, 
        inputs=[topic_input, tavily_key, google_key, thread_id],
        outputs=[status_indicator, plan_output, approve_btn, revise_btn, progress_indicator, output]
    )
    
    revise_btn.click(
        fn=handle_revise_click,
        inputs=[],
        outputs=[feedback_input, submit_feedback_btn, status_indicator]
    )
    
    submit_feedback_btn.click(
        fn=handle_feedback_submit,
        inputs=[topic_input, tavily_key, google_key, feedback_input, thread_id],
        outputs=[status_indicator, plan_output, feedback_input, submit_feedback_btn, thread_id]
    )

if __name__ == "__main__":
    demo.launch()