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
REPORT_STRUCTURE = """You are an AI assistant tasked with generating a clinical report based on a user-provided topic or clinical case. Using information from reliable medical sources (e.g., peer-reviewed journals, official guidelines, reputable medical databases), generate a concise, evidence-based report. Adapt the content and structure based on the query type as outlined below, but feel free to add or modify subsections as necessary to comprehensively address the user's query. Use your judgment to determine the most appropriate structure and content based on the specific user input.

**Determine the Query Type:**
- If the user input describes a patient's presentation with symptoms, history, physical exam findings, and/or diagnostic test results, generate a **Clinical Case Report**.
- If the user input specifies a drug, disease, procedure, or other clinical topic, generate the corresponding **Drug Report**, **Disease Report**, or adapt the structure for other clinical topics as outlined below.

---

### Report Structures by Query Type

#### For **Clinical Case Reports**:
1. **Key Findings Summary**  
   - Research: No  
   - Provide a detailed outline of the patient's presentation, including symptoms, history, physical exam findings, and test results. Highlight critical details (e.g., red flags) that will guide the diagnostic process. Present this information in a table format for clarity.

2. **Terminology Spotlight**  
   - Research: No  
   - Define key medical terms or findings from the case (e.g., hematemesis, crepitus) to clarify their clinical significance, based on general medical knowledge.

3. **Additional Research Section**
   - Research: Yes
   - Additional sections as needed for more information.

4. **Initial Analysis**  
   - Research: No  
   - Analyze the key findings step-by-step to identify patterns or significant abnormalities. Note possible organ systems involved and any urgent concerns without assuming a diagnosis yet.

5. **Proposed Diagnosis**  
   - Research: No  
   - Suggest the most likely diagnosis based solely on the analysis of the provided data. Provide a brief rationale linking findings to the conclusion.

6. **Clinical Suggestions**  
   - Research: No  
   - Offer immediate next steps for management or further evaluation based on the proposed diagnosis and standard clinical reasoning (e.g., stabilization, diagnostics, consultation).

7. **Evidence-Based Review**  
   - Research: Yes  
   - Search reliable medical sources to confirm the proposed diagnosis and suggestions. Provide:  
     - A detailed explanation of the diagnosis (e.g., definition, etiology).  
     - Supporting evidence for diagnostic criteria or management (e.g., "Per UpToDate, 2023...").  
     - Any recent updates or guidelines (past 2-5 years).

8. **Differential Considerations**  
   - Research: Yes  
   - List other possible diagnoses, explain why they are less likely based on the case and researched evidence, and cite sources where applicable. Use table format for clarity. Make this as detailed as possible.

9. **References**  
   - List all cited sources with title, authors, publication year, and journal/source name (e.g., "Smith J, et al., Journal of Medicine, 2023").

---

#### For **Drug Reports**:
1. **Introduction**  
   - Provide a brief overview of the drug and the purpose of the report (2-3 sentences max).

2. **Main Body Sections**  
   - **Indications**: Approved and notable off-label uses.  
   - **Dosage**: Standard doses for adults and relevant populations.  
   - **Administration**: Route and key instructions.  
   - **Side Effects**: Common and severe effects, with prevalence if known.  
   - **Interactions**: Significant drug-drug or drug-food interactions.  
   - **Contraindications**: Absolute and relative, with explanations.  
   - Use clear, bolded subheadings for each section.  
   - Include evidence-based information with inline citations (e.g., "Smith et al., 2023").

3. **Special Considerations**  
   - Address use in special populations (e.g., pediatrics, geriatrics, pregnancy, renal impairment) if relevant.  
   - Note significant interactions or context-specific factors.

4. **Recent Updates**  
   - Summarize recent research findings or guideline changes (past 2-5 years), with source years (e.g., "Updated guidelines in 2023 recommend...").

5. **Clinical Takeaways**  
   - List concise, actionable key points in bullet points.  
   - Include a structural element (e.g., dosage table).

6. **References**  
   - List all cited sources with title, authors, publication year, and journal/source name.

---

#### For **Disease Reports**:
1. **Introduction**  
   - Provide a brief overview of the disease and the purpose of the report (2-3 sentences max).

2. **Main Body Sections**  
   - **Definition/Etiology**: Brief description and causes.  
   - **Diagnosis**: Key diagnostic criteria and tests.  
   - **Treatment Options**: First-line treatments and alternatives.  
   - **Prognosis**: Expected outcomes with and without treatment.  
   - Use clear, bolded subheadings for each section.  
   - Include evidence-based information with inline citations (e.g., "Smith et al., 2023").

3. **Special Considerations**  
   - Address management in special populations or context-specific factors.

4. **Recent Updates**  
   - Summarize recent research findings or guideline changes (past 2-5 years).

5. **Clinical Takeaways**  
   - List concise, actionable key points in bullet points.  
   - Include a structural element (e.g., treatment comparison list).

6. **References**  
   - List all cited sources with title, authors, publication year, and journal/source name.

---

#### For Other Clinical Topics (e.g., procedures, diagnostic tests):
- Adapt the structure to include relevant sections such as indications, methodology, risks, benefits, or limitations.  
- Follow the same principles of concise, evidence-based reporting with citations.

---

### Additional Instructions for All Reports:
- **Flexibility**: You can add or modify subsections as needed to comprehensively address the user's query. Each section should focus on a sub-topic relevant to the user-provided topic. Anticipate potential follow-up questions or areas of interest related to the user's query and incorporate relevant information accordingly.
- **Conciseness**: Write concisely, avoiding unnecessary elaboration, and use precise medical terminology. While being thorough, prioritize the most relevant and critical information to keep the report concise and actionable.
- **Audience**: Assume the reader is a healthcare professional; avoid basic explanations unless critical.
- **Critical Information**: Emphasize critical information (e.g., **contraindications**, *severe side effects*) using bold or italics.
- **Specific Data**: Include specific data (e.g., dosages in mg, survival rates in %) where applicable.
- **Ambiguous Queries**: If the query type is ambiguous, ask for clarification or provide a general overview.
- **Simulate Research**: Simulate real-time data retrieval by citing plausible sources and years up to your knowledge cutoff or beyond as a simulation.
- **Format**: Use Markdown format for clarity (no codeblock).
- **Insightfulness**: Leverage your medical knowledge and reasoning abilities to provide insights that go beyond a standard report, offering unique perspectives or connections that might be valuable to the user.

---

Generate the report following the appropriate structure and instructions based on the user's input. Think step by step, ensuring the report is tailored, comprehensive, and clinically relevant. Let's think step by step."""

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
            "max_search_depth": 2,
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

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 Open Medical Research Assistant")
    gr.Markdown("Generate comprehensive, evidence-based medical reports with AI assistance")
    
    # Hidden state for thread ID
    thread_id = gr.State(None)
    
    with gr.Row():
        # Sidebar for API keys and configuration
        with gr.Column(scale=1, min_width=300):
            with gr.Accordion("API Configuration", open=True):
                tavily_key = gr.Textbox(
                    label="Tavily API Key", 
                    type="password",
                    placeholder="Enter your Tavily API key",
                    info="Required for web search functionality"
                )
                google_key = gr.Textbox(
                    label="Google API Key", 
                    type="password",
                    placeholder="Enter your Google API key",
                    info="Required for Gemini model access"
                )
            
            with gr.Accordion("About", open=False):
                gr.Markdown("""
                ## How it works
                1. Enter your research topic
                2. Review the generated research plan
                3. Approve or provide feedback to revise
                4. Get a comprehensive research report
                
                This tool uses LangGraph to orchestrate a research workflow with web search and AI writing capabilities.
                """)
        
        # Main content area
        with gr.Column(scale=3):
            # Topic input section - replacing Box with a Group
            with gr.Group():
                gr.Markdown("### 📝 Research Topic")
                topic_input = gr.Textbox(
                    placeholder="Enter a medical or clinical research topic (e.g., 'Metformin for Type 2 Diabetes')",
                    label="",
                    lines=2
                )
                submit_btn = gr.Button("Generate Research Plan", variant="primary", size="lg")
                # Add a loading indicator that appears when generating the plan
                plan_loading = gr.Markdown(visible=False, value="🔄 Generating research plan... This may take a minute.")
            
            # Status indicator
            status_indicator = gr.Markdown("Enter a topic and click 'Generate Research Plan' to begin")
            
            # Plan review section (initially hidden)
            with gr.Group(visible=False) as plan_review_group:
                gr.Markdown("### 📋 Research Plan")
                gr.Markdown("Review the generated plan and approve or request revisions:")
                plan_output = gr.Textbox(label="", lines=10)
                
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve Plan", variant="primary")
                    revise_btn = gr.Button("🔄 Request Revisions", variant="secondary")
                
                # Add a loading indicator for when the plan is being revised
                revision_loading = gr.Markdown(visible=False, value="🔄 Revising plan based on feedback...")
                
                with gr.Group(visible=False) as feedback_group:
                    feedback_input = gr.Textbox(
                        label="Feedback for Plan Revision", 
                        placeholder="Provide specific feedback to improve the plan...",
                        lines=3
                    )
                    submit_feedback_btn = gr.Button("Submit Feedback", variant="primary")
            
            # Progress indicator (initially hidden)
            progress_indicator = gr.Markdown(visible=False)
            
            # Final report section (initially hidden)
            with gr.Group(visible=False) as report_group:
                gr.Markdown("### 📊 Generated Research Report")
                output = gr.Markdown()
                
                restart_btn = gr.Button("Start New Research", variant="secondary")
                # Add download button for the report
                download_btn = gr.Button("💾 Download Report as Markdown", variant="primary")
                
                # Hidden textbox to store the raw markdown for download
                report_markdown = gr.Textbox(visible=False)
                # Add a file component for downloading
                report_file = gr.File(visible=False, label="Download Report")
    
    # Function to handle initial plan generation
    def handle_submit(topic, tavily_key, google_key):
        if not topic or not tavily_key or not google_key:
            missing = []
            if not topic: missing.append("Research Topic")
            if not tavily_key: missing.append("Tavily API Key")
            if not google_key: missing.append("Google API Key")
            
            return {
                status_indicator: gr.Markdown(value=f"⚠️ Please fill in all required fields: {', '.join(missing)}"),
                plan_review_group: gr.Group(visible=False),
                report_group: gr.Group(visible=False),
                plan_loading: gr.Markdown(visible=False)
            }
        
        # Show loading indicator
        plan_loading.visible = True
        status_indicator.value = "🔍 Analyzing topic and generating research plan..."
            
        result = run_report_generation(topic, tavily_key, google_key)
        
        if result["status"] == "plan_ready":
            return {
                status_indicator: gr.Markdown(value="✅ Research plan generated! Please review and approve or request revisions."),
                plan_review_group: gr.Group(visible=True),
                plan_output: gr.Textbox(value=result["plan"]),
                thread_id: result["thread_id"],
                feedback_group: gr.Group(visible=False),
                report_group: gr.Group(visible=False),
                plan_loading: gr.Markdown(visible=False)
            }
        else:
            return {
                status_indicator: gr.Markdown(value="❌ Error generating research plan. Please try again."),
                plan_review_group: gr.Group(visible=False),
                report_group: gr.Group(visible=False),
                plan_loading: gr.Markdown(visible=False)
            }
    
    # Function to handle plan approval
    def handle_approve(topic, tavily_key, google_key, current_thread_id):
        status_indicator.value = "🔍 Researching and generating report based on approved plan..."
        progress_indicator.value = "This may take a few minutes. The system is searching the web and compiling information..."
        progress_indicator.visible = True
        
        result = run_report_generation(topic, tavily_key, google_key, feedback=True, thread_id=current_thread_id)
        
        if result["status"] == "report_ready":
            return {
                status_indicator: gr.Markdown(value="✅ Report generation complete!"),
                plan_review_group: gr.Group(visible=False),
                progress_indicator: gr.Markdown(visible=False),
                report_group: gr.Group(visible=True),
                output: gr.Markdown(value=result["report"]),
                report_markdown: result["report"]  # Store raw markdown for download
            }
        else:
            return {
                status_indicator: gr.Markdown(value="❌ Error generating report. Please try again."),
                progress_indicator: gr.Markdown(visible=False)
            }
    
    # Function to show feedback input when revise is clicked
    def handle_revise_click():
        return {
            feedback_group: gr.Group(visible=True),
            status_indicator: gr.Markdown(value="Please provide specific feedback to improve the research plan")
        }
    
    # Function to handle feedback submission
    def handle_feedback_submit(topic, tavily_key, google_key, feedback, current_thread_id):
        if not feedback:
            return {
                status_indicator: gr.Markdown(value="⚠️ Please provide feedback for revision")
            }
        
        # Show loading indicator for revision
        revision_loading.visible = True
        status_indicator.value = "🔄 Revising research plan based on your feedback..."
            
        result = run_report_generation(topic, tavily_key, google_key, feedback=feedback, thread_id=current_thread_id)
        
        if result["status"] == "plan_ready":
            return {
                status_indicator: gr.Markdown(value="✅ Research plan revised! Please review again."),
                plan_output: gr.Textbox(value=result["plan"]),
                feedback_group: gr.Group(visible=False),
                thread_id: result["thread_id"],
                revision_loading: gr.Markdown(visible=False)
            }
        else:
            return {
                status_indicator: gr.Markdown(value="❌ Error revising plan. Please try again."),
                revision_loading: gr.Markdown(visible=False)
            }
    
    # Function to reset the interface for a new research topic
    def reset_interface():
        return {
            topic_input: gr.Textbox(value=""),
            status_indicator: gr.Markdown(value="Enter a topic and click 'Generate Research Plan' to begin"),
            plan_review_group: gr.Group(visible=False),
            report_group: gr.Group(visible=False),
            thread_id: None,
            report_file: gr.File(visible=False)
        }
    
    # Function to create and download the markdown file
    def download_markdown(report_text, topic):
        import tempfile
        import os
        
        # Create a safe filename from the topic
        safe_topic = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in topic)
        safe_topic = safe_topic.replace(' ', '_')
        filename = f"{safe_topic}_report.md"
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.md', mode='w', encoding='utf-8') as f:
            f.write(report_text)
            temp_path = f.name
        
        return {
            report_file: gr.File(value=temp_path, visible=True, label=f"Download: {filename}")
        }
    
    # Connect event handlers
    submit_btn.click(
        fn=handle_submit,
        inputs=[topic_input, tavily_key, google_key],
        outputs=[status_indicator, plan_review_group, plan_output, thread_id, feedback_group, report_group, plan_loading]
    )
    
    approve_btn.click(
        fn=handle_approve, 
        inputs=[topic_input, tavily_key, google_key, thread_id],
        outputs=[status_indicator, plan_review_group, progress_indicator, report_group, output, report_markdown]
    )
    
    revise_btn.click(
        fn=handle_revise_click,
        inputs=[],
        outputs=[feedback_group, status_indicator]
    )
    
    submit_feedback_btn.click(
        fn=handle_feedback_submit,
        inputs=[topic_input, tavily_key, google_key, feedback_input, thread_id],
        outputs=[status_indicator, plan_output, feedback_group, thread_id, revision_loading]
    )
    
    restart_btn.click(
        fn=reset_interface,
        inputs=[],
        outputs=[topic_input, status_indicator, plan_review_group, report_group, thread_id, report_file]
    )
    
    download_btn.click(
        fn=download_markdown,
        inputs=[report_markdown, topic_input],
        outputs=[report_file]
    )

if __name__ == "__main__":
    demo.launch()