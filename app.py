import os
from dotenv import load_dotenv

# Load environment from .env if present, then set GROQ key as a fallback.
load_dotenv()
# If GROQ_API_KEY not set via environment or .env, hardcode the provided key here.
if not os.getenv('GROQ_API_KEY'):
    os.environ['GROQ_API_KEY'] = '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'

from rag import answer_question


def main():
    print("Laptop RAG System")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        answer, source = answer_question(query)
        print("\nAnswer:")
        print(answer)
        print(f"\nSource used: {source}")
        print("-" * 40)


def gradio_ui():
    try:
        import gradio as gr
    except Exception:
        print("Gradio not installed; falling back to CLI.")
        return main()

    def handle(query: str):
        answer, source = answer_question(query)
        return answer, source

    with gr.Blocks() as demo:
        gr.Markdown("# Laptop RAG System")
        with gr.Row():
            inp = gr.Textbox(label="Ask a question", lines=2, placeholder="Type your question here...")
            btn = gr.Button("Ask")
        out = gr.Textbox(label="Answer", lines=10)
        src = gr.Textbox(label="Source", lines=2)
        btn.click(fn=handle, inputs=inp, outputs=[out, src])

    demo.launch(debug=True, share=True)


if __name__ == "__main__":
    # Prefer Gradio UI if available, otherwise CLI
    try:
        gradio_ui()
    except Exception:
        main()
