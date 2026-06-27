import uuid
import gradio as gr
from agent import build_agent, run_agent

graph, _ = build_agent()

def chat(user_message: str, history: list, thread_id: str):
    if not user_message.strip():
        return "", history, thread_id
    if not thread_id:
        thread_id = str(uuid.uuid4())
    response = run_agent(graph, thread_id, user_message)
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response}
    ]
    return "", history, thread_id

def reset_conversation():
    return [], str(uuid.uuid4()), ""

EXAMPLE_PROMPTS = [
    "What's the weather like in Paris right now?",
    "Trip from 2025-09-05 to 2025-09-20, how long is that?",
    "Convert 1000 USD to EUR.",
    "Top 5 things to do in Rome?",
    "Best time to visit Bali?",
    "Visa requirements for Pakistan nationals visiting Japan?",
]

with gr.Blocks(title="✈️ AI Travel Assistant") as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
        <h1 style="font-size:2rem; margin:0;">✈️ AI Travel Assistant</h1>
        <p style="color:#555; margin-top:0.4rem;">
            Powered by LangGraph · OpenAI · Weather · Search · Currency
        </p>
    </div>
    """)

    thread_id_state = gr.State(str(uuid.uuid4()))
    chatbot = gr.Chatbot(label="Travel Assistant", height=500)

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask me anything about your trip…",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button("Send ✈️", scale=1, variant="primary")

    gr.HTML("<p style='text-align:center; color:#888; font-size:0.85rem;'>Try an example:</p>")
    with gr.Row():
        for ex in EXAMPLE_PROMPTS[:3]:
            gr.Button(ex).click(fn=lambda e=ex: e, outputs=msg_box)
    with gr.Row():
        for ex in EXAMPLE_PROMPTS[3:]:
            gr.Button(ex).click(fn=lambda e=ex: e, outputs=msg_box)

    reset_btn = gr.Button("🔄 New Conversation", variant="secondary")

    gr.HTML("""
    <div style="margin-top:1rem; padding:1rem; background:#f0f7ff; border-radius:8px; font-size:0.8rem; color:#444;">
        <b>🛠 Tools:</b> 🌤 Weather · 🔍 Search · 💱 Currency · 📅 Trip Duration
    </div>
    """)

    send_btn.click(chat, [msg_box, chatbot, thread_id_state], [msg_box, chatbot, thread_id_state])
    msg_box.submit(chat, [msg_box, chatbot, thread_id_state], [msg_box, chatbot, thread_id_state])
    reset_btn.click(reset_conversation, outputs=[chatbot, thread_id_state, msg_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)