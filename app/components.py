import gradio as gr


def status_html(text: str) -> str:
    return f"<div style='padding:8px;border-radius:8px;background:#eef4ff'>{text}</div>"
