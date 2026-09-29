import streamlit as st
import ollama
from pathlib import Path

MODEL = "qwen3:8b"
ROOT = Path(r"C:\LocalAI\workspace").resolve()
ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Qwen3 File Agent", page_icon="🤖")
st.title("🤖 Qwen3 Local File Agent")
st.caption("Local AI chat with file-management tools")

def safe_path(path):
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Path outside workspace is prohibited.")
    return target

def create_file(path: str, content: str) -> str:
    """Create a file with the supplied content."""
    target = safe_path(path)
    if target.exists():
        return "Error: File already exists."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Created: {target}"

def rename_file(old_path: str, new_path: str) -> str:
    """Rename or move a file within the workspace."""
    source = safe_path(old_path)
    destination = safe_path(new_path)
    if not source.is_file():
        return "Error: Source file does not exist."
    if destination.exists():
        return "Error: Destination already exists."
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    return f"Renamed to: {destination}"

def delete_file(path: str, confirmation: str = "") -> str:
    """Delete a file. Requires the user to explicitly confirm."""
    target = safe_path(path)
    if not target.is_file():
        return "Error: File does not exist."
    if confirmation != "yes":
        return "Deletion not authorized. Ask the user to confirm."
    target.unlink()
    return f"Deleted: {target}"

def list_files(path: str = ".") -> str:
    """List files and folders in the workspace."""
    target = safe_path(path)
    if not target.is_dir():
        return "Error: Directory does not exist."
    return "\n".join(
        ("[DIR] " if p.is_dir() else "[FILE] ") + p.name
        for p in sorted(target.iterdir())
    ) or "Directory is empty."

TOOL_FUNCS = {
    "create_file": create_file,
    "rename_file": rename_file,
    "delete_file": delete_file,
    "list_files": list_files,
}

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rename_file",
            "description": "Rename a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_path": {"type": "string"},
                    "new_path": {"type": "string"}
                },
                "required": ["old_path", "new_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file only after explicit user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "confirmation": {
                        "type": "string",
                        "description": "Must be yes only when the user explicitly confirms deletion."
                    }
                },
                "required": ["path", "confirmation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and folders in a workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    }
]

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": (
            "You are a local file-management assistant. "
            "Use tools for file operations. Paths are relative "
            "to the workspace. Never claim an operation succeeded "
            "unless the tool confirms it. Before deleting, ask "
            "the user to explicitly confirm deletion."
        )
    }]

if st.button("Clear conversation"):
    st.session_state.messages = [st.session_state.messages[0]]
    st.rerun()

for msg in st.session_state.messages[1:]:
    if msg["role"] in ("user", "assistant") and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

prompt = st.chat_input("Ask Qwen3 to manage your files...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            for _ in range(8):
                response = ollama.chat(
                    model=MODEL,
                    messages=st.session_state.messages,
                    tools=TOOL_DEFS,
                )
                msg = response.message
                st.session_state.messages.append(msg)

                if not msg.tool_calls:
                    st.markdown(msg.content or "Done.")
                    break

                for call in msg.tool_calls:
                    name = call.function.name
                    args = call.function.arguments

                    # Require a separate, explicit confirmation
                    # phrase in the user's latest message.
                    if name == "delete_file":
                        authorized = (
                            prompt.strip().lower().startswith(
                                "confirm deletion "
                            )
                        )
                        if not authorized:
                            result = (
                                "Deletion blocked. Ask the user to "
                                "type: CONFIRM DELETION <filename>"
                            )
                        else:
                            args["confirmation"] = "yes"

                    try:
                        if name == "delete_file" and not authorized:
                            pass
                        elif name not in TOOL_FUNCS:
                            result = "Error: Unknown tool."
                        else:
                            result = TOOL_FUNCS[name](**args)
                    except Exception as exc:
                        result = f"Error: {exc}"

                    st.info(f"{name}: {result}")
                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_name": name,
                        "content": str(result),
                    })
            else:
                st.warning("Tool-call limit reached.")
        except Exception as exc:
            st.error(f"Agent error: {exc}")

