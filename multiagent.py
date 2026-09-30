import json
import subprocess
import psutil
import ollama

MODEL_NAME = "qwen3:8b"

# ==========================================
# 1. DEFINE AGENT TOOLS (Python Functions)
# ==========================================

def get_system_metrics() -> str:
    """Returns local host system resource utilization (CPU, RAM, Disk)."""
    metrics = {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    return json.dumps(metrics)

def ping_host(hostname: str) -> str:
    """
    Pings a network host to evaluate connectivity.
    Args:
        hostname: Domain or IP to test (e.g. '1.1.1.1' or 'google.com')
    """
    try:
        res = subprocess.check_output(["ping", "-c", "2", hostname], stderr=subprocess.STDOUT, text=True)
        return json.dumps({"status": "success", "raw_output": res.strip()})
    except Exception as err:
        return json.dumps({"status": "error", "message": str(err)})

# Map function names to executable Python code
SYSTEM_TOOLS_MAP = {"get_system_metrics": get_system_metrics}
NETWORK_TOOLS_MAP = {"ping_host": ping_host}

# ==========================================
# 2. DEFINE JSON SCHEMAS
# ==========================================

SYSTEM_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_system_metrics",
            "description": "Get current CPU, RAM, and Disk metrics of the machine.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

NETWORK_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "ping_host",
            "description": "Ping a specified hostname or IP address to verify connectivity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {
                        "type": "string",
                        "description": "Target hostname or IP address."
                    }
                },
                "required": ["hostname"]
            }
        }
    }
]

# ==========================================
# 3. GENERIC AGENT RUNNER
# ==========================================

def run_agent(agent_name: str, system_prompt: str, user_query: str, tools_schema: list, tools_map: dict):
    print(f"\n--- [{agent_name}] Activated ---")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    # 1. Ask Qwen3:8b if a tool call is needed
    response = ollama.chat(model=MODEL_NAME, messages=messages, tools=tools_schema)
    messages.append(response["message"])
    
    tool_calls = response["message"].get("tool_calls", [])
    
    if not tool_calls:
        print(f"[{agent_name} Response]:")
        print(response["message"]["content"])
        return

    # 2. Execute tool locally if requested
    for call in tool_calls:
        fn_name = call["function"]["name"]
        fn_args = call["function"]["arguments"]
        
        print(f"  └─ Executing Tool: `{fn_name}` with parameters: {fn_args}")
        if fn_name in tools_map:
            result = tools_map[fn_name](**fn_args)
            
            # Feed tool execution result back into conversation context
            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": result
            })

    # 3. Final response synthesis by Qwen3:8b
    final_response = ollama.chat(model=MODEL_NAME, messages=messages)
    print(f"\n[{agent_name} Final Answer]:")
    print(final_response["message"]["content"])

# ==========================================
# 4. ORCHESTRATOR ROUTER
# ==========================================

def orchestrate_query(user_query: str):
    print(f"\n==========================================")
    print(f"USER QUERY: \"{user_query}\"")
    print(f"==========================================")
    
    router_prompt = f"""You are a query router. Analyze the user prompt and respond with ONLY ONE word:
    - 'SYSTEM' if the query asks about local CPU, disk, memory, or system hardware status.
    - 'NETWORK' if the query asks about pinging, network latency, or internet connectivity.
    - 'UNKNOWN' if it fits neither.
    Query: {user_query}"""

    route_res = ollama.chat(
        model=MODEL_NAME, 
        messages=[{"role": "user", "content": router_prompt}]
    )
    
    decision = route_res["message"]["content"].strip().upper()

    if "SYSTEM" in decision:
        run_agent(
            agent_name="System Health Agent",
            system_prompt="You are a system administration agent. Use tools to check CPU, RAM, or Disk metrics.",
            user_query=user_query,
            tools_schema=SYSTEM_TOOLS_SCHEMA,
            tools_map=SYSTEM_TOOLS_MAP
        )
    elif "NETWORK" in decision:
        run_agent(
            agent_name="Network Agent",
            system_prompt="You are a network diagnostic agent. Use tools to check network status and ping target hosts.",
            user_query=user_query,
            tools_schema=NETWORK_TOOLS_SCHEMA,
            tools_map=NETWORK_TOOLS_MAP
        )
    else:
        print("\n[Router]: Request does not match active agent domains.")
# ==========================================
# 5. TEST RUNS
# ==========================================

if __name__ == "__main__":
    # Test 1: Routes to System Health Agent
    orchestrate_query("Can you check if my CPU or RAM are overloading right now?")
    
    # Test 2: Routes to Network Diagnostics Agent
    orchestrate_query("Ping 8.8.8.8 to see if our network connection to DNS is stable.")

