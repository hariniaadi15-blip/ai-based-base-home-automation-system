"""
AI-Based Home Automation System
Stack: Streamlit + LangChain + Ollama (local LLM) + optional ESP32 (pyserial / HTTP)

Run:
    ollama pull llama3.2
    streamlit run app.py
"""
import json
import random
import re
from datetime import datetime

import requests
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

try:
    import serial  # pyserial (only needed for ESP32 serial mode)
except ImportError:
    serial = None

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="AI Home Automation", page_icon="🏠", layout="wide")

DEVICES = {
    "living_room_light": "💡 Living Room Light",
    "bedroom_light": "🛏️ Bedroom Light",
    "fan": "🌀 Fan",
    "ac": "❄️ Air Conditioner",
    "tv": "📺 TV",
    "door_lock": "🔒 Door Lock",
}


def init_state():
    ss = st.session_state
    ss.setdefault("devices", {d: "off" for d in DEVICES})
    ss["devices"].setdefault("door_lock", "on")  # on = locked
    ss.setdefault("ac_temp", 24)
    ss.setdefault("chat", [])
    ss.setdefault("log", [])
    ss.setdefault("temp", 29.0)
    ss.setdefault("humidity", 55.0)


init_state()

# --------------------------------------------------------------------------
# Sidebar: settings
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
model_name = st.sidebar.text_input("Ollama model", "llama3.2")
mode = st.sidebar.selectbox(
    "Hardware mode", ["Simulation", "ESP32 (Serial)", "ESP32 (HTTP)"]
)
serial_port, esp_ip = "COM3", "192.168.1.50"
if mode == "ESP32 (Serial)":
    serial_port = st.sidebar.text_input("Serial port", "COM3")
elif mode == "ESP32 (HTTP)":
    esp_ip = st.sidebar.text_input("ESP32 IP address", "192.168.1.50")
use_llm = st.sidebar.checkbox("Use AI (Ollama)", value=True)
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.chat = []
    st.rerun()

# --------------------------------------------------------------------------
# Hardware layer (simulation / ESP32)
# --------------------------------------------------------------------------
def send_to_hardware(device: str, state: str, value=None) -> str:
    """Send a command to ESP32 (serial/HTTP) or just simulate."""
    if mode == "Simulation":
        return "simulated"
    try:
        if mode == "ESP32 (Serial)":
            if serial is None:
                return "pyserial not installed"
            with serial.Serial(serial_port, 115200, timeout=2) as ser:
                ser.write(f"{device}:{state}:{value or ''}\n".encode())
            return "sent via serial"
        r = requests.get(
            f"http://{esp_ip}/control",
            params={"device": device, "state": state, "value": value or ""},
            timeout=3,
        )
        return f"HTTP {r.status_code}"
    except Exception as e:  # keep the app alive if hardware is offline
        return f"hardware error: {e}"


def read_sensors() -> dict:
    """Read temperature/humidity from ESP32 or generate simulated values."""
    ss = st.session_state
    if mode == "ESP32 (HTTP)":
        try:
            data = requests.get(f"http://{esp_ip}/sensors", timeout=3).json()
            ss.temp, ss.humidity = float(data["temp"]), float(data["humidity"])
            return {"temp": ss.temp, "humidity": ss.humidity}
        except Exception:
            pass
    elif mode == "ESP32 (Serial)" and serial is not None:
        try:
            with serial.Serial(serial_port, 115200, timeout=2) as ser:
                ser.write(b"READ\n")
                data = json.loads(ser.readline().decode().strip())
            ss.temp, ss.humidity = float(data["temp"]), float(data["humidity"])
            return {"temp": ss.temp, "humidity": ss.humidity}
        except Exception:
            pass
    # Simulated random walk
    ss.temp = round(min(40, max(18, ss.temp + random.uniform(-0.6, 0.6))), 1)
    ss.humidity = round(min(90, max(30, ss.humidity + random.uniform(-1.5, 1.5))), 1)
    return {"temp": ss.temp, "humidity": ss.humidity}


def apply_action(device: str, state: str, value=None) -> str:
    ss = st.session_state
    if device not in DEVICES or state not in ("on", "off"):
        return f"Ignored invalid action: {device}/{state}"
    ss.devices[device] = state
    if device == "ac" and value:
        try:
            ss.ac_temp = max(16, min(30, int(value)))
        except ValueError:
            pass
    status = send_to_hardware(device, state, value)
    msg = f"{DEVICES[device]} → {state.upper()} ({status})"
    ss.log.insert(0, f"{datetime.now():%H:%M:%S}  {msg}")
    return msg


# --------------------------------------------------------------------------
# AI layer (LangChain + Ollama)
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a smart home assistant controlling these devices:
{devices}

Current device states: {states}
Sensor data: {sensors}

Convert the user's request into JSON ONLY (no extra text, no markdown):
{{"actions": [{{"device": "<device_id>", "state": "on|off", "value": "<optional, e.g. AC temperature>"}}],
  "reply": "<short friendly reply to the user>"}}

Rules:
- device must be one of the device ids above. For door_lock, "on" = locked, "off" = unlocked.
- If the user only asks a question (e.g. temperature), use an empty actions list and answer in reply.
- Use the sensor data when answering or suggesting actions."""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


@st.cache_resource(show_spinner=False)
def get_chain(model: str):
    llm = ChatOllama(model=model, temperature=0, format="json")
    return prompt | llm | StrOutputParser()


def ai_interpret(question: str, sensors: dict) -> dict:
    chain = get_chain(model_name)
    raw = chain.invoke(
        {
            "devices": ", ".join(DEVICES),
            "states": json.dumps(st.session_state.devices),
            "sensors": json.dumps(sensors),
            "question": question,
        }
    )
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group(0) if match else raw)


# Fallback when Ollama is not running: simple keyword parser
KEYWORDS = {
    "living": "living_room_light", "bedroom": "bedroom_light", "light": "living_room_light",
    "fan": "fan", "ac": "ac", "air condition": "ac", "tv": "tv",
    "television": "tv", "door": "door_lock", "lock": "door_lock",
}


def rule_based(question: str, sensors: dict) -> dict:
    q = question.lower()
    if "temperature" in q or "humid" in q:
        return {"actions": [], "reply": f"It's {sensors['temp']}°C with {sensors['humidity']}% humidity."}
    state = "off" if any(w in q for w in ["off", "close", "unlock", "stop"]) else "on"
    if "all" in q and "light" in q:
        devs = ["living_room_light", "bedroom_light"]
    else:
        devs = [d for k, d in KEYWORDS.items() if k in q]
        if "bedroom" in q:
            devs = [d for d in devs if d != "living_room_light"]
    devs = list(dict.fromkeys(devs))
    if not devs:
        return {"actions": [], "reply": "Sorry, I didn't understand. Try: 'turn on the fan'."}
    value = re.search(r"(\d{2})", q)
    actions = [{"device": d, "state": state, "value": value.group(1) if d == "ac" and value else None} for d in devs]
    return {"actions": actions, "reply": "Done! (rule-based mode)"}


def handle_command(question: str) -> str:
    sensors = read_sensors()
    result, note = None, ""
    if use_llm:
        try:
            result = ai_interpret(question, sensors)
        except Exception as e:
            note = f"\n\n_⚠️ Ollama unavailable ({type(e).__name__}); used rule-based mode._"
    if result is None:
        result = rule_based(question, sensors)
    done = [apply_action(a.get("device", ""), a.get("state", ""), a.get("value")) for a in result.get("actions", [])]
    reply = result.get("reply", "Done.")
    if done:
        reply += "\n\n" + "\n".join(f"- {d}" for d in done)
    return reply + note


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🏠 AI Home Automation")

sensors = read_sensors()
c1, c2, c3 = st.columns(3)
c1.metric("🌡️ Temperature", f"{sensors['temp']} °C")
c2.metric("💧 Humidity", f"{sensors['humidity']} %")
c3.metric("❄️ AC set point", f"{st.session_state.ac_temp} °C")

if sensors["temp"] > 32 and st.session_state.devices["ac"] == "off":
    st.warning("It's hot! Try saying: *turn on the AC at 24*")

st.subheader("Devices")
cols = st.columns(3)
for i, (dev, label) in enumerate(DEVICES.items()):
    with cols[i % 3]:
        current = st.session_state.devices[dev] == "on"
        text = label + (" (locked)" if dev == "door_lock" and current else "")
        new = st.toggle(text, value=current, key=f"tg_{dev}_{current}")
        if new != current:
            apply_action(dev, "on" if new else "off")
            st.rerun()

st.divider()
st.subheader("💬 Talk to your home")
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("e.g. Turn on the fan and the bedroom light"):
    st.session_state.chat.append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        answer = handle_command(question)
    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.rerun()

with st.expander("📜 Activity log"):
    st.write("\n\n".join(st.session_state.log) or "No activity yet.")
