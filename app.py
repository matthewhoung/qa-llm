import streamlit as st
import time
import random
import os
from rag import Generator

# =============================================================================
# 1. CONSTANTS & MAPPING
# =============================================================================
st.set_page_config(page_title="AI Fortune Teller (六爻神卦)", page_icon="🔮", layout="centered")

DATA_FILE = "fortune-table.txt"

# King Wen Sequence Mapping (Binary -> ID)
BINARY_TO_ID = {
    "111111": 1, "000000": 2, "100010": 3, "010001": 4, "111010": 5, "010111": 6,
    "010000": 7, "000010": 8, "111011": 9, "110111": 10, "111000": 11, "000111": 12,
    "101111": 13, "111101": 14, "001000": 15, "000100": 16, "100110": 17, "011001": 18,
    "110000": 19, "000011": 20, "100101": 21, "101001": 22, "000001": 23, "100000": 24,
    "100111": 25, "111001": 26, "100001": 27, "011110": 28, "010010": 29, "101101": 30,
    "001110": 31, "011100": 32, "001111": 33, "111100": 34, "000101": 35, "101000": 36,
    "101011": 37, "110101": 38, "001010": 39, "010100": 40, "110001": 41, "100011": 42,
    "111110": 43, "011111": 44, "000110": 45, "011000": 46, "010110": 47, "011010": 48,
    "101110": 49, "011101": 50, "100100": 51, "001001": 52, "001011": 53, "110100": 54,
    "101100": 55, "001101": 56, "011011": 57, "110110": 58, "010011": 59, "110010": 60,
    "110011": 61, "001100": 62, "101010": 63, "010101": 64
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
@st.cache_data
def load_fortune_table():
    """Parses the fortune-table.txt into a dictionary."""
    data = {}
    if not os.path.exists(DATA_FILE):
        return None
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) >= 4:
                idx = parts[0].strip()
                data[idx] = {
                    "name": parts[1],
                    "structure": parts[2],
                    "meaning": parts[3],
                    "luck": parts[4] if len(parts) > 4 else "Unknown",
                    "full_text": line.strip()
                }
    return data

def toss_coin_animation():
    """Simulates the animation of tossing 3 coins."""
    placeholders = [st.empty() for _ in range(3)]
    
    for _ in range(5):
        for p in placeholders:
            p.markdown("<div style='font-size: 40px; text-align: center;'>🪙</div>", unsafe_allow_html=True)
        time.sleep(0.1)
        for p in placeholders:
            p.markdown("<div style='font-size: 40px; text-align: center;'>💫</div>", unsafe_allow_html=True)
        time.sleep(0.1)
    
    toss_values = [random.choice([2, 3]) for _ in range(3)]
    total = sum(toss_values)
    
    for p in placeholders:
        p.empty()
        
    return total

# =============================================================================
# 3. MAIN APP
# =============================================================================
def main():
    with st.sidebar:
        st.header("⚙️ Settings")
        hf_token = st.text_input("HuggingFace Token", type="password")
        if not hf_token:
            hf_token = st.secrets.get("HF_TOKEN", "")
            
        fortune_data = load_fortune_table()
        if fortune_data:
            st.success(f"Loaded {len(fortune_data)} Hexagrams.")
        else:
            st.error(f"File '{DATA_FILE}' not found.")

    st.title("🔮 AI Fortune Teller (六爻神卦)")
    st.markdown("Focus on your question. The coins will reveal the path, and AI will interpret the meaning.")

    user_question = st.text_input("What do you wish to ask?", placeholder="e.g., Will my new project succeed?")

    if "hex_result" not in st.session_state:
        st.session_state.hex_result = None

    if st.button("🎲 擲卦 (Cast Coins)"):
        if not user_question:
            st.warning("Please enter your question first.")
            return

        if not fortune_data:
            st.error("Database missing.")
            return

        st.divider()
        st.write("### Casting the Six Lines...")
        
        lines_display = []
        binary_code = []
        
        for i in range(1, 7):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"Line {i}:")
            with col2:
                total = toss_coin_animation()
                if total == 6:   # Old Yin (Change)
                    symbol = "━  ━  X"
                    val = "0"
                elif total == 7: # Young Yang
                    symbol = "━━━"
                    val = "1"
                elif total == 8: # Young Yin
                    symbol = "━  ━"
                    val = "0"
                elif total == 9: # Old Yang (Change)
                    symbol = "━━━  O"
                    val = "1"
                
                st.markdown(f"**{symbol}**")
                lines_display.insert(0, symbol)
                binary_code.append(val)
        
        full_binary = "".join(binary_code)
        hex_id = BINARY_TO_ID.get(full_binary)
        
        if hex_id:
            hex_info = fortune_data.get(str(hex_id))
            st.session_state.hex_result = {
                "id": hex_id,
                "info": hex_info,
                "lines": lines_display,
                "question": user_question
            }
        else:
            st.error(f"Error: Hexagram pattern {full_binary} not found.")

    if st.session_state.hex_result:
        res = st.session_state.hex_result
        info = res["info"]
        
        st.divider()
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader(f"Hexagram {res['id']}: {info['name']}")
            st.info(f"**Luck Level:** {info['luck']}")
            st.write(f"**Structure:** {info['structure']}")
            st.text("\n".join(res["lines"]))

        with c2:
            st.subheader("The Oracle's Meaning")
            st.success(info["meaning"])

        st.divider()
        st.subheader("🤖 AI Interpretation")
        
        if not hf_token:
            st.warning("Please enter HuggingFace Token.")
        else:
            with st.spinner("Interpreting the signs..."):
                try:
                    gen = Generator(hf_token, model_key="qwen-2.5")
                    
                    # --- FIX IS HERE: Use Chat Completion Format ---
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an expert I Ching Master. Interpret the user's fortune based on the hexagram provided."
                        },
                        {
                            "role": "user", 
                            "content": f"""
                            The User asked: "{res['question']}"
                            
                            The divination result is:
                            - Hexagram Name: {info['name']}
                            - Luck Level: {info['luck']}
                            - Traditional Meaning: {info['meaning']}
                            
                            Please interpret this result specifically for the user's question. 
                            Explain what "{info['meaning']}" implies for their situation in simple terms.
                            """
                        }
                    ]
                    
                    response = gen.client.chat_completion(
                        messages=messages,
                        model=gen.model,
                        max_tokens=600,
                        temperature=0.7
                    )
                    
                    # Extract the actual text content from the message
                    ai_reply = response.choices[0].message.content
                    st.write(ai_reply)
                    
                except Exception as e:
                    st.error(f"AI Error: {e}")

if __name__ == "__main__":
    main()