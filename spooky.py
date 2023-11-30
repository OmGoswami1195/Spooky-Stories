import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Create a sidebar for page selection
st.sidebar.title("Eldritch Menu")
page = st.sidebar.selectbox("Select a Page", ["Generate Story", "Prompt Examples"])

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1663443525887-f2ef0cfc7684?auto=format&fit=crop&q=80&w=3132&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: cover;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Main content area
st.title("🎃🕷️🕸️ Sinister Tale Weaver 🕸️🕷️🎃")

# Load the GPT-2 Model
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define a function to generate text
def generate_horror_story(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    generated_text = model.generate(input_ids, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7, pad_token_id=50256, attention_mask=attention_mask)
    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return generated_text

if page == "Generate Story":
    st.header("Generate a Horror Story  👻 🦇 👻 🦇")

    # Create a text input field for the user to enter a prompt
    user_prompt = st.text_area("Enter a horror story prompt:")
    if user_prompt:
      if st.button("Generate Story"):
          story = generate_horror_story(user_prompt)
          st.write("Generated Horror Story:")
          st.write(story)

if page == "Prompt Examples":
    st.header("Prompt Examples")
    
    # Provide some example prompts
    st.write("Try using one of these prompts:")
    st.text("💀 In the eerie silence of the old mansion, I heard footsteps approaching...")
    st.text("💀 As the clock struck midnight, a chilling wind blew through the graveyard...")
    st.text("💀 Deep in the forest, I stumbled upon an ancient, overgrown cemetery...")
    st.text("💀 In the old, abandoned asylum, I heard the whispers of long-forgotten souls...")
    st.text("💀 The antique mirror in the attic reflected something that wasn't there...")
    st.text("💀 A mysterious figure stood at the end of the dimly lit alley, beckoning me to approach...")
    st.text("💀 On a moonless night, the town's clock tower chimed thirteen times...")
    st.text("💀 The cursed painting in the haunted house seemed to follow my every move...")
    st.text("💀 A child's laughter echoed through the empty playground in the dead of night...")
    st.text("💀 The flickering candlelight revealed strange symbols etched into the basement walls...")
    st.text("💀 I found a diary in the attic, filled with the ramblings of someone who had gone mad...")
    st.text("💀 In the foggy forest, I stumbled upon an ancient, overgrown cemetery...")
    st.text("💀 As I gazed out the window, I saw a face staring back at me from the rain-soaked glass...")

