import streamlit as st
from story_generator import generate_story

st.set_page_config(page_title="AI Story Generator", layout="centered")

st.title("🧙‍♂️ AI Story Generator")
st.subheader("Enter a prompt and watch the story unfold...")

prompt = st.text_area("Enter your story prompt", placeholder="e.g. A knight finds a dragon made of ice")

length = st.slider("Max story length (tokens)", min_value=50, max_value=500, value=200, step=50)

if st.button("Generate Story"):
    if prompt.strip():
        with st.spinner("Crafting your story..."):
            story = generate_story(prompt, max_length=length)
            st.markdown("### ✨ Your Story:")
            st.write(story)
    else:
        st.warning("Please enter a prompt.")
