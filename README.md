# Story-Generator-Transformers

Enter a prompt and watch the story unfold... powered by Falcon-7B-Instruct from Hugging Face!

## Overview

This is a simple yet powerful AI-powered story generation app. Users can input a story idea, and the app generates a creative short story using the `tiiuae/falcon-7b-instruct` large language model.


---

## Features

- Prompt-based story generation
- Adjustable token length
- Creative outputs using Falcon-7B-Instruct
- Built using Streamlit + Hugging Face Transformers

---

## Tech Stack

- [Streamlit](https://streamlit.io/) - UI framework
- [Transformers](https://huggingface.co/docs/transformers/index) - Hugging Face Transformers
- [Torch](https://pytorch.org/) - Backend framework
- Model used: [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct)

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/YourUsername/story-generator-transformers.git
cd story-generator-transformers
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
