# Blog Writing Agent

A LangGraph-based agent that plans and writes blog posts section by section. The
app can decide whether a topic needs current web research, gather evidence with
Tavily, generate illustrations, and present the result in a Streamlit interface.

## Features

- Routes topics into research or closed-book writing.
- Creates a structured outline and writes its sections in parallel.
- Can use Tavily search results as evidence for current topics.
- Plans and generates up to three supporting images.
- Shows the plan, sources, Markdown preview, images, and run logs in Streamlit.
- Saves generated Markdown in the working directory and images under `images/`.

## Getting started

The commands below use PowerShell on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root and add the keys for the services you
want to use:

```dotenv
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
```

- `OPENAI_API_KEY` is used by the app's `gpt-4.1-mini` model.
- `TAVILY_API_KEY` enables web research; it can be omitted for topics that do
  not need research.
- `GOOGLE_API_KEY` enables image generation. Image generation also requires the
  optional `google-genai` package: `pip install google-genai`.

Keep real keys in `.env`; do not commit them. `.gitignore` excludes `.env` and
local virtual environments.

## Run the app

From the project root, activate the environment and run:

```powershell
streamlit run bwa_frontend.py
```

Enter a topic and select **Generate Blog**. Generated Markdown is saved as a
`.md` file in the current working directory. Image files are saved in `images/`.
The interface also provides Markdown and Markdown-plus-images downloads.

## Project layout

| File | Description |
| --- | --- |
| `bwa_backend.py` | LangGraph workflow for routing, research, planning, parallel writing, and image handling. |
| `bwa_frontend.py` | Streamlit UI for generating, previewing, and downloading blog posts. |
| `requirements.txt` | Pinned Python dependencies for the main app. |
| `1_bwa_basic.ipynb` | Initial LangGraph blog-writing workflow. |
| `2_bwa_improved_prompting.ipynb` | Improved planning and section-writing prompts. |
| `3_bwa_research.ipynb` | Adds topic routing and Tavily research. |
| `4_bwa_research_fine_tuned.ipynb` | Refines the research-enabled workflow. |
| `5_bwa_image.ipynb` | Adds image planning and generation to the workflow. |
| `tavily_test.ipynb` | Small notebook for trying Tavily search. |
| `understanding_self_attention_in_deep_learning.md` and `unlocking_the_power_of_self-attention_*.md` | Example generated blog posts. |

The notebooks are exploratory examples and may use providers or packages beyond
the main app's pinned requirements. For the Groq-based notebooks, install
`langchain-groq`; research notebooks may also need `langchain-community`.
