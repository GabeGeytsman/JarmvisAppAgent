<h1 align="center">JarmvisAppAgent</h1>
<p align="center">
  <em>A macOS-compatible fork of AppAgentX for GUI automation via LLM-powered agents</em>
  <br><br>
  <a href='https://github.com/Westlake-AGI-Lab/AppAgentX'><img src='https://img.shields.io/badge/Based_on-AppAgentX-blue'></a>&nbsp;
  <a href='https://arxiv.org/abs/2503.02268'><img src='https://img.shields.io/badge/Paper-ArXiv-red'></a>&nbsp;
</p>

## Abstract

Recent advancements in Large Language Models (LLMs) have led to the development of intelligent LLM-based agents capable of interacting with graphical user interfaces (GUIs). These agents demonstrate strong reasoning and adaptability, enabling them to perform complex tasks that traditionally required predefined rules. However, the reliance on step-by-step reasoning in LLM-based agents often results in inefficiencies, particularly for routine tasks. In contrast, traditional rule-based systems excel in efficiency but lack the intelligence and flexibility to adapt to novel scenarios.
To address this challenge, we propose a novel evolutionary framework for GUI agents that enhances operational efficiency while retaining intelligence and flexibility. Our approach incorporates a memory mechanism that records the agent's task execution history. By analyzing this history, the agent identifies repetitive action sequences and evolves high-level actions that act as shortcuts, replacing these low-level operations and improving efficiency. This allows the agent to focus on tasks requiring more complex reasoning, while simplifying routine actions.
Experimental results on multiple benchmark tasks demonstrate that our approach significantly outperforms existing methods in both efficiency and accuracy. The code will be open-sourced to support further research.

---

## Detailed Setup Guide (macOS)

This guide covers setting up AppAgentX on macOS with local services (no Docker). Tested on macOS with Apple Silicon (M-series chips).

### Prerequisites

- **Python 3.11+** (tested with 3.13)
- **Android Studio** (for emulator) or physical Android device with USB debugging
- **Homebrew** (for macOS package management)
- **Neo4j Desktop** (graph database)

### 1. Clone and Set Up Python Environment

```bash
git clone <your-repo-url>
cd JarmvisAppAgent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate
```

### 2. API Keys Required

You'll need accounts and API keys for the following services:

| Service | Purpose | Where to Get |
|---------|---------|--------------|
| **OpenAI** | LLM for task reasoning (GPT-4o with vision) | https://platform.openai.com/api-keys |
| **Pinecone** | Vector database for embeddings | https://www.pinecone.io/ (free tier available) |
| **Neo4j** | Graph database for action chains | Local install or https://neo4j.com/ |

**Important:** DeepSeek's `deepseek-chat` model does NOT support vision/images. You must use OpenAI GPT-4o or another vision-capable model.

### 3. Configure Secrets

Copy the example secrets file and add your API keys:

```bash
cp app_secrets.example.py app_secrets.py
```

Edit `app_secrets.py`:
```python
# OpenAI (required - must be vision-capable model)
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = "sk-proj-your-key-here"

# Pinecone (required)
PINECONE_API_KEY = "your-pinecone-key"

# Neo4j (local installation)
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-neo4j-password"
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Critical Version Notes

Some packages have compatibility issues. Install these specific versions:

```bash
# CRITICAL: transformers 4.57+ breaks Florence2 beam search
pip install transformers==4.49.0

# PaddleOCR 3.x has langchain conflicts - use 2.x
pip install "paddleocr<3.0"
pip install paddlepaddle
```

### 5. Download OmniParser Model Weights

Download the following models from HuggingFace and place them in the correct directories:

```bash
# Create weights directory
mkdir -p backend/OmniParser/weights

# Download from HuggingFace:
# 1. Icon detection model (YOLO-based)
#    https://huggingface.co/microsoft/OmniParser
#    -> Place in: backend/OmniParser/weights/icon_detect_v1_5/model.pt

# 2. Icon caption model (Florence2-based)
#    https://huggingface.co/microsoft/OmniParser
#    -> Place in: backend/OmniParser/weights/icon_caption_florence/
```

Expected structure:
```
backend/OmniParser/weights/
├── icon_detect_v1_5/
│   └── model.pt (or best.pt - check your download)
└── icon_caption_florence/
    ├── config.json
    ├── model.safetensors
    └── ... (other Florence2 files)
```

### 6. Set Up Neo4j

1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and database
3. Start the database
4. Set your password (update `app_secrets.py` accordingly)
5. The database should be accessible at `bolt://localhost:7687`

### 7. Set Up Android Emulator

1. Install [Android Studio](https://developer.android.com/studio)
2. Open **Device Manager** (Tools > Device Manager)
3. Create a new virtual device (recommend Pixel with recent Android)
4. Start the emulator
5. Verify ADB connection:
   ```bash
   adb devices
   # Should show: emulator-5554    device
   ```

### 8. Start Backend Services

You need to run two backend services in separate terminals:

**Terminal 1 - OmniParser (Screen Parsing)**
```bash
cd backend/OmniParser
python omni.py
# Runs on http://127.0.0.1:8000
```

**Terminal 2 - ImageEmbedding (Feature Extraction)**
```bash
cd backend/ImageEmbedding
python image_embedding.py
# Runs on http://127.0.0.1:8001
```

Wait for both services to fully load (OmniParser takes ~30-60 seconds to load Florence2 model).

### 9. Launch the Demo

**Terminal 3 - Main Application**
```bash
python demo.py
# Opens Gradio UI at http://localhost:7860
```

### 10. Using the Application

1. Open http://localhost:7860 in your browser
2. Go to **Initialization** tab
3. Click **Refresh Device List** and select your emulator
4. Enter a task description (e.g., "Open Chrome and search for weather")
5. Click **Initialize**
6. Go to **Auto Exploration** tab
7. Click **Start Exploration**

**Tip:** Make sure the emulator is on the home screen before starting exploration.

---

## Troubleshooting

### Florence2 Beam Search Error
```
'NoneType' object has no attribute 'shape'
```
**Solution:** Downgrade transformers to 4.49.0
```bash
pip install transformers==4.49.0
```

### PaddleOCR langchain Import Error
```
ModuleNotFoundError: No module named 'langchain.docstore'
```
**Solution:** Use PaddleOCR 2.x instead of 3.x
```bash
pip uninstall paddleocr paddlex
pip install "paddleocr<3.0"
```

### OmniParser Model Loading - attention_mask Warning
If you see warnings about `attention_mask` with Florence2, the model will still work but you can suppress warnings by ensuring you have the correct transformers version.

### LLM Vision Not Supported
```
openai.BadRequestError: unknown variant 'image_url'
```
**Solution:** Your LLM model doesn't support vision. Use OpenAI GPT-4o or another vision-capable model. DeepSeek's standard API does NOT support images.

### ADB Device Not Found
```bash
# Check if ADB sees the device
adb devices

# If empty, ensure:
# 1. Emulator is running
# 2. USB debugging is enabled (for physical devices)
# 3. ADB is in your PATH
```

### Port Already in Use
```bash
# Find and kill process on port
lsof -ti:8000 | xargs kill -9  # OmniParser
lsof -ti:8001 | xargs kill -9  # ImageEmbedding
lsof -ti:7860 | xargs kill -9  # Gradio
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio UI (demo.py)                     │
│                    http://localhost:7860                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 LangGraph State Machine                      │
│   tsk_setting → page_understand → perform_action → ...      │
│                    (explor_auto.py)                          │
└─────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenAI    │    │   OmniParser    │    │  ADB Controller │
│   GPT-4o    │    │  (port 8000)    │    │  (Android)      │
│   (Vision)  │    │  YOLO+Florence2 │    │                 │
└─────────────┘    └─────────────────┘    └─────────────────┘
                           │
                           ▼
                   ┌─────────────────┐
                   │ ImageEmbedding  │
                   │  (port 8001)    │
                   │   ResNet50      │
                   └─────────────────┘
```

---

## Related Links

- [AppAgent](https://arxiv.org/abs/id) - First LLM-based intelligent smartphone application agent
- [OmniParser](https://github.com/microsoft/OmniParser) - Microsoft's multimodal interface parsing tool
- [LangChain](https://github.com/langchain-ai/langchain) - Framework for building LLM-powered applications

## Attribution

This project is a fork/adaptation of [AppAgentX](https://github.com/Westlake-AGI-Lab/AppAgentX) by the Westlake AGI Lab. Huge thanks to the original authors for their groundbreaking work on evolving GUI agents. If you use this work academically, please cite their original paper.

## Acknowledgments

This codebase was extensively developed and refactored with the help of **Claude Code** (Anthropic's AI coding assistant). All the coding work, debugging, and documentation was done by Claude, with one friendly human in the loop providing direction and testing.

## Contact

Questions or feedback? Reach out to [Gabriel Geytsman](mailto:ggeyts@gmail.com).
