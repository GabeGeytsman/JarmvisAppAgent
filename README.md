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

## Current Architecture (Grid-Based, No OmniParser)

This fork has been significantly refactored to use a **numbered grid overlay system** instead of OmniParser for UI element identification. This approach is simpler, faster, and more reliable.

### How the Grid System Works

1. **Screenshot Capture**: Raw screenshot is taken from Android device via ADB
2. **Grid Overlay**: A 9×22 = 198 numbered squares are overlaid on the screenshot
3. **Adaptive Colors**: Grid numbers use the complement of the dominant background color for visibility
4. **LLM Reasoning**: Claude analyzes the gridded image and returns a square number to tap
5. **Coordinate Translation**: Square number is converted to (x, y) coordinates for ADB

```
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  9  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 10  │ 11  │ 12  │ 13  │ 14  │ 15  │ 16  │ 17  │ 18  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ ... │     │     │     │     │     │     │     │ ... │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 190 │ 191 │ 192 │ 193 │ 194 │ 195 │ 196 │ 197 │ 198 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Grid Overlay | `explor_auto.py` | `add_coordinate_grid()` - adds numbered squares to screenshots |
| Screen Actions | `tool/screen_content.py` | `screen_action` tool - translates squares to ADB commands |
| Exploration | `explor_auto.py` | LangGraph workflow for autonomous task exploration |
| Deployment | `deployment.py` | Execute tasks, optionally using knowledge graph templates |
| Knowledge Graph | `data/graph_db.py` | Neo4j storage for learned action sequences |
| Vector Embeddings | `data/vector_db.py` | Pinecone storage for visual similarity search |
| Image Features | `backend/ImageEmbedding/` | ResNet50-based feature extraction service |

### Services Overview

| Service | Port | Purpose | Required? |
|---------|------|---------|-----------|
| **Gradio Demo** | 7860 | Main UI | Yes |
| **ImageEmbedding** | 8001 | ResNet50 feature extraction for visual similarity | For knowledge graph |
| **Neo4j** | 7687 | Graph database for action chains | For knowledge graph |
| **~~OmniParser~~** | ~~8000~~ | ~~UI element detection~~ | **REMOVED** |

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
| **Anthropic** | Claude for task reasoning (vision-capable) | https://console.anthropic.com/ |
| **Pinecone** | Vector database for embeddings (optional) | https://www.pinecone.io/ (free tier available) |
| **Neo4j** | Graph database for action chains (optional) | Local install or https://neo4j.com/ |

**Note:** The system uses Claude (claude-sonnet-4-5-20250929 or claude-opus-4-5-20251101) for vision-based reasoning. Configure the model in `config.py`.

### 3. Configure Secrets

Copy the example secrets file and add your API keys:

```bash
cp app_secrets.example.py app_secrets.py
```

Edit `app_secrets.py`:
```python
# Anthropic (required)
ANTHROPIC_API_KEY = "sk-ant-your-key-here"

# Legacy OpenAI (optional, not currently used)
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = "sk-proj-your-key-here"

# Pinecone (optional - for knowledge graph visual matching)
PINECONE_API_KEY = "your-pinecone-key"

# Neo4j (optional - for knowledge graph)
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

### 5. Set Up Neo4j (Optional - for Knowledge Graph)

1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and database
3. Start the database
4. Set your password (update `app_secrets.py` accordingly)
5. The database should be accessible at `bolt://localhost:7687`

To start Neo4j from command line:
```bash
neo4j start
# Check status: neo4j status
# Stop: neo4j stop
```

### 6. Set Up Android Emulator

1. Install [Android Studio](https://developer.android.com/studio)
2. Open **Device Manager** (Tools > Device Manager)
3. Create a new virtual device (recommend Pixel with recent Android)
4. Start the emulator
5. Verify ADB connection:
   ```bash
   adb devices
   # Should show: emulator-5554    device
   ```

### 7. Start Backend Services (Optional)

The ImageEmbedding service is only needed if you want to use the knowledge graph for visual similarity matching. **For basic exploration/deployment, you can skip this.**

**Terminal 1 - ImageEmbedding (Feature Extraction) - Optional**
```bash
cd backend/ImageEmbedding
python image_embedding.py
# Runs on http://127.0.0.1:8001
# Uses ResNet50 for image feature extraction
```

After starting, initialize the model:
```bash
curl -X POST "http://127.0.0.1:8001/set_model" \
    -H "Content-Type: application/json" \
    -d '{"model_name": "resnet50"}'
```

### 8. Launch the Demo

**Main Terminal - Application**
```bash
python demo.py
# Opens Gradio UI at http://localhost:7860
```

### 9. Using the Application

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

### ADB Device Not Found
```bash
# Check if ADB sees the device
adb devices

# If empty, ensure:
# 1. Emulator is running
# 2. USB debugging is enabled (for physical devices)
# 3. ADB is in your PATH
```

### Neo4j Connection Failed
```bash
# Check if Neo4j is running
neo4j status

# Start Neo4j
neo4j start

# If using Neo4j Desktop, ensure the database is started in the UI
```

### Port Already in Use
```bash
# Find and kill process on port
lsof -ti:8001 | xargs kill -9  # ImageEmbedding
lsof -ti:7860 | xargs kill -9  # Gradio
lsof -ti:7687 | xargs kill -9  # Neo4j
```

### Claude API Errors
- Ensure your `ANTHROPIC_API_KEY` is valid in `app_secrets.py`
- Check you have sufficient API credits
- The model name in `config.py` must be a valid Claude model (e.g., `claude-sonnet-4-5-20250929`)

---

## Neo4j Queries (Knowledge Graph)

Access the Neo4j browser at http://localhost:7474 (default credentials: neo4j / your-password).

### View All Nodes and Relationships
```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```

### View All Pages
```cypher
MATCH (p:Page) RETURN p
```

### View All Elements
```cypher
MATCH (e:Element) RETURN e
```

### View Page → Element → Page Chains
```cypher
MATCH (p1:Page)-[:HAS_ELEMENT]->(e:Element)-[:LEADS_TO]->(p2:Page)
RETURN p1, e, p2
```

### View Chain Starting Points (First Pages)
```cypher
MATCH (p:Page)
WHERE NOT EXISTS { MATCH ()-[:LEADS_TO]->(p) }
RETURN p
```

### View Complete Action Chain from a Task
```cypher
MATCH path = (start:Page)-[:HAS_ELEMENT|LEADS_TO*]->(end:Page)
WHERE NOT EXISTS { (end)-[:HAS_ELEMENT]->() }
RETURN path LIMIT 10
```

### Clear All Data (Use with Caution!)
```cypher
MATCH (n) DETACH DELETE n
```

### Count Nodes by Type
```cypher
MATCH (n) RETURN labels(n) as type, count(n) as count
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Gradio UI (demo.py)                              │
│                       http://localhost:7860                              │
│  ┌──────────────┬─────────────────┬──────────────┬───────────────────┐  │
│  │Initialization│ Auto Exploration│User Exploration│ Action Execution │  │
│  └──────────────┴─────────────────┴──────────────┴───────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LangGraph State Machine                             │
│                                                                          │
│   Exploration (explor_auto.py):                                         │
│   tsk_setting → page_understand → perform_action → task_judgment        │
│                                                                          │
│   Deployment (deployment.py):                                           │
│   capture_screen → [knowledge_graph_lookup] → react_action → judgment   │
└─────────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐
│    Anthropic    │  │  Grid Overlay   │  │      ADB Controller         │
│     Claude      │  │  (9x22=198 sq)  │  │        (Android)            │
│ claude-sonnet   │  │  Adaptive color │  │  tap/swipe/text/screenshot  │
│    (Vision)     │  │  number labels  │  │                             │
└─────────────────┘  └─────────────────┘  └─────────────────────────────┘
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
┌─────────────────────────┐                    ┌─────────────────────────┐
│   Neo4j (port 7687)     │                    │  ImageEmbedding (8001)  │
│   Knowledge Graph       │◄──────────────────►│  ResNet50 Features      │
│   Page→Element→Page     │                    │         │               │
│   Action Templates      │                    │         ▼               │
└─────────────────────────┘                    │  Pinecone (cloud)       │
                                               │  Vector Similarity      │
                                               └─────────────────────────┘
```

### Data Flow

1. **Screenshot** → ADB captures raw screen from Android device
2. **Grid Overlay** → 198 numbered squares added with adaptive colors
3. **Claude Vision** → Analyzes gridded image, reasons about task, returns square number
4. **Coordinate Translation** → Square number → (x, y) pixel coordinates
5. **ADB Action** → Executes tap/swipe/text at coordinates
6. **Knowledge Graph** (optional) → Stores successful action sequences for replay

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
