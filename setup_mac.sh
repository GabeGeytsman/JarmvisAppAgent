#!/bin/bash
# AppAgentX Mac Setup Script (No Docker Required)
# This script sets up the project to run directly on Mac without Docker

set -e

echo "========================================="
echo "AppAgentX Mac Setup (No Docker Required)"
echo "========================================="

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install ADB
echo ""
echo "Step 1: Installing ADB..."
brew install android-platform-tools 2>/dev/null || echo "ADB already installed"

# Install Neo4j
echo ""
echo "Step 2: Installing Neo4j..."
brew install neo4j 2>/dev/null || echo "Neo4j already installed"

# Create Python virtual environment
echo ""
echo "Step 3: Setting up Python environment..."
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install main requirements
echo "Installing main dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install pinecone-client

# Install backend requirements
echo "Installing OmniParser dependencies..."
pip install -r backend/OmniParser/requirements.txt

echo "Installing ImageEmbedding dependencies..."
pip install -r backend/ImageEmbedding/requirements.txt

# Add fastapi and uvicorn if not already installed
pip install fastapi uvicorn python-multipart

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Start Neo4j:"
echo "   neo4j start"
echo "   (Access at http://localhost:7474, default login: neo4j/neo4j)"
echo ""
echo "2. Edit config.py with your API keys:"
echo "   - LLM_BASE_URL and LLM_API_KEY (OpenAI or DeepSeek)"
echo "   - PINECONE_API_KEY (free at pinecone.io)"
echo "   - Neo4j_AUTH = ('neo4j', 'your-password')"
echo ""
echo "3. Start the backend services (in separate terminals):"
echo "   Terminal 1: cd backend/OmniParser && python -m uvicorn omni:app --port 8000"
echo "   Terminal 2: cd backend/ImageEmbedding && python -m uvicorn image_embedding:app --port 8001"
echo ""
echo "4. Connect Android device/emulator:"
echo "   adb devices"
echo ""
echo "5. Run the demo:"
echo "   source venv/bin/activate"
echo "   python demo.py"
echo ""
