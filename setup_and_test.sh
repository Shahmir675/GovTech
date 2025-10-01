#!/bin/bash

echo "========================================="
echo "RAGBot-v2 Setup and Testing Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv venv
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${RED}✗ Failed to create virtual environment${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo ""
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo -e "${YELLOW}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Download spaCy model
echo ""
echo -e "${YELLOW}Downloading spaCy language model...${NC}"
python -m spacy download en_core_web_sm --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ spaCy model downloaded${NC}"
else
    echo -e "${RED}✗ Failed to download spaCy model${NC}"
    exit 1
fi

# Run syntax checks
echo ""
echo "========================================="
echo "Running Syntax Checks"
echo "========================================="
echo ""

FILES=(
    "processing.py"
    "case_agent.py"
    "law_agent.py"
    "drafting_agent.py"
    "orchestrator.py"
    "app_v2.py"
    "app.py"
    "pdf_processor.py"
    "vector_store.py"
    "hybrid_search.py"
    "gemini_client.py"
)

SYNTAX_ERRORS=0
for file in "${FILES[@]}"; do
    python -m py_compile "$file" 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $file syntax OK"
    else
        echo -e "${RED}✗${NC} $file has syntax errors"
        SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
    fi
done

if [ $SYNTAX_ERRORS -gt 0 ]; then
    echo -e "${RED}Found $SYNTAX_ERRORS files with syntax errors${NC}"
    exit 1
fi

# Run import tests
echo ""
echo "========================================="
echo "Running Import Tests"
echo "========================================="
echo ""

python -c "
import sys
sys.path.insert(0, '.')

modules = [
    ('processing', 'Processing Layer'),
    ('case_agent', 'Case Agent'),
    ('law_agent', 'Law Agent'),
    ('drafting_agent', 'Drafting Agent'),
    ('orchestrator', 'Orchestrator'),
]

errors = 0
for module, name in modules:
    try:
        __import__(module)
        print(f'✓ {name} imports successfully')
    except Exception as e:
        print(f'✗ {name} import failed: {e}')
        errors += 1

sys.exit(errors)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All modules import successfully${NC}"
else
    echo -e "${RED}✗ Some modules failed to import${NC}"
    exit 1
fi

# Run pytest
echo ""
echo "========================================="
echo "Running Unit Tests"
echo "========================================="
echo ""

pytest tests/ -v --tb=short
TEST_RESULT=$?

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Run v1 Q&A mode: streamlit run app.py"
    echo "3. Run v2 Multi-Agent mode: streamlit run app_v2.py"
else
    echo -e "${YELLOW}⚠ Some tests failed (this may be expected if external services are not configured)${NC}"
    echo ""
    echo "You can still run the application if .env is properly configured:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Run v1 Q&A mode: streamlit run app.py"
    echo "3. Run v2 Multi-Agent mode: streamlit run app_v2.py"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
