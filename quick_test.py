#!/usr/bin/env python3
"""
Quick test script to verify RAGBot-v2 code structure and basic functionality
without requiring all dependencies to be installed.
"""

import sys
import os
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*50}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓{Colors.NC} {text}")

def print_error(text):
    print(f"{Colors.RED}✗{Colors.NC} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠{Colors.NC} {text}")

def check_file_exists(filepath):
    """Check if a file exists"""
    if Path(filepath).exists():
        print_success(f"{filepath} exists")
        return True
    else:
        print_error(f"{filepath} MISSING")
        return False

def check_syntax(filepath):
    """Check Python file syntax"""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        print_success(f"{filepath} syntax valid")
        return True
    except SyntaxError as e:
        print_error(f"{filepath} syntax error: {e}")
        return False

def check_imports_exist(filepath):
    """Check if import statements are valid (not if packages are installed)"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Count import statements
        import_lines = [line for line in content.split('\n') if line.strip().startswith(('import ', 'from '))]
        print_success(f"{filepath} has {len(import_lines)} import statements")
        return True
    except Exception as e:
        print_error(f"{filepath} error reading imports: {e}")
        return False

def test_env_file():
    """Test .env file configuration"""
    print_header("Testing Environment Configuration")

    if not check_file_exists('.env'):
        return False

    required_vars = [
        'GEMINI_API_KEY',
        'QDRANT_URL',
        'QDRANT_API_KEY',
        'QDRANT_COLLECTION_NAME',
    ]

    try:
        with open('.env', 'r') as f:
            content = f.read()

        missing = []
        for var in required_vars:
            if var in content:
                # Check if it has a value (not empty or placeholder)
                for line in content.split('\n'):
                    if line.startswith(f'{var}='):
                        value = line.split('=', 1)[1].strip()
                        if value and 'your_' not in value.lower():
                            print_success(f"{var} is configured")
                        else:
                            print_warning(f"{var} is empty or placeholder")
                            missing.append(var)
                        break
            else:
                print_error(f"{var} not found in .env")
                missing.append(var)

        if missing:
            print_warning(f"Missing or incomplete variables: {', '.join(missing)}")
            return False
        else:
            print_success("All required environment variables configured")
            return True

    except Exception as e:
        print_error(f"Error reading .env: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print_header("Testing File Structure")

    required_files = {
        'Core Files': [
            'processing.py',
            'case_agent.py',
            'law_agent.py',
            'drafting_agent.py',
            'orchestrator.py',
        ],
        'UI Files': [
            'app.py',
            'app_v2.py',
        ],
        'Existing Infrastructure': [
            'pdf_processor.py',
            'vector_store.py',
            'hybrid_search.py',
            'gemini_client.py',
        ],
        'Configuration': [
            'requirements.txt',
            '.env',
            '.env.example',
            'README.md',
        ],
        'Documentation': [
            'docs/architecture.md',
        ],
        'Tests': [
            'tests/conftest.py',
            'tests/test_processing.py',
            'tests/test_orchestrator.py',
        ]
    }

    all_ok = True
    for category, files in required_files.items():
        print(f"\n{category}:")
        for filepath in files:
            if not check_file_exists(filepath):
                all_ok = False

    return all_ok

def test_syntax():
    """Test Python syntax of all files"""
    print_header("Testing Python Syntax")

    python_files = [
        'processing.py',
        'case_agent.py',
        'law_agent.py',
        'drafting_agent.py',
        'orchestrator.py',
        'app_v2.py',
        'app.py',
        'pdf_processor.py',
        'vector_store.py',
        'hybrid_search.py',
        'gemini_client.py',
    ]

    all_ok = True
    for filepath in python_files:
        if not check_syntax(filepath):
            all_ok = False

    return all_ok

def test_module_structure():
    """Test that modules have expected structure"""
    print_header("Testing Module Structure")

    modules = {
        'processing.py': ['NERExtractor', 'ClaimExtractor', 'DocumentProcessor'],
        'case_agent.py': ['CaseAgent'],
        'law_agent.py': ['LawAgent'],
        'drafting_agent.py': ['DraftingAgent'],
        'orchestrator.py': ['AgentOrchestrator', 'WorkflowState'],
    }

    all_ok = True
    for filepath, expected_classes in modules.items():
        try:
            with open(filepath, 'r') as f:
                content = f.read()

            missing_classes = []
            for class_name in expected_classes:
                if f'class {class_name}' in content:
                    print_success(f"{filepath} has class {class_name}")
                else:
                    print_error(f"{filepath} missing class {class_name}")
                    missing_classes.append(class_name)
                    all_ok = False

            if not missing_classes:
                print_success(f"{filepath} has all expected classes")

        except Exception as e:
            print_error(f"Error checking {filepath}: {e}")
            all_ok = False

    return all_ok

def test_documentation():
    """Test that documentation is complete"""
    print_header("Testing Documentation")

    # Check README
    try:
        with open('README.md', 'r') as f:
            readme = f.read()

        required_sections = [
            'RAGBot-v2',
            'Multi-Agent',
            'Setup Instructions',
            'Usage',
            'v1 Q&A Mode',
            'v2 Multi-Agent Mode',
            'Testing',
        ]

        missing = []
        for section in required_sections:
            if section in readme:
                print_success(f"README has '{section}' section")
            else:
                print_error(f"README missing '{section}' section")
                missing.append(section)

        if not missing:
            print_success("README.md is complete")
            return True
        else:
            return False

    except Exception as e:
        print_error(f"Error checking README.md: {e}")
        return False

def main():
    print(f"{Colors.BLUE}")
    print("╔════════════════════════════════════════════════╗")
    print("║     RAGBot-v2 Quick Test Suite                ║")
    print("║     Testing code structure and syntax         ║")
    print("╚════════════════════════════════════════════════╝")
    print(f"{Colors.NC}")

    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_syntax),
        ("Module Structure", test_module_structure),
        ("Environment Config", test_env_file),
        ("Documentation", test_documentation),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"{test_name} failed with exception: {e}")
            results[test_name] = False

    # Print summary
    print_header("Test Summary")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")

    print(f"\n{Colors.BLUE}{'='*50}{Colors.NC}")
    if passed == total:
        print(f"{Colors.GREEN}✓ ALL TESTS PASSED ({passed}/{total}){Colors.NC}")
        print(f"{Colors.GREEN}✓ Code structure is valid!{Colors.NC}")
        print(f"\n{Colors.YELLOW}Next steps:{Colors.NC}")
        print("1. Run: ./setup_and_test.sh")
        print("   (This will create venv, install dependencies, and run full tests)")
        print("\n2. Or manually:")
        print("   python -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        print("   python -m spacy download en_core_web_sm")
        print("   streamlit run app_v2.py")
        return 0
    else:
        print(f"{Colors.RED}✗ SOME TESTS FAILED ({passed}/{total} passed){Colors.NC}")
        print(f"{Colors.YELLOW}Please review the errors above{Colors.NC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
