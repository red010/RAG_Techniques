"""
3D Embeddings Visualization - CLI Dataset Manager

A powerful command-line tool for creating and managing multiple embedding datasets
with persistent storage and easy visualization switching.

Commands:
    create <name> --file <path>   Create a new dataset from JSON input
    list                           List all available datasets
    activate <name>                Set a dataset as active
    delete <name>                  Remove a dataset
    run [--port]                   Start visualization server with active dataset
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from http.server import SimpleHTTPRequestHandler
import socketserver
from dotenv import load_dotenv
import numpy as np

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_STORE_DIR = SCRIPT_DIR / "data_store"
INPUT_EXAMPLES_DIR = SCRIPT_DIR / "input_examples"
PUBLIC_DIR = SCRIPT_DIR / "public"
CONFIG_FILE = DATA_STORE_DIR / "config.json"

# Load environment variables
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)

# Check for API key (required for create command)
def check_api_key():
    """Check if GEMINI_API_KEY is available."""
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ Error: GEMINI_API_KEY not found in .env file")
        print(f"   Looking for .env at: {env_path}")
        print("\nPlease add your Google API key to the .env file:")
        print("   GEMINI_API_KEY=your_key_here")
        sys.exit(1)

# Lazy imports for dependencies
def import_dependencies():
    """Import heavy dependencies only when needed."""
    global genai, umap
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    except ImportError:
        print("❌ Error: google-generativeai is not installed")
        print("\nPlease install it with:")
        print("   pip install google-generativeai")
        sys.exit(1)
    
    try:
        import umap
    except ImportError:
        print("❌ Error: umap-learn is not installed")
        print("\nPlease install it with:")
        print("   pip install umap-learn")
        sys.exit(1)

# ============================================================================
# CONFIG MANAGER
# ============================================================================

class ConfigManager:
    """Manages dataset configuration and state."""
    
    @staticmethod
    def ensure_data_store():
        """Create data_store directory if it doesn't exist."""
        DATA_STORE_DIR.mkdir(exist_ok=True)
        PUBLIC_DIR.mkdir(exist_ok=True)
    
    @staticmethod
    def load_config():
        """Load or create config.json."""
        ConfigManager.ensure_data_store()
        
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = {
                "active_dataset": None,
                "datasets": {}
            }
            ConfigManager.save_config(default_config)
            return default_config
    
    @staticmethod
    def save_config(config):
        """Save config.json atomically."""
        ConfigManager.ensure_data_store()
        
        # Write to temp file first
        temp_file = CONFIG_FILE.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # Atomic rename
        temp_file.replace(CONFIG_FILE)
    
    @staticmethod
    def add_dataset(name, metadata):
        """Add a dataset to config."""
        config = ConfigManager.load_config()
        config['datasets'][name] = metadata
        
        # If first dataset, set as active
        if config['active_dataset'] is None:
            config['active_dataset'] = name
        
        ConfigManager.save_config(config)
    
    @staticmethod
    def remove_dataset(name):
        """Remove a dataset from config."""
        config = ConfigManager.load_config()
        
        if name not in config['datasets']:
            raise ValueError(f"Dataset '{name}' not found")
        
        del config['datasets'][name]
        
        # If removing active dataset, clear active
        if config['active_dataset'] == name:
            # Try to set another dataset as active
            if config['datasets']:
                config['active_dataset'] = list(config['datasets'].keys())[0]
            else:
                config['active_dataset'] = None
        
        ConfigManager.save_config(config)
    
    @staticmethod
    def set_active(name):
        """Set a dataset as active."""
        config = ConfigManager.load_config()
        
        if name not in config['datasets']:
            raise ValueError(f"Dataset '{name}' not found")
        
        config['active_dataset'] = name
        ConfigManager.save_config(config)
    
    @staticmethod
    def get_active():
        """Get active dataset info."""
        config = ConfigManager.load_config()
        active_name = config['active_dataset']
        
        if active_name is None:
            return None, None
        
        return active_name, config['datasets'].get(active_name)
    
    @staticmethod
    def dataset_exists(name):
        """Check if a dataset exists."""
        config = ConfigManager.load_config()
        return name in config['datasets']

# ============================================================================
# JSON VALIDATOR
# ============================================================================

class JSONValidator:
    """Validates input JSON files against expected schema."""
    
    @staticmethod
    def validate_input_file(data):
        """Validate input JSON structure."""
        errors = []
        
        # Check top-level structure
        if 'dataset_info' not in data:
            errors.append("Missing 'dataset_info' field")
        else:
            if 'language' not in data['dataset_info']:
                errors.append("Missing 'dataset_info.language' field")
            if 'description' not in data['dataset_info']:
                errors.append("Missing 'dataset_info.description' field")
        
        if 'items' not in data:
            errors.append("Missing 'items' field")
        elif not isinstance(data['items'], list):
            errors.append("'items' must be a list")
        elif len(data['items']) == 0:
            errors.append("'items' list is empty")
        else:
            # Validate each item
            for i, item in enumerate(data['items']):
                item_errors = JSONValidator._validate_item(item, i)
                errors.extend(item_errors)
        
        return errors
    
    @staticmethod
    def _validate_item(item, index):
        """Validate a single item."""
        errors = []
        prefix = f"Item {index}"
        
        # Required fields
        if 'name' not in item:
            errors.append(f"{prefix}: Missing 'name' field")
        elif not isinstance(item['name'], dict):
            errors.append(f"{prefix}: 'name' must be an object")
        else:
            if 'local' not in item['name']:
                errors.append(f"{prefix}: Missing 'name.local' field")
            if 'en' not in item['name']:
                errors.append(f"{prefix}: Missing 'name.en' field")
        
        if 'category' not in item:
            errors.append(f"{prefix}: Missing 'category' field")
        
        if 'description' not in item:
            errors.append(f"{prefix}: Missing 'description' field")
        
        # Optional metadata field validation
        if 'metadata' in item and not isinstance(item['metadata'], dict):
            errors.append(f"{prefix}: 'metadata' must be an object")
        
        return errors

# ============================================================================
# DATASET PROCESSOR
# ============================================================================

class DatasetProcessor:
    """Processes datasets: computes embeddings and reduces dimensionality."""
    
    def __init__(self):
        """Initialize processor (requires API key and dependencies)."""
        check_api_key()
        import_dependencies()
    
    @staticmethod
    def create_embedding_text(item):
        """Create composite text for embedding from item fields."""
        local_name = item['name']['local']
        en_name = item['name']['en']
        category = item['category']
        description = item['description']
        
        return f"{local_name} ({en_name}) - Categoria: {category}. {description}"
    
    def compute_embeddings(self, items):
        """Compute embeddings for all items."""
        print(f"\n📊 Computing embeddings ({len(items)} items)...")
        
        embeddings = []
        for i, item in enumerate(items, 1):
            text = self.create_embedding_text(item)
            local_name = item['name']['local']
            
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                print(f"   [{i}/{len(items)}] {local_name}")
            except Exception as e:
                print(f"\n❌ Error embedding '{local_name}': {e}")
                sys.exit(1)
        
        return np.array(embeddings)
    
    def reduce_dimensionality(self, embeddings, n_components=3):
        """Reduce embeddings to 3D using UMAP."""
        print(f"\n🔄 Reducing dimensionality ({embeddings.shape[1]}D → {n_components}D)...")
        
        try:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(embeddings) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            
            reduced = reducer.fit_transform(embeddings)
            print(f"   ✓ Dimensionality reduced successfully")
            return reduced
            
        except Exception as e:
            print(f"\n❌ Error during UMAP reduction: {e}")
            print("   Tip: Ensure you have at least 10 items in your dataset")
            sys.exit(1)
    
    def process_dataset(self, input_data):
        """Complete pipeline: embedding + UMAP + enrichment."""
        items = input_data['items']
        
        # Step 1: Compute embeddings
        embeddings = self.compute_embeddings(items)
        
        # Step 2: Reduce dimensionality
        positions_3d = self.reduce_dimensionality(embeddings)
        
        # Step 3: Enrich items with positions
        enriched_items = []
        for item, position in zip(items, positions_3d):
            enriched_item = item.copy()
            enriched_item['position'] = position.tolist()
            enriched_items.append(enriched_item)
        
        # Step 4: Create output dataset
        output_dataset = {
            'dataset_info': input_data['dataset_info'],
            'items': enriched_items
        }
        
        return output_dataset

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def cmd_create(args):
    """Create a new dataset."""
    name = args.name
    input_file = Path(args.file)
    
    print("=" * 70)
    print(f"📊 Creating dataset: {name}")
    print("=" * 70)
    
    # Check if dataset already exists
    if ConfigManager.dataset_exists(name):
        print(f"\n❌ Error: Dataset '{name}' already exists")
        print("   Use 'delete' command first or choose a different name")
        sys.exit(1)
    
    # Check if input file exists
    if not input_file.exists():
        print(f"\n❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Load input file
    print(f"\n📂 Loading input file... ", end="")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print("✓")
    except json.JSONDecodeError as e:
        print(f"\n❌ Error: Invalid JSON in input file")
        print(f"   Details: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error reading file: {str(e)}")
        sys.exit(1)
    
    # Validate JSON schema
    print("🔍 Validating schema... ", end="")
    errors = JSONValidator.validate_input_file(input_data)
    if errors:
        print("✗")
        print("\n❌ Validation errors:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    print("✓")
    
    # Process dataset
    processor = DatasetProcessor()
    try:
        output_dataset = processor.process_dataset(input_data)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    
    # Save dataset
    print("\n💾 Saving dataset... ", end="")
    output_path = DATA_STORE_DIR / f"{name}.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dataset, f, ensure_ascii=False, indent=2)
        print("✓")
    except Exception as e:
        print(f"\n❌ Error saving dataset: {str(e)}")
        sys.exit(1)
    
    # Update config
    metadata = {
        'path': str(output_path.relative_to(SCRIPT_DIR)),
        'description': input_data['dataset_info']['description'],
        'created_at': datetime.now().isoformat(),
        'item_count': len(input_data['items']),
        'language': input_data['dataset_info']['language']
    }
    ConfigManager.add_dataset(name, metadata)
    
    # Check if this is the first/active dataset
    config = ConfigManager.load_config()
    is_active = config['active_dataset'] == name
    
    # Final summary
    print("\n" + "=" * 70)
    print(f"✅ Dataset '{name}' created successfully!")
    print("=" * 70)
    print(f"   Path: {output_path.relative_to(SCRIPT_DIR)}")
    print(f"   Items: {len(input_data['items'])}")
    print(f"   Language: {input_data['dataset_info']['language']}")
    if is_active:
        print(f"   Status: Active (first dataset)")
    print()

def cmd_list(args):
    """List all available datasets."""
    config = ConfigManager.load_config()
    
    if not config['datasets']:
        print("\n📁 No datasets available")
        print("   Create one with: python prepare_data.py create <name> --file <path>")
        return
    
    print("\n📁 Available Datasets:")
    print()
    
    for name, metadata in config['datasets'].items():
        is_active = name == config['active_dataset']
        marker = "*" if is_active else " "
        count = metadata['item_count']
        lang = metadata.get('language', 'unknown')
        
        status = "Active" if is_active else ""
        print(f"   {marker} {name} ({count} items, {lang}) {status}")
    
    print(f"\nTotal: {len(config['datasets'])} dataset(s)")
    print()

def cmd_activate(args):
    """Activate a dataset."""
    name = args.name
    
    try:
        ConfigManager.set_active(name)
        config = ConfigManager.load_config()
        metadata = config['datasets'][name]
        
        print(f"\n✅ Dataset '{name}' is now active")
        print(f"   Items: {metadata['item_count']}")
        print(f"   Description: {metadata['description']}")
        print()
        print("💡 Tip: If the server is running, restart it to see the changes:")
        print("   python prepare_data.py run --port 8000")
        print()
    except ValueError as e:
        print(f"\n❌ Error: {str(e)}")
        print("   Use 'list' command to see available datasets")
        sys.exit(1)

def cmd_delete(args):
    """Delete a dataset."""
    name = args.name
    force = args.force
    
    # Check if dataset exists
    if not ConfigManager.dataset_exists(name):
        print(f"\n❌ Error: Dataset '{name}' not found")
        print("   Use 'list' command to see available datasets")
        sys.exit(1)
    
    config = ConfigManager.load_config()
    metadata = config['datasets'][name]
    is_active = config['active_dataset'] == name
    
    # Confirm deletion (unless --force)
    if not force:
        print(f"\n⚠️  You are about to delete dataset '{name}'")
        print(f"   Items: {metadata['item_count']}")
        if is_active:
            print("   Status: ACTIVE (currently in use)")
        print("   This action cannot be undone.")
        print()
        
        response = input("   Are you sure? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("\n   Deletion cancelled")
            return
    
    # Delete file
    dataset_path = DATA_STORE_DIR / f"{name}.json"
    try:
        if dataset_path.exists():
            dataset_path.unlink()
    except Exception as e:
        print(f"\n❌ Error deleting file: {str(e)}")
        sys.exit(1)
    
    # Update config
    ConfigManager.remove_dataset(name)
    
    # Check new active dataset
    new_active, _ = ConfigManager.get_active()
    
    print(f"\n🗑️  Dataset '{name}' deleted successfully")
    if is_active and new_active:
        print(f"   New active dataset: {new_active}")
    elif is_active:
        print("   No active dataset (all datasets deleted)")
    print()

class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler that suppresses 404 errors for favicon.ico."""
    
    def log_message(self, format, *args):
        """Override to suppress favicon 404 errors."""
        # Suppress favicon 404 errors
        if len(args) >= 2 and args[0] == '/favicon.ico' and '404' in str(args[1]):
            return
        # Log everything else normally
        super().log_message(format, *args)
    
    def do_GET(self):
        """Override GET to handle favicon gracefully."""
        if self.path == '/favicon.ico':
            # Return empty 204 No Content for favicon
            self.send_response(204)
            self.end_headers()
            return
        # Handle all other requests normally
        super().do_GET()

def cmd_run(args):
    """Start visualization server."""
    port = args.port
    
    print("=" * 70)
    print("🚀 Starting visualization server...")
    print("=" * 70)
    
    # Check for active dataset
    active_name, active_metadata = ConfigManager.get_active()
    
    if active_name is None:
        print("\n❌ Error: No active dataset")
        print("   Create a dataset with: python prepare_data.py create <name> --file <path>")
        print("   Or activate one with: python prepare_data.py activate <name>")
        sys.exit(1)
    
    print(f"\n   Active dataset: {active_name}")
    print(f"   Items: {active_metadata['item_count']}")
    print(f"   Description: {active_metadata['description']}")
    
    # Copy dataset to public/data.json
    print(f"\n📋 Copying to public/data.json... ", end="")
    dataset_path = DATA_STORE_DIR / f"{active_name}.json"
    public_data_path = PUBLIC_DIR / "data.json"
    
    try:
        shutil.copy2(dataset_path, public_data_path)
        print("✓")
    except Exception as e:
        print(f"\n❌ Error copying dataset: {str(e)}")
        sys.exit(1)
    
    # Start server with custom handler
    print(f"\n🌐 Server running at: http://localhost:{port}")
    print()
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        # Change to script directory to serve files
        os.chdir(SCRIPT_DIR)
        
        # Create server with custom handler
        with socketserver.TCPServer(("", port), QuietHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)

# ============================================================================
# CLI PARSER
# ============================================================================

def create_parser():
    """Create argument parser with all commands."""
    parser = argparse.ArgumentParser(
        description="3D Embeddings Visualization - CLI Dataset Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py create ricette_italiane --file input_examples/ricette_italiane.json
  python prepare_data.py list
  python prepare_data.py activate ricette_italiane
  python prepare_data.py run
  python prepare_data.py delete old_dataset --force
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True
    
    # CREATE command
    parser_create = subparsers.add_parser(
        'create',
        help='Create a new dataset from JSON input'
    )
    parser_create.add_argument('name', help='Name for the new dataset')
    parser_create.add_argument('--file', required=True, help='Path to input JSON file')
    parser_create.set_defaults(func=cmd_create)
    
    # LIST command
    parser_list = subparsers.add_parser(
        'list',
        help='List all available datasets'
    )
    parser_list.set_defaults(func=cmd_list)
    
    # ACTIVATE command
    parser_activate = subparsers.add_parser(
        'activate',
        help='Set a dataset as active'
    )
    parser_activate.add_argument('name', help='Name of the dataset to activate')
    parser_activate.set_defaults(func=cmd_activate)
    
    # DELETE command
    parser_delete = subparsers.add_parser(
        'delete',
        help='Remove a dataset'
    )
    parser_delete.add_argument('name', help='Name of the dataset to delete')
    parser_delete.add_argument('--force', action='store_true', help='Skip confirmation')
    parser_delete.set_defaults(func=cmd_delete)
    
    # RUN command
    parser_run = subparsers.add_parser(
        'run',
        help='Start visualization server with active dataset'
    )
    parser_run.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser_run.set_defaults(func=cmd_run)
    
    return parser

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Execute command
    args.func(args)

if __name__ == "__main__":
    main()
