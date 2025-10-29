# Quick Start Guide

## Installation

```bash
# Python dependencies
pip install google-generativeai umap-learn python-dotenv numpy

# JavaScript dependencies
npm install
```

## Setup

1. Add your API key to `.env` in the project root:
   ```
   GEMINI_API_KEY=your_key_here
   ```

## Basic Usage

### Create a dataset
```bash
python prepare_data.py create ricette_italiane --file input_examples/ricette_italiane.json
```

### List all datasets
```bash
python prepare_data.py list
```

### Switch active dataset
```bash
python prepare_data.py activate ricette_italiane
```

### Start visualization
```bash
python prepare_data.py run
```
Then open: http://localhost:8000

### Delete a dataset
```bash
python prepare_data.py delete old_dataset
```

## CLI Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `create <name> --file <path>` | Create new dataset | `create recipes --file data.json` |
| `list` | Show all datasets | `list` |
| `activate <name>` | Set active dataset | `activate recipes` |
| `delete <name> [--force]` | Remove dataset | `delete old --force` |
| `run [--port]` | Start server | `run --port 3000` |

## JSON Input Format

```json
{
  "dataset_info": {
    "language": "it",
    "description": "Dataset description"
  },
  "items": [
    {
      "name": {"local": "Nome", "en": "Name"},
      "category": "category_id",
      "description": "Detailed description for embedding",
      "metadata": {"key": "value"}
    }
  ]
}
```

## Visualization Controls

- **Drag**: Rotate view
- **Scroll**: Zoom in/out
- **Hover**: Show item name
- **Type**: Search/filter items
- **Click point**: Show 5 nearest neighbors
- **Click background**: Reset view

## Tips

1. **Create multiple datasets**: Each takes 1-2 minutes to compute, but switching is instant
2. **Use descriptive names**: They appear in the list and logs
3. **Add metadata**: Preserved in the dataset for future use
4. **Minimum 20 items**: For better UMAP results

## Troubleshooting

- **API key error**: Check `.env` file in project root
- **Dataset exists**: Use `delete` first or choose a different name
- **No active dataset**: Run `create` or `activate` first
- **Port in use**: Try `run --port 3000` or another port

