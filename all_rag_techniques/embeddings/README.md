# 🍝 3D Embeddings Visualization - CLI Dataset Manager

A powerful command-line tool for creating and managing multiple embedding datasets with persistent storage and easy visualization switching.

## 📑 Table of Contents

- [Overview](#overview)
- [📋 Command Cheat Sheet](#-command-cheat-sheet) ⭐ **Start here!**
- [Quick Start](#quick-start)
- [CLI Commands Reference](#cli-commands-reference)
- [JSON Input Format Specification](#json-input-format-specification)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Changelog](#changelog)

---

## Overview

This project provides a **CLI-based dataset manager** that separates the expensive embedding computation from visualization. You can create multiple datasets, switch between them, and visualize them in an interactive 3D space powered by three.js.

### Key Features

- **Multiple Dataset Support**: Create and manage as many datasets as you need
- **Persistent Storage**: All datasets are saved in `data_store/` for reuse
- **Easy Switching**: Activate any dataset instantly without recomputing
- **Flexible Input**: Standard JSON format for easy dataset creation
- **Category Color-Coding**: Automatic color assignment based on categories
- **Interactive 3D Visualization**: Navigate, search, and explore semantic relationships

---

## 📋 Command Cheat Sheet

Quick reference for all CLI commands:

```bash
# Create a new dataset from JSON file (~1-2 minutes for 50-100 items)
python prepare_data.py create <dataset_name> --file <path/to/input.json>

# List all available datasets (shows which one is active)
python prepare_data.py list

# Switch to a different dataset (instant, no recomputation)
python prepare_data.py activate <dataset_name>

# Delete a dataset (with confirmation prompt)
python prepare_data.py delete <dataset_name>

# Start the visualization server (uses active dataset)
python prepare_data.py run [--port 8000]
```

**Example Workflow:**
```bash
# 1. Create datasets
python prepare_data.py create recipes --file input_examples/ricette_italiane.json
python prepare_data.py create movies --file input_examples/movies.json

# 2. List and switch
python prepare_data.py list
python prepare_data.py activate movies

# 3. Visualize
python prepare_data.py run
# → Open http://localhost:8000
```

**Need Help?** Jump to [Detailed CLI Reference](#cli-commands-reference) or [JSON Format Guide](#json-input-format-specification)

---

## Quick Start

### 1. Install Dependencies

```bash
# Python dependencies
pip install google-generativeai umap-learn python-dotenv numpy

# Node.js dependencies (for three.js)
npm install
```

### 2. Configure API Key

Ensure your `GEMINI_API_KEY` is set in the `.env` file at the project root:

```bash
GEMINI_API_KEY=your_google_api_key_here
```

### 3. Create Your First Dataset

```bash
python prepare_data.py create ricette_italiane --file input_examples/ricette_italiane.json
```

This will:
- Compute embeddings for all items (~1 minute for 70 items)
- Reduce dimensionality to 3D using UMAP
- Save the dataset for future use
- Set it as the active dataset

### 4. Start the Visualization

```bash
python prepare_data.py run
```

Then open your browser at `http://localhost:8000`

## CLI Commands Reference

The CLI tool provides 5 commands for complete dataset lifecycle management. All commands follow the pattern:

```bash
python prepare_data.py <command> [arguments] [options]
```

---

### 📊 `create` - Create a New Dataset

**Purpose:** Creates a new dataset by computing embeddings and reducing dimensionality to 3D.

**Syntax:**
```bash
python prepare_data.py create <dataset_name> --file <path_to_json_file>
```

**Arguments:**
- `dataset_name` (required): Unique name for the dataset (alphanumeric, underscores, hyphens)
- `--file` (required): Path to the input JSON file containing items

**Examples:**
```bash
# Create dataset from example file
python prepare_data.py create ricette_italiane --file input_examples/ricette_italiane.json

# Create dataset from custom file
python prepare_data.py create movies_2023 --file ~/data/movies.json

# Create dataset from different directory
python prepare_data.py create tech_products --file /path/to/products.json
```

**Process Flow:**
1. **Validation**: Checks if dataset name already exists
2. **Loading**: Reads and parses the JSON file
3. **Schema Validation**: Verifies JSON structure matches required format
4. **Embedding Computation**: Calls Google Generative AI for each item (~1 second per item)
5. **Dimensionality Reduction**: Applies UMAP to reduce 768D embeddings to 3D
6. **Storage**: Saves enriched data to `data_store/<dataset_name>.json`
7. **Registration**: Updates `config.json` with dataset metadata
8. **Activation**: Sets as active if it's the first dataset

**Output Example:**
```
======================================================================
📊 Creating dataset: ricette_italiane
======================================================================

📂 Loading input file... ✓
🔍 Validating schema... ✓

📊 Computing embeddings (70 items)...
   [1/70] Spaghetti alla Carbonara
   [2/70] Bucatini all'Amatriciana
   ...
   [70/70] Colomba Pasquale

🔄 Reducing dimensionality (768D → 3D)...
   ✓ Dimensionality reduced successfully

💾 Saving dataset... ✓

======================================================================
✅ Dataset 'ricette_italiane' created successfully!
======================================================================
   Path: data_store/ricette_italiane.json
   Items: 70
   Language: it
   Status: Active (first dataset)
```

**Time Estimate:**
- Small dataset (20-50 items): ~1-2 minutes
- Medium dataset (50-100 items): ~2-3 minutes
- Large dataset (100-200 items): ~3-5 minutes

**Common Errors:**
- `Dataset already exists`: Use a different name or delete the existing one first
- `File not found`: Check the file path is correct
- `Validation errors`: See detailed error messages and fix JSON structure
- `API key not found`: Ensure `GEMINI_API_KEY` is in `.env` file

---

### 📁 `list` - List All Datasets

**Purpose:** Shows all available datasets with their metadata.

**Syntax:**
```bash
python prepare_data.py list
```

**Arguments:** None

**Examples:**
```bash
python prepare_data.py list
```

**Output Example:**
```
📁 Available Datasets:

   * ricette_italiane (70 items, it) Active
     movies_2023 (120 items, en)
     tech_products (85 items, en)
     
Total: 3 dataset(s)
```

**Output Details:**
- `*` marker: Indicates the currently active dataset
- `(N items)`: Number of items in the dataset
- `(language)`: Language code from dataset_info
- `Active`: Only shown for the active dataset

**Use Cases:**
- Check which datasets are available
- See which dataset is currently active
- Get item counts before activation
- Verify a dataset was created successfully

---

### ✅ `activate` - Switch Active Dataset

**Purpose:** Sets a different dataset as active for visualization without recomputing anything.

**Syntax:**
```bash
python prepare_data.py activate <dataset_name>
```

**Arguments:**
- `dataset_name` (required): Name of an existing dataset

**Examples:**
```bash
# Switch to a different dataset
python prepare_data.py activate movies_2023

# Switch back to recipes
python prepare_data.py activate ricette_italiane
```

**Output Example:**
```
✅ Dataset 'movies_2023' is now active
   Items: 120
   Description: Top movies of 2023 with genre classification
```

**What Happens:**
- Updates `config.json` to mark the dataset as active
- Next `run` command will use this dataset
- **No recomputation** - switching is instant!
- Previous active dataset remains in storage

**⚠️ Important:** If the visualization server is already running, you must **restart it** to see the changes:
```bash
# Stop the server (Ctrl+C), then restart:
python prepare_data.py run --port 8000
```

**Use Cases:**
- Quickly switch between visualizations
- Compare different datasets
- Demo multiple datasets without recalculating
- Test different categorizations of the same domain

**Common Errors:**
- `Dataset not found`: Check spelling with `list` command
- `Visualization not updating`: Restart the server after activating a new dataset

---

### 🗑️ `delete` - Remove a Dataset

**Purpose:** Permanently deletes a dataset from storage.

**Syntax:**
```bash
python prepare_data.py delete <dataset_name> [--force]
```

**Arguments:**
- `dataset_name` (required): Name of the dataset to delete
- `--force` (optional): Skip confirmation prompt

**Examples:**
```bash
# Delete with confirmation
python prepare_data.py delete old_dataset

# Delete without confirmation
python prepare_data.py delete temp_test --force
```

**Output Example (with confirmation):**
```
⚠️  You are about to delete dataset 'old_dataset'
   Items: 45
   Status: Active (currently in use)
   This action cannot be undone.
   
   Are you sure? [y/N]: y
   
🗑️  Dataset 'old_dataset' deleted successfully
   New active dataset: ricette_italiane
```

**What Happens:**
1. Verifies dataset exists
2. Shows confirmation prompt (unless `--force` is used)
3. Deletes `data_store/<dataset_name>.json`
4. Updates `config.json` to remove dataset entry
5. If deleting active dataset, automatically activates another one

**Safety Features:**
- Confirmation required by default
- Cannot be undone
- If deleting active dataset, automatically activates the next available one
- If deleting last dataset, sets active to `null`

**Use Cases:**
- Clean up old experiments
- Remove duplicate datasets
- Free disk space
- Remove test datasets

**Common Errors:**
- `Dataset not found`: Check name with `list` command

---

### 🚀 `run` - Start Visualization Server

**Purpose:** Starts an HTTP server to visualize the active dataset in 3D.

**Syntax:**
```bash
python prepare_data.py run [--port PORT]
```

**Arguments:**
- `--port` (optional): Port number for the server (default: 8000)

**Examples:**
```bash
# Start on default port 8000
python prepare_data.py run

# Start on custom port
python prepare_data.py run --port 3000

# Start on port 8080 (alternative HTTP port)
python prepare_data.py run --port 8080
```

**Output Example:**
```
======================================================================
🚀 Starting visualization server...
======================================================================

   Active dataset: ricette_italiane
   Items: 70
   Description: 70 ricette italiane iconiche organizzate per categoria

📋 Copying to public/data.json... ✓

🌐 Server running at: http://localhost:8000

   Press Ctrl+C to stop

Serving HTTP on :: port 8000 (http://[::]:8000/) ...
```

**What Happens:**
1. Checks if an active dataset exists
2. Copies `data_store/<active_dataset>.json` to `public/data.json`
3. Starts Python HTTP server on specified port
4. Serves all files in the `embeddings/` directory
5. Waits for Ctrl+C to stop

**Accessing the Visualization:**
1. Open your browser
2. Navigate to `http://localhost:8000` (or your custom port)
3. The 3D visualization will load automatically

**Visualization Controls:**
- **Mouse Drag**: Rotate the 3D view
- **Mouse Scroll**: Zoom in/out
- **Mouse Hover**: See item details in tooltip (name, category, description, metadata)
- **Search Box**: Type to filter items by name
- **Click on Point**: Highlights selected item (red) and 5 nearest neighbors (green)
- **Click on Background**: Resets to category colors

**Common Errors:**
- `No active dataset`: Run `create` or `activate` first
- `Port already in use`: Use a different port with `--port`
- `Permission denied`: Use a port > 1024 (ports 1-1023 require root)

**Stopping the Server:**
- Press `Ctrl+C` in the terminal
- Server will shut down gracefully

---

### 🔧 Command Chaining Examples

**Complete Workflow:**
```bash
# 1. Create a new dataset
python prepare_data.py create my_data --file data.json

# 2. List to verify
python prepare_data.py list

# 3. Visualize
python prepare_data.py run
```

**Managing Multiple Datasets:**
```bash
# Create multiple datasets
python prepare_data.py create dataset_v1 --file v1.json
python prepare_data.py create dataset_v2 --file v2.json
python prepare_data.py create dataset_v3 --file v3.json

# List all
python prepare_data.py list

# Switch and visualize different versions
python prepare_data.py activate dataset_v2
python prepare_data.py run

# Clean up old version
python prepare_data.py delete dataset_v1 --force
```

**Batch Operations:**
```bash
# Create multiple datasets from a directory
for file in input_examples/*.json; do
    name=$(basename "$file" .json)
    python prepare_data.py create "$name" --file "$file"
done

# List all created datasets
python prepare_data.py list
```

---

## JSON Input Format Specification

### Overview

This section provides a **complete specification** for creating input JSON files. This format is designed to be used by AI systems (LLMs, code generators) to automatically generate new datasets.

**Key Requirements:**
- Minimum **20 items** recommended (UMAP works better with more data)
- Maximum **500 items** per dataset (for reasonable processing time)
- All fields must use **UTF-8 encoding**
- JSON must be **valid** and parseable

---

### Complete JSON Structure

```json
{
  "dataset_info": {
    "language": "LANGUAGE_CODE",
    "description": "DATASET_DESCRIPTION"
  },
  "items": [
    {
      "name": {
        "local": "LOCAL_NAME",
        "en": "ENGLISH_NAME"
      },
      "category": "CATEGORY_ID",
      "description": "DETAILED_DESCRIPTION",
      "metadata": {
        "CUSTOM_FIELD_1": "VALUE_1",
        "CUSTOM_FIELD_2": "VALUE_2"
      }
    }
  ]
}
```

---

### Field-by-Field Specification

#### 1. `dataset_info` Object (Required)

Contains metadata about the entire dataset.

| Field | Type | Required | Description | Examples |
|-------|------|----------|-------------|----------|
| `language` | string | **Yes** | ISO 639-1 language code (2 letters) | `"it"`, `"en"`, `"fr"`, `"es"`, `"de"`, `"ja"` |
| `description` | string | **Yes** | Brief description of what the dataset contains (1-200 characters) | `"70 famous Italian recipes by category"`, `"Top 100 movies of 2023"` |

**Example:**
```json
"dataset_info": {
  "language": "en",
  "description": "Most popular programming languages with their use cases"
}
```

---

#### 2. `items` Array (Required)

Array of objects, each representing one item to visualize. Minimum 20 items recommended.

---

#### 3. `name` Object (Required for each item)

Contains the name of the item in multiple languages.

| Field | Type | Required | Description | Examples |
|-------|------|----------|-------------|----------|
| `local` | string | **Yes** | Name in the dataset's original language | `"Spaghetti alla Carbonara"`, `"培根意粉"`, `"Espaguetis Carbonara"` |
| `en` | string | **Yes** | English translation or transliteration | `"Carbonara Spaghetti"`, `"Bacon Spaghetti"`, `"Carbonara Spaghetti"` |

**Notes:**
- If the original language is English, `local` and `en` can be the same
- Transliteration is acceptable if direct translation doesn't exist
- Both fields should be informative and descriptive

**Examples:**
```json
// Italian recipe
"name": {
  "local": "Tiramisù",
  "en": "Tiramisu"
}

// Japanese dish
"name": {
  "local": "寿司",
  "en": "Sushi"
}

// English movie (same in both)
"name": {
  "local": "The Godfather",
  "en": "The Godfather"
}

// French wine
"name": {
  "local": "Château Margaux",
  "en": "Château Margaux" 
}
```

---

#### 4. `category` String (Required for each item)

A short identifier used for grouping and color-coding items.

**Requirements:**
- Lowercase letters, underscores allowed
- No spaces (use underscores instead)
- Consistent within the dataset
- Between 3-30 characters
- Descriptive but concise

**Recommendations:**
- Use 5-10 distinct categories per dataset
- Make categories meaningful for semantic grouping
- Categories should represent logical divisions

**Examples by Domain:**

**Food/Recipes:**
```json
"category": "primi"           // First courses
"category": "secondi_carne"   // Meat mains
"category": "secondi_pesce"   // Fish mains
"category": "contorni"        // Side dishes
"category": "dolci"           // Desserts
"category": "antipasti"       // Appetizers
```

**Movies:**
```json
"category": "action"
"category": "comedy"
"category": "drama"
"category": "scifi"
"category": "horror"
"category": "romance"
```

**Technology:**
```json
"category": "programming_language"
"category": "framework"
"category": "database"
"category": "cloud_service"
"category": "dev_tool"
```

**Books:**
```json
"category": "fiction"
"category": "non_fiction"
"category": "biography"
"category": "science"
"category": "history"
```

---

#### 5. `description` String (Required for each item)

Detailed description used for computing semantic embeddings. **This is the most important field for quality**.

**Requirements:**
- Minimum: 20 characters
- Recommended: 50-200 characters
- Maximum: 500 characters (longer descriptions are truncated in tooltips)
- Should be informative and specific

**Quality Guidelines:**
- Be descriptive and factual
- Include key characteristics
- Mention distinguishing features
- Avoid generic text
- Use complete sentences

**Good Examples:**
```json
// Recipe
"description": "Classic Roman pasta dish made with eggs, Pecorino Romano cheese, guanciale (cured pork cheek), and black pepper. The heat of the pasta cooks the eggs creating a creamy sauce."

// Movie
"description": "Epic crime drama following the Corleone family's rise and near fall within organized crime, directed by Francis Ford Coppola. Explores themes of power, family loyalty, and the American Dream."

// Programming Language
"description": "High-level, interpreted programming language known for its simple syntax and readability. Widely used for web development, data science, machine learning, and automation scripts."

// City
"description": "Capital city of France, known for iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Cultural center renowned for art, fashion, gastronomy, and historical significance."
```

**Bad Examples (Avoid):**
```json
// Too short
"description": "A pasta dish"

// Too generic
"description": "This is a movie"

// Not informative
"description": "Popular item"

// Just keywords
"description": "pasta, eggs, cheese, Italian"
```

---

#### 6. `metadata` Object (Optional for each item)

Custom fields with additional information. These fields:
- Are **preserved** in the output
- Appear in the **tooltip** on hover
- Do **NOT** affect embeddings
- Can be any valid JSON types

**Recommendations:**
- Use 2-6 metadata fields per item
- Keep keys short and descriptive
- Use snake_case for keys
- Values can be strings, numbers, or booleans

**Common Metadata Fields by Domain:**

**Recipes:**
```json
"metadata": {
  "region": "Lazio",
  "difficulty": "medium",
  "prep_time_minutes": 30,
  "vegetarian": false,
  "calories_per_serving": 450
}
```

**Movies:**
```json
"metadata": {
  "director": "Francis Ford Coppola",
  "year": 1972,
  "runtime_minutes": 175,
  "rating": "R",
  "imdb_score": 9.2
}
```

**Programming Languages:**
```json
"metadata": {
  "paradigm": "object_oriented",
  "typing": "dynamic",
  "year_created": 1991,
  "difficulty": "beginner_friendly",
  "popularity_rank": 1
}
```

**Books:**
```json
"metadata": {
  "author": "George Orwell",
  "year_published": 1949,
  "pages": 328,
  "genre": "dystopian_fiction",
  "awards": "Prometheus Hall of Fame Award"
}
```

**Products:**
```json
"metadata": {
  "brand": "Apple",
  "price_usd": 999,
  "release_year": 2023,
  "category": "electronics",
  "rating": 4.5
}
```

---

### Complete Examples

#### Example 1: Italian Recipes (Original)

```json
{
  "dataset_info": {
    "language": "it",
    "description": "70 ricette italiane iconiche organizzate per categoria"
  },
  "items": [
    {
      "name": {
        "local": "Spaghetti alla Carbonara",
        "en": "Carbonara Spaghetti"
      },
      "category": "primi",
      "description": "Piatto tipico laziale a base di uova, guanciale, pecorino romano e pepe nero. La pasta viene mantecata con le uova creando una crema senza panna.",
      "metadata": {
        "region": "Lazio",
        "difficulty": "medium",
        "prep_time": "20 minutes"
      }
    },
    {
      "name": {
        "local": "Tiramisù",
        "en": "Tiramisu"
      },
      "category": "dolci",
      "description": "Dolce al cucchiaio con savoiardi imbevuti nel caffè, mascarpone, uova e cacao amaro. Dessert cremoso e delicato tipico del Veneto.",
      "metadata": {
        "region": "Veneto",
        "difficulty": "easy",
        "prep_time": "30 minutes"
      }
    }
  ]
}
```

#### Example 2: Programming Languages

```json
{
  "dataset_info": {
    "language": "en",
    "description": "Popular programming languages with their characteristics and use cases"
  },
  "items": [
    {
      "name": {
        "local": "Python",
        "en": "Python"
      },
      "category": "interpreted",
      "description": "High-level, interpreted language with emphasis on code readability and simplicity. Widely used in data science, machine learning, web development, automation, and scientific computing.",
      "metadata": {
        "year_created": 1991,
        "typing": "dynamic",
        "difficulty": "beginner_friendly",
        "paradigm": "multi_paradigm"
      }
    },
    {
      "name": {
        "local": "Rust",
        "en": "Rust"
      },
      "category": "compiled",
      "description": "Systems programming language focused on safety, speed, and concurrency. Features a unique ownership system that guarantees memory safety without garbage collection.",
      "metadata": {
        "year_created": 2010,
        "typing": "static",
        "difficulty": "advanced",
        "paradigm": "multi_paradigm"
      }
    }
  ]
}
```

#### Example 3: Movies

```json
{
  "dataset_info": {
    "language": "en",
    "description": "Critically acclaimed films from various genres and eras"
  },
  "items": [
    {
      "name": {
        "local": "The Shawshank Redemption",
        "en": "The Shawshank Redemption"
      },
      "category": "drama",
      "description": "Drama about hope and friendship between two life-sentenced prisoners. Follows Andy Dufresne's journey in Shawshank prison over two decades, exploring themes of resilience and redemption.",
      "metadata": {
        "director": "Frank Darabont",
        "year": 1994,
        "runtime": 142,
        "rating": "R",
        "imdb_score": 9.3
      }
    },
    {
      "name": {
        "local": "Spirited Away",
        "en": "Spirited Away"
      },
      "category": "animation",
      "description": "Japanese animated fantasy about a young girl who enters a magical world of spirits and must find a way to save her parents. Explores themes of courage, identity, and environmentalism.",
      "metadata": {
        "director": "Hayao Miyazaki",
        "year": 2001,
        "runtime": 125,
        "rating": "PG",
        "studio": "Studio Ghibli"
      }
    }
  ]
}
```

---

### Embedding Computation Strategy

**How the Description is Used:**

When computing embeddings, the system creates a composite text from each item:

```python
composite_text = f"{name.local} ({name.en}) - Categoria: {category}. {description}"
```

**Example Composite Texts:**

```
"Spaghetti alla Carbonara (Carbonara Spaghetti) - Categoria: primi. Piatto tipico laziale a base di uova, guanciale, pecorino romano e pepe nero."

"Python (Python) - Categoria: interpreted. High-level, interpreted language with emphasis on code readability and simplicity. Widely used in data science..."

"The Shawshank Redemption (The Shawshank Redemption) - Categoria: drama. Drama about hope and friendship between two life-sentenced prisoners..."
```

**Why This Matters:**
- The embedding captures semantic meaning from **all three components**
- Items with similar names, categories, or descriptions will cluster together in 3D space
- Write descriptions that emphasize the **unique characteristics** of each item

---

### Validation Rules

The system automatically validates:

✅ **Required Fields:**
- `dataset_info` exists
- `dataset_info.language` exists and is a string
- `dataset_info.description` exists and is a string
- `items` exists and is an array
- Each item has `name`, `category`, `description`
- Each `name` has both `local` and `en` fields

❌ **Common Validation Errors:**

```json
// Missing required field
{
  "dataset_info": {
    "language": "en"
    // ❌ Missing "description"
  }
}

// Missing name.en
{
  "name": {
    "local": "Pasta"
    // ❌ Missing "en" field
  }
}

// Empty items array
{
  "items": []  // ❌ No items provided
}

// Wrong type
{
  "items": "not an array"  // ❌ Must be an array
}
```

---

### Tips for AI-Generated Datasets

When using AI (LLMs) to generate datasets:

1. **Be Specific in Prompts:**
   ```
   Generate a JSON file with 50 programming languages, including their
   paradigms, typical use cases, and difficulty levels. Use categories:
   compiled, interpreted, functional, and scripting.
   ```

2. **Request Diverse Categories:**
   - Aim for 5-10 categories
   - Ensure even distribution (not 90% in one category)

3. **Emphasize Description Quality:**
   ```
   Each description should be 100-150 characters, focusing on unique
   features, typical use cases, and key characteristics that distinguish
   it from similar items.
   ```

4. **Include Rich Metadata:**
   ```
   Add metadata fields: year_created, difficulty_level, popularity_rank,
   and primary_use_case for each language.
   ```

5. **Validate JSON:**
   - Use a JSON validator before creating the dataset
   - Check for missing commas, quotes, brackets

6. **Test with Sample:**
   - Generate 5-10 items first
   - Create the dataset and visualize
   - If clustering looks good, generate the full set

---

### Category Color Customization

**Default Colors (for Italian recipes):**

| Category | Color | Hex |
|----------|-------|-----|
| `primi` | Gold | #FFD700 |
| `secondi_carne` | Tomato | #FF6347 |
| `secondi_pesce` | Royal Blue | #4169E1 |
| `contorni` | Lime Green | #32CD32 |
| `pizza` | Dark Orange | #FF8C00 |
| `antipasti` | Medium Purple | #9370DB |
| `dolci` | Hot Pink | #FF69B4 |
| *other* | Orange | #ff6b35 |

**To Add Custom Categories:**

Edit `js/main.js`, find `CATEGORY_COLORS` and add your categories:

```javascript
const CATEGORY_COLORS = {
    // Add your custom categories here
    action: 0xFF0000,         // Red for action movies
    comedy: 0xFFFF00,         // Yellow for comedy
    drama: 0x0000FF,          // Blue for drama
    scifi: 0x00FFFF,          // Cyan for sci-fi
    horror: 0x8B008B,         // Dark magenta for horror
    
    // Keep existing ones or remove them
    primi: 0xFFD700,
    // ... etc
};
```

---

### Checklist for Creating a New Dataset

- [ ] JSON is valid and parseable
- [ ] `dataset_info` has `language` and `description`
- [ ] At least 20 items in `items` array
- [ ] Each item has `name` with `local` and `en`
- [ ] Each item has a `category` (lowercase, no spaces)
- [ ] Each item has a meaningful `description` (50-200 chars)
- [ ] Metadata is optional but recommended
- [ ] Categories are consistent across items
- [ ] Descriptions are unique and informative
- [ ] File is saved with `.json` extension
- [ ] UTF-8 encoding is used

---

---

## Project Structure

```
embeddings/
├── prepare_data.py                    # CLI tool
├── data_store/                        # Persistent storage
│   ├── config.json                   # Dataset registry & active selection
│   └── <dataset_name>.json           # Processed datasets with 3D positions
├── input_examples/                    # Example input files
│   └── ricette_italiane.json         # 70 Italian recipes
├── public/
│   └── data.json                     # Active dataset (copied on 'run')
├── js/
│   └── main.js                       # Three.js visualization logic
├── index.html                         # Web interface
├── package.json                       # npm dependencies
└── README.md                          # This file
```

---

## Architecture

### 1. Data Flow

```
Input JSON → Embedding API → UMAP → Enriched JSON → data_store/
                                                           ↓
                                                      config.json
                                                           ↓
                                              (on 'run' command)
                                                           ↓
                                                   public/data.json
                                                           ↓
                                                   Three.js Frontend
```

### 2. Config Management

The `data_store/config.json` file maintains:
- List of all datasets with metadata
- Which dataset is currently active
- Creation timestamps and item counts

Example:
```json
{
  "active_dataset": "ricette_italiane",
  "datasets": {
    "ricette_italiane": {
      "path": "data_store/ricette_italiane.json",
      "description": "70 ricette italiane iconiche",
      "created_at": "2025-10-29T18:45:00",
      "item_count": 70,
      "language": "it"
    }
  }
}
```

### 3. Dataset Storage

Each dataset in `data_store/` contains:
- Original input data (names, categories, descriptions, metadata)
- Computed 3D positions from UMAP
- All data needed for visualization

This means you can switch between datasets instantly without recomputing embeddings.

---

## Examples

### Example 1: Italian Recipes (Included)

```bash
# Create the dataset
python prepare_data.py create ricette_italiane --file input_examples/ricette_italiane.json

# Visualize it
python prepare_data.py run
```

### Example 2: Create Your Own Dataset

1. Create a JSON file following the schema (e.g., `my_movies.json`):

```json
{
  "dataset_info": {
    "language": "en",
    "description": "Top 50 movies of all time"
  },
  "items": [
    {
      "name": {
        "local": "The Shawshank Redemption",
        "en": "The Shawshank Redemption"
      },
      "category": "drama",
      "description": "Two imprisoned men bond over years, finding solace and redemption",
      "metadata": {
        "year": 1994,
        "director": "Frank Darabont"
      }
    }
  ]
}
```

2. Create the dataset:

```bash
python prepare_data.py create movies --file my_movies.json
```

3. Run the visualization:

```bash
python prepare_data.py run
```

### Example 3: Managing Multiple Datasets

```bash
# Create multiple datasets
python prepare_data.py create ricette_italiane --file input_examples/ricette_italiane.json
python prepare_data.py create movies --file input_examples/movies.json

# List them all
python prepare_data.py list

# Switch between them - the visualization automatically updates!
python prepare_data.py activate movies
python prepare_data.py run
# → Title, categories, and colors all update automatically

# Switch back to recipes
python prepare_data.py activate ricette_italiane
python prepare_data.py run
# → Everything updates again to match the new dataset

# Clean up old ones
python prepare_data.py delete dataset1 --force
```

**Important:** When you switch datasets and run the visualization:
- ✅ The **page title** updates with the new dataset's description
- ✅ The **category legend** updates with the new dataset's categories
- ✅ The **colors** are automatically assigned to the new categories
- ✅ The **tooltips** show the correct category labels

No manual configuration needed!

---

## Technical Details

### Embeddings
- **Model**: Google Generative AI `models/embedding-001`
- **Dimensionality**: 768 dimensions (original)
- **Task type**: `retrieval_document`

### Dimensionality Reduction
- **Algorithm**: UMAP (Uniform Manifold Approximation and Projection)
- **Target dimensions**: 3 (for 3D visualization)
- **Metric**: Cosine similarity
- **Parameters**:
  - `n_neighbors=15` (adjusted automatically for small datasets)
  - `min_dist=0.1`
  - `random_state=42` (for reproducibility)

### Visualization
- **Framework**: three.js (WebGL-based 3D library)
- **Point size**: 0.08 radius spheres
- **Controls**: OrbitControls for navigation
- **Rendering**: WebGL with antialiasing
- **Interactivity**: Raycasting for hover and click detection
- **Dynamic UI**: Title and category legend automatically update when switching datasets
- **Color Assignment**: Categories are automatically assigned colors from a 15-color palette

---

## Troubleshooting

### "GEMINI_API_KEY not found"
- Ensure the `.env` file exists in the project root (`RAG_Techniques/.env`)
- Verify the key is correctly formatted: `GEMINI_API_KEY=your_key_here`

### "Dataset already exists"
- Use `python prepare_data.py list` to see existing datasets
- Delete the old one: `python prepare_data.py delete <name>`
- Or choose a different name

### "No active dataset"
- Create a dataset: `python prepare_data.py create <name> --file <file>`
- Or activate an existing one: `python prepare_data.py activate <name>`

### "Validation errors" on create
- Check that your JSON matches the required schema
- Ensure all required fields are present (name.local, name.en, category, description)
- Use `input_examples/ricette_italiane.json` as a reference

### UMAP errors
- Ensure you have at least 10 items in your dataset
- UMAP requires a minimum number of samples to work properly

### Server won't start
- Check if another process is using the port: `lsof -i :8000`
- Use a different port: `python prepare_data.py run --port 3000`

---

## Advanced Usage

### Dynamic Category Colors

**Categories are now automatically handled!** The system:

1. **Extracts** all unique categories from your dataset
2. **Assigns** colors automatically from a 15-color palette
3. **Updates** the legend dynamically based on the active dataset

**You don't need to modify any code** - just use any category names in your JSON input.

**Custom Category Display Names:**

If you want prettier display names for your categories (e.g., "Sci-Fi" instead of "scifi"), you can add them to the `getCategoryLabel()` function in `js/main.js`:

```javascript
function getCategoryLabel(category) {
    const categoryLabels = {
        // Add your custom mappings here
        your_category: 'Your Category Display Name',
        scifi: 'Sci-Fi',
        // ...
    };
    
    return categoryLabels[category] || 
           category.charAt(0).toUpperCase() + category.slice(1).replace(/_/g, ' ');
}
```

If no custom mapping is found, the system automatically capitalizes and formats the category name (e.g., `action_movie` → "Action Movie").

### Batch Processing

Create multiple datasets programmatically:

```bash
for file in input_examples/*.json; do
    name=$(basename "$file" .json)
    python prepare_data.py create "$name" --file "$file"
done
```

### Export Dataset Info

```bash
python prepare_data.py list > datasets_inventory.txt
```

---

## Contributing

When creating example datasets, please:
1. Include at least 20 items (for better UMAP results)
2. Provide meaningful descriptions (used for embedding quality)
3. Use consistent category names
4. Add metadata fields for future filtering

---

## Changelog

### Version 2.0 - Dynamic Dataset Support (2025-10-29)

**New Features:**
- ✨ **Dynamic title**: Page title automatically updates with dataset description
- ✨ **Dynamic categories**: Legend automatically populates with dataset categories
- ✨ **Automatic color assignment**: Categories get colors from a 15-color palette
- ✨ **Smart category labels**: Automatic capitalization and formatting of category names
- ✨ **Multi-dataset support**: Switch between datasets without manual configuration

**Technical Changes:**
- Refactored `CATEGORY_COLORS` from static object to dynamic mapping
- Added `extractCategoriesAndColors()` function for automatic category extraction
- Added `updateLegend()` function for dynamic legend population
- Added `updateTitle()` function for dynamic page title
- Added `getCategoryLabel()` function for consistent category naming
- Updated tooltip to use dynamic category labels

**Breaking Changes:**
- None! The system is backward compatible with existing datasets.

---

## License

MIT License - Feel free to use and modify for your own projects.

---

## Acknowledgments

- **Google Generative AI** for embedding computation
- **UMAP** for dimensionality reduction
- **three.js** for 3D visualization
- Italian cuisine for inspiring the example dataset 🇮🇹
