# Handwriting Synthesis - Comprehensive Documentation

This document provides complete documentation for all functions, classes, modules, and APIs in the handwriting synthesis codebase.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Configuration & Environment](#configuration--environment)
5. [Usage Examples](#usage-examples)
6. [Main Module](#main-module)
7. [Data Frame Module](#data-frame-module)
8. [Drawing Module](#drawing-module)
9. [Hand Module](#hand-module)
10. [RNN Module](#rnn-module)
11. [TensorFlow Utilities](#tensorflow-utilities)
12. [Training Module](#training-module)
13. [Troubleshooting](#troubleshooting)

---

## Overview

The Handwriting Synthesis project is a **macOS-compatible implementation** of handwriting generation using Recurrent Neural Networks (RNNs) with attention mechanisms. This system implements the research from Alex Graves' paper ["Generating Sequences with Recurrent Neural Networks"](https://arxiv.org/abs/1308.0850).

### Key Features

* **Handwriting Generation**: Converts text strings into realistic handwritten SVG images
* **Style Control**: Supports 13 different handwriting styles (0-12)
* **Bias Adjustment**: Control randomness/creativity in generation
* **Customization**: Configurable stroke colors and widths
* **Pre-trained Model**: Includes trained weights for immediate use
* **macOS Optimized**: Works seamlessly on Intel and Apple Silicon Macs

### Project Purpose

This system enables:
- Generating realistic handwritten text programmatically
- Creating personalized handwriting styles
- Data augmentation for handwriting recognition systems
- Artistic and creative applications

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Application                        │
│                     (main.py, custom)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│                   Hand Interface (API)                       │
│              handwriting_synthesis.Hand                      │
│         - write() method (main entry point)                  │
└────────────┬────────────────────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────────────────────┐
│                  RNN Generation Engine                       │
│   - LSTMAttentionCell: 3-layer LSTM with attention          │
│   - Mixture Density Network: Stroke prediction              │
│   - Style Priming: Condition on handwriting styles          │
└────────────┬────────────────────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────────────────────┐
│              Stroke Processing Pipeline                      │
│   - Alignment, denoising, interpolation                     │
│   - Coordinate transformations                              │
│   - SVG rendering                                           │
└─────────────────────────────────────────────────────────────┘
```

### Component Overview

1. **Hand Module**: High-level API for text-to-handwriting conversion
2. **RNN Module**: Neural network architecture for sequence generation
3. **Drawing Module**: Stroke manipulation and SVG rendering utilities
4. **Training Module**: Data loading and model training infrastructure
5. **TensorFlow Utilities**: Custom layers and helpers for model construction
6. **Data Frame Module**: Lightweight data container for batch processing

### Data Flow

```
Text Input → Encoding → RNN with Attention → Stroke Sequences → 
Processing (align, denoise) → SVG Rendering → Output File
```

---

## Installation & Setup

### System Requirements

* **Operating System**: macOS 11+ (Big Sur or later)
* **Python**: 3.8, 3.9, or 3.10
* **Hardware**: 
  - Intel or Apple Silicon processor
  - Minimum 4GB RAM (8GB recommended for training)
  - 500MB disk space for model and dependencies

### Dependencies

The project requires the following Python packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | 2.15.0 | Neural network framework |
| `tensorflow-probability` | 0.20.1 | Mixture density networks |
| `matplotlib` | ≥2.1.0 | Visualization and plotting |
| `pandas` | ≥0.22.0 | Data manipulation |
| `scikit-learn` | ≥0.19.1 | Train/test splitting |
| `scipy` | ≥1.0.0 | Signal processing (interpolation, filtering) |
| `svgwrite` | ≥1.1.12 | SVG file generation |
| `numpy` | (via dependencies) | Numerical computations |

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/samarthnaikk/svgsynthesis.git
   cd svgsynthesis
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python main.py
   ```

This should generate several SVG files in the `img/` directory.

### Pre-trained Model

The repository includes a pre-trained model located in the `model/` directory:
- **Checkpoint**: `model/checkpoint/`
- **Style embeddings**: `model/style/`

No additional downloads are required for basic usage.

---

## Configuration & Environment

### Configuration File

**Location**: `handwriting_synthesis/config.py`

This module defines all global paths used throughout the project.

**Configuration Variables**:

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `BASE_PATH` | `"model"` | Root directory for model files |
| `BASE_DATA_PATH` | `"data"` | Root directory for datasets |
| `data_path` | `"model/data"` | Combined model data path |
| `processed_data_path` | `"model/data/processed"` | Preprocessed training data |
| `raw_data_path` | `"model/data/raw"` | Original IAM dataset |
| `ascii_data_path` | `"model/data/raw/ascii"` | Text transcriptions |
| `checkpoint_path` | `"model/checkpoint"` | Saved model weights |
| `prediction_path` | `"model/prediction"` | Model prediction outputs |
| `style_path` | `"model/style"` | Style embedding vectors |

**Customization**: To change paths, modify the variables in `config.py` before importing any modules.

### Environment Variables

The project uses one environment variable internally:

| Variable | Value | Purpose | Set By |
|----------|-------|---------|--------|
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Suppresses TensorFlow info/warning logs | `Hand.__init__()` |

**Note**: This is set automatically and does not require user configuration.

### Required Files

For basic usage (generation only):
- `model/checkpoint/`: Pre-trained model checkpoint
- `model/style/`: Style embedding files

For training from scratch:
- `model/data/raw/`: IAM On-Line Handwriting Database
- `model/data/raw/ascii/`: Corresponding text transcriptions

---

## Usage Examples

### Basic Usage

Generate handwritten text with default settings:

```python
from handwriting_synthesis import Hand

hand = Hand()
hand.write(
    filename='output.svg',
    lines=['Hello, World!']
)
```

### Custom Styling

Use specific handwriting styles and biases:

```python
from handwriting_synthesis import Hand

hand = Hand()

lines = [
    "The quick brown fox",
    "jumps over the lazy dog"
]

# Style 9, moderate randomness
biases = [0.75, 0.75]
styles = [9, 9]

hand.write(
    filename='styled_output.svg',
    lines=lines,
    biases=biases,
    styles=styles
)
```

### Multiple Styles with Colors

Create multi-line output with different colors per line:

```python
from handwriting_synthesis import Hand

hand = Hand()

lines = [
    "Line one in red",
    "Line two in blue",
    "Line three in green",
    "Line four in black"
]

biases = [0.5, 0.75, 1.0, 0.5]
styles = [0, 3, 7, 12]
stroke_colors = ['red', 'blue', 'green', 'black']
stroke_widths = [1, 2, 1, 3]

hand.write(
    filename='colorful_output.svg',
    lines=lines,
    biases=biases,
    styles=styles,
    stroke_colors=stroke_colors,
    stroke_widths=stroke_widths
)
```

### Understanding Parameters

**Biases**: Controls randomness (0.0 to 1.5 typical range)
- Lower values (0.1-0.5): More consistent, less creative
- Medium values (0.5-0.8): Balanced
- Higher values (0.8-1.5): More variation, creative

**Styles**: Select handwriting style (integers 0-12)
- Each style represents a different handwriting personality
- Styles are learned from the training data
- Experiment to find preferred styles

**Character Limits**:
- Maximum line length: **75 characters**
- Lines exceeding this will raise a `ValueError`

**Supported Characters**:
```
Space, !, ", #, ', (, ), ,, -, ., 0-9, :, ;, ?,
A-Z (except Q, X, Z), a-z
```

### Drawing Strokes Directly

For custom visualization of stroke data:

```python
from handwriting_synthesis.drawing.operations import draw
import numpy as np

# offsets: numpy array of shape (N, 3) where columns are [dx, dy, pen_up]
offsets = np.load('my_strokes.npy')

draw(
    offsets=offsets,
    ascii_seq='Custom text',
    align_strokes=True,
    denoise_strokes=True,
    interpolation_factor=2,
    save_file='visualization.png'
)
```

---

## Main Module

### main.py

**File Location**: `/main.py`

The main entry point script demonstrating handwriting synthesis capabilities through multiple examples.

**Purpose**: Provides runnable demonstrations of the handwriting synthesis system with various configurations.

**Script Execution**:

When run directly (`python main.py`), generates four SVG examples:

1. **Usage Demo** (`img/usage_demo.svg`)
   - Multi-line text with varying colors and stroke widths
   - Demonstrates all customization options
   - Style: 9, Bias: 0.75 for all lines

2. **All Star Lyrics** (`img/all_star.svg`)
   - Fixed style (12) and bias (0.75)
   - Demonstrates consistent handwriting across multiple lines

3. **Downtown Lyrics** (`img/downtown.svg`)
   - Fixed bias (0.75), varying styles
   - Shows style changes between paragraphs

4. **Never Gonna Give You Up** (`img/give_up.svg`)
   - Varying bias (decreasing), fixed style (7)
   - Demonstrates bias effects on handwriting consistency

**Usage**:
```bash
python main.py
```

**Output**: Creates SVG files in the `img/` directory (created automatically if missing).

**Dependencies**:
- `numpy`: Array operations
- `handwriting_synthesis.Hand`: Main handwriting generation class

**Notes**:
- Requires pre-trained model in `model/checkpoint/`
- All examples use text within character limits (≤75 chars per line)
- Demonstrates best practices for parameter selection

---

## Data Frame Module

### handwriting_synthesis/data_frame/DataFrame.py

**File Location**: `/handwriting_synthesis/data_frame/DataFrame.py`

#### Class: `DataFrame`

A minimal pandas DataFrame analog designed for handling n-dimensional numpy matrices with support for shuffling, batching, and train/test splitting.

**Purpose**: Provides efficient data container optimized for machine learning batch processing with multiple synchronized arrays.

**Design Philosophy**: 
- Lightweight alternative to pandas for ML workflows
- All data arrays must share the same first dimension (sample axis)
- Maintains synchronization across all columns during operations

---

#### Methods:

##### `__init__(self, columns, data)`

Initializes the DataFrame with column names and data matrices.

**Parameters:**
- `columns` (list of str): Column names corresponding to data matrices
- `data` (list of numpy.ndarray): N-dimensional arrays (all must have same first dimension)

**Returns**: None

**Raises**: 
- `AssertionError`: If number of columns doesn't match number of data arrays
- `AssertionError`: If first dimensions of data arrays don't match

**Purpose**: Creates a data container optimized for batch processing in neural network training.

**Example**:
```python
from handwriting_synthesis.data_frame import DataFrame
import numpy as np

X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.randint(0, 2, 100)  # 100 labels

df = DataFrame(columns=['features', 'labels'], data=[X, y])
```

---

##### `shapes(self)`

Returns a pandas Series showing the shape of each data matrix.

**Parameters**: None

**Returns**: `pandas.Series` mapping column names to matrix shapes

**Purpose**: Quick inspection of data dimensions for debugging and validation.

**Example**:
```python
print(df.shapes())
# features    (100, 10)
# labels      (100,)
```

---

##### `dtypes(self)`

Returns a pandas Series showing the data type of each matrix.

**Parameters**: None

**Returns**: `pandas.Series` mapping column names to numpy dtypes

**Purpose**: Inspect data types for all columns to ensure compatibility.

**Example**:
```python
print(df.dtypes())
# features    float64
# labels      int64
```

---

##### `shuffle(self)`

Randomly shuffles the order of samples in the DataFrame.

**Parameters**: None

**Returns**: None (in-place operation)

**Purpose**: Randomizes data order for training to prevent learning artifacts from data ordering.

**Example**:
```python
df.shuffle()  # All columns shuffled with same permutation
```

---

##### `train_test_split(self, train_size, random_state=np.random.randint(1000), stratify=None)`

Splits DataFrame into training and testing sets.

**Parameters:**
- `train_size` (float or int): 
  - If float (0.0-1.0): Fraction of samples for training
  - If int: Absolute number of samples for training
- `random_state` (int, optional): Random seed for reproducibility. Default: random integer
- `stratify` (array-like, optional): Array for stratified splitting (ensures balanced classes)

**Returns**: Tuple of `(train_df, test_df)` - Two DataFrame objects

**Purpose**: Creates train/test splits while maintaining data structure and synchronization.

**Example**:
```python
train_df, test_df = df.train_test_split(train_size=0.8, random_state=42)
print(len(train_df), len(test_df))  # 80, 20
```

---

##### `batch_generator(self, batch_size, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False)`

Generates batches of data for training.

**Parameters:**
- `batch_size` (int): Number of samples per batch
- `shuffle` (bool, optional): Whether to shuffle data each epoch. Default: `True`
- `num_epochs` (int, optional): Maximum number of epochs to generate. Default: `10000`
- `allow_smaller_final_batch` (bool, optional): Whether to yield partial final batches. Default: `False`

**Yields**: `DataFrame` objects containing batch data

**Purpose**: Provides efficient batch iteration for model training with automatic epoch management.

**Example**:
```python
for batch in df.batch_generator(batch_size=32, num_epochs=10):
    X_batch = batch['features']
    y_batch = batch['labels']
    # Train model on batch
```

---

##### `iterrows(self)`

Iterates over rows (samples) in the DataFrame.

**Parameters**: None

**Yields**: `pandas.Series` for each sample with column names as index

**Purpose**: Row-by-row iteration for processing individual samples.

**Example**:
```python
for row in df.iterrows():
    features = row['features']
    label = row['labels']
    # Process individual sample
```

---

##### `mask(self, mask)`

Filters DataFrame using a boolean mask.

**Parameters:**
- `mask` (numpy.ndarray of bool): Boolean array for filtering (same length as DataFrame)

**Returns**: New `DataFrame` with filtered data

**Purpose**: Subset selection based on conditions without modifying original.

**Example**:
```python
# Select positive samples
positive_df = df.mask(df['labels'] == 1)
```

---

##### `concat(self, other_df)`

Concatenates another DataFrame along the sample axis.

**Parameters:**
- `other_df` (DataFrame): DataFrame to concatenate (must have same columns)

**Returns**: New concatenated `DataFrame`

**Purpose**: Combines multiple DataFrames vertically (stacking samples).

**Example**:
```python
combined_df = train_df.concat(val_df)
```

---

##### `items(self)`

Returns an iterator over (column_name, data) pairs.

**Parameters**: None

**Returns**: Dictionary items iterator

**Purpose**: Iterate over columns and their data arrays.

**Example**:
```python
for col_name, col_data in df.items():
    print(f"{col_name}: {col_data.shape}")
```

---

##### `__iter__(self)`

Makes DataFrame iterable over (column_name, data) pairs.

**Parameters**: None

**Returns**: Iterator over dictionary items

**Purpose**: Allows using DataFrame in for-loops directly.

**Example**:
```python
for col_name, col_data in df:
    print(f"{col_name}: {col_data.shape}")
```

---

##### `__len__(self)`

Returns the number of samples in the DataFrame.

**Parameters**: None

**Returns**: `int` - Sample count

**Purpose**: Get DataFrame size using `len()` function.

**Example**:
```python
num_samples = len(df)  # 100
```

---

##### `__getitem__(self, key)`

Accesses data by column name or row index.

**Parameters:**
- `key` (str or int): 
  - If `str`: Column name - returns entire data array
  - If `int`: Row index - returns pandas Series for that sample

**Returns**: 
- `numpy.ndarray` if key is column name
- `pandas.Series` if key is row index

**Purpose**: Provides dictionary-like and list-like data access interface.

**Example**:
```python
# Access column
features = df['features']  # Returns full array

# Access row
sample = df[0]  # Returns Series with all columns for first sample
```

---

##### `__setitem__(self, key, value)`

Sets data for a column.

**Parameters:**
- `key` (str): Column name
- `value` (numpy.ndarray): Data array to set (first dimension must match DataFrame length)

**Returns**: None

**Purpose**: Add or update columns with dictionary-like syntax.

**Example**:
```python
df['predictions'] = model.predict(df['features'])
```

---

## Drawing Module

### handwriting_synthesis/drawing/operations.py

**File Location**: `/handwriting_synthesis/drawing/operations.py`

Contains utility functions for stroke processing, manipulation, visualization, and encoding/decoding operations.

---

#### Module-Level Constants

##### Character Encoding Constants

**`alphabet`** (list of str)  
Complete set of supported characters for handwriting synthesis:
```python
['\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
 '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
 'y', 'z']
```
**Purpose**: Defines valid input characters (Note: Missing Q, X, Z in uppercase)

**`alphabet_ord`** (list of int)  
ASCII codes for all characters in `alphabet`.
```python
alphabet_ord = list(map(ord, alphabet))
```
**Purpose**: Integer representation of valid characters for encoding operations.

**`alpha_to_num`** (defaultdict)  
Dictionary mapping characters to their indices in the alphabet.
```python
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))
```
**Purpose**: Fast lookup for encoding characters to indices. Returns 0 (null character) for unsupported characters.

**`num_to_alpha`** (dict)  
Dictionary mapping indices to ASCII codes.
```python
num_to_alpha = dict(enumerate(alphabet_ord))
```
**Purpose**: Decoding indices back to character codes.

##### Length Constraints

**`MAX_STROKE_LEN`** (int) = `1200`  
Maximum number of stroke points in a sequence.
**Purpose**: Upper bound for stroke sequence length to prevent memory issues.

**`MAX_CHAR_LEN`** (int) = `75`  
Maximum number of characters in a text line.
**Purpose**: Enforced limit on input text length for generation.

---

#### Functions:

##### `align(coords)`

Corrects for global slant and offset in handwriting strokes using linear regression.

**Parameters:**
- `coords` (numpy.ndarray): Stroke coordinates of shape (N, 2) where columns are [x, y]

**Returns**: `numpy.ndarray` - Aligned coordinates with same shape

**Purpose**: Normalizes handwriting by removing slant. Uses linear regression to find the dominant writing angle and rotates strokes to be vertical.

**Algorithm**:
1. Compute linear regression slope from coordinates
2. Calculate rotation angle from slope
3. Apply rotation transformation to remove slant

**Example**:
```python
from handwriting_synthesis.drawing.operations import align
import numpy as np

slanted_coords = np.array([[0, 0], [1, 1], [2, 2]])
straight_coords = align(slanted_coords)
```

---

##### `skew(coords, degrees)`

Skews strokes by a specified angle using horizontal shear transformation.

**Parameters:**
- `coords` (numpy.ndarray): Stroke coordinates of shape (N, 2)
- `degrees` (float): Angle in degrees to skew (positive = right lean)

**Returns**: `numpy.ndarray` - Skewed coordinates with same shape

**Purpose**: Applies horizontal shear transformation to create italic or slanted handwriting effect.

**Transform Matrix**:
```
| 1   tan(θ) |
| 0      1   |
```

**Example**:
```python
from handwriting_synthesis.drawing.operations import skew

# Create 15-degree right slant
italic_coords = skew(coords, 15)
```

---

##### `stretch(coords, x_factor, y_factor)`

Stretches strokes along x and y axes.

**Parameters:**
- `coords` (numpy.ndarray): Stroke coordinates of shape (N, 2)
- `x_factor` (float): Horizontal scaling factor (1.0 = no change)
- `y_factor` (float): Vertical scaling factor (1.0 = no change)

**Returns**: `numpy.ndarray` - Stretched coordinates with same shape

**Purpose**: Scales handwriting dimensions independently along each axis.

**Example**:
```python
from handwriting_synthesis.drawing.operations import stretch

# Make handwriting wider and shorter
stretched_coords = stretch(coords, x_factor=1.5, y_factor=0.8)
```

---

##### `add_noise(coords, scale)`

Adds Gaussian noise to strokes for data augmentation.

**Parameters:**
- `coords` (numpy.ndarray): Stroke coordinates of shape (N, 2)
- `scale` (float): Standard deviation of noise distribution

**Returns**: `numpy.ndarray` - Noisy coordinates with same shape

**Purpose**: Data augmentation for training robustness by adding random perturbations.

**Noise Model**: Gaussian with mean=0, std=scale, applied independently to x and y.

**Example**:
```python
from handwriting_synthesis.drawing.operations import add_noise

# Add small random perturbations
noisy_coords = add_noise(coords, scale=0.1)
```

---

##### `encode_ascii(ascii_string)`

Encodes ASCII string to array of integer indices.

**Parameters:**
- `ascii_string` (str): Text string to encode

**Returns**: `numpy.ndarray` - Integer array of character indices (dtype: int32)

**Purpose**: Converts text to numerical representation for model input. Unsupported characters map to index 0 (null).

**Example**:
```python
from handwriting_synthesis.drawing.operations import encode_ascii

text = "Hello, World!"
encoded = encode_ascii(text)
# Returns array of indices corresponding to each character
```

---

##### `denoise(coords)`

Applies Savitzky-Golay filter to smooth strokes.

**Parameters:**
- `coords` (numpy.ndarray): Stroke coordinates of shape (N, 2)

**Returns**: `numpy.ndarray` - Smoothed coordinates with same shape

**Purpose**: Reduces noise artifacts from data collection using polynomial smoothing filter.

**Filter Parameters**:
- Window length: 15 points
- Polynomial order: 3

**Note**: Uses `scipy.signal.savgol_filter`

**Example**:
```python
from handwriting_synthesis.drawing.operations import denoise

smoothed_coords = denoise(noisy_coords)
```

---

##### `interpolate(coords, factor=2)`

Interpolates strokes using cubic spline for higher resolution.

**Parameters:**
- `coords` (numpy.ndarray): Stroke coordinates of shape (N, 2)
- `factor` (int, optional): Interpolation factor. Default: 2
  - factor=2: Doubles the number of points
  - factor=N: Multiplies points by N

**Returns**: `numpy.ndarray` - Interpolated coordinates of shape (N*factor, 2)

**Purpose**: Increases stroke resolution for smoother rendering and more detailed representation.

**Method**: Cubic spline interpolation using `scipy.interpolate.interp1d`

**Example**:
```python
from handwriting_synthesis.drawing.operations import interpolate

# Double the resolution
high_res_coords = interpolate(coords, factor=2)
```

---

##### `normalize(offsets)`

Normalizes strokes to median unit norm.

**Parameters:**
- `offsets` (numpy.ndarray): Stroke offset vectors of shape (N, 2)

**Returns**: `numpy.ndarray` - Normalized offsets with same shape

**Purpose**: Standardizes stroke magnitudes for consistent scaling across different handwriting samples.

**Algorithm**:
1. Compute norm of each offset vector
2. Calculate median norm across all vectors
3. Scale all offsets by (1 / median_norm)

**Example**:
```python
from handwriting_synthesis.drawing.operations import normalize

normalized_offsets = normalize(raw_offsets)
```

---

##### `coords_to_offsets(coords)`

Converts absolute coordinates to relative offset representation.

**Parameters:**
- `coords` (numpy.ndarray): Absolute coordinates of shape (N, 2)

**Returns**: `numpy.ndarray` - Relative offset vectors of shape (N, 2)

**Purpose**: Transforms from absolute to relative positions. This representation is more suitable for sequence models as it encodes local motion patterns.

**Transform**: `offsets[i] = coords[i] - coords[i-1]` (first offset is zero)

**Example**:
```python
from handwriting_synthesis.drawing.operations import coords_to_offsets

offsets = coords_to_offsets(absolute_coords)
```

---

##### `offsets_to_coords(offsets)`

Converts relative offsets back to absolute coordinates.

**Parameters:**
- `offsets` (numpy.ndarray): Relative offset vectors of shape (N, 2 or 3)

**Returns**: `numpy.ndarray` - Absolute coordinates of shape (N, 2)

**Purpose**: Reconstructs absolute positions from relative offsets (inverse of `coords_to_offsets`).

**Transform**: `coords[i] = sum(offsets[0:i+1])`

**Note**: If offsets have 3 columns, only first 2 (dx, dy) are used; third column (pen up/down) is ignored.

**Example**:
```python
from handwriting_synthesis.drawing.operations import offsets_to_coords

absolute_coords = offsets_to_coords(offset_vectors)
```

---

##### `draw(offsets, ascii_seq=None, align_strokes=True, denoise_strokes=True, interpolation_factor=None, save_file=None)`

Visualizes handwriting strokes using matplotlib.

**Parameters:**
- `offsets` (numpy.ndarray): Stroke offsets to draw, shape (N, 2 or 3)
- `ascii_seq` (str, optional): Text to display as title. Default: `None`
- `align_strokes` (bool, optional): Whether to align strokes. Default: `True`
- `denoise_strokes` (bool, optional): Whether to denoise strokes. Default: `True`
- `interpolation_factor` (int, optional): Interpolation factor (None = no interpolation). Default: `None`
- `save_file` (str, optional): Path to save figure (None = display interactively). Default: `None`

**Returns**: None

**Purpose**: Creates matplotlib visualization of handwriting with optional preprocessing and saving.

**Behavior**:
- Converts offsets to coordinates
- Applies alignment and denoising if requested
- Optionally interpolates for smoother appearance
- Handles pen-up strokes (draws disconnected segments)
- Saves to file or displays interactively

**Example**:
```python
from handwriting_synthesis.drawing.operations import draw

# Display interactively
draw(strokes, ascii_seq="Hello World", interpolation_factor=2)

# Save to file
draw(strokes, ascii_seq="Hello World", save_file="output.png", 
     align_strokes=True, denoise_strokes=True)
```

---

## Hand Module

### handwriting_synthesis/hand/Hand.py

**File Location**: `/handwriting_synthesis/hand/Hand.py`

#### Class: `Hand`

The primary user-facing interface for generating handwritten text. This class encapsulates the RNN model and provides a simple API for text-to-handwriting conversion.

**Purpose**: High-level API that abstracts away model complexity, allowing users to generate handwritten SVG output with a single method call.

---

#### Methods:

##### `__init__(self)`

Initializes the handwriting synthesis model and loads pre-trained weights.

**Parameters**: None

**Returns**: None

**Purpose**: 
- Configures TensorFlow environment (suppresses warnings)
- Loads RNN model architecture
- Restores pre-trained checkpoint from `model/checkpoint/`
- Initializes TensorFlow session

**Side Effects**:
- Sets `TF_CPP_MIN_LOG_LEVEL=2` environment variable
- Creates TensorFlow session and loads model weights

**Raises**:
- `FileNotFoundError`: If checkpoint files are missing
- `ValueError`: If model configuration is invalid

**Example**:
```python
from handwriting_synthesis import Hand

hand = Hand()  # Loads model and restores checkpoint
```

**Notes**:
- This operation can take a few seconds on first load
- Model is loaded once and reused for multiple write() calls

---

##### `write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None)`

Generates handwritten text and saves to SVG file. This is the main API method for handwriting generation.

**Parameters:**
- `filename` (str): Output SVG file path (absolute or relative)
- `lines` (list of str): Text strings to write (one per line)
- `biases` (list of float, optional): Bias values per line controlling randomness. Default: `None` (uses default bias)
  - Range: Typically 0.0-1.5
  - Higher values = more variation/creativity
- `styles` (list of int, optional): Style indices per line (0-12). Default: `None` (random style)
- `stroke_colors` (list of str, optional): CSS color names or hex codes per line. Default: `None` (black)
- `stroke_widths` (list of int, optional): Stroke width in pixels per line. Default: `None` (default width)

**Returns**: None

**Raises**:
- `ValueError`: If any line exceeds 75 characters
- `ValueError`: If any line contains unsupported characters
- `ValueError`: If parameter list lengths don't match `lines` length

**Purpose**: Main API for generating handwritten SVG output with full customization.

**Validation**:
- Ensures all lines are ≤75 characters
- Validates characters against supported alphabet
- Checks parameter list lengths match

**Example**:
```python
from handwriting_synthesis import Hand

hand = Hand()

lines = ["Hello, World!", "This is a test."]
biases = [0.75, 0.5]
styles = [9, 3]
stroke_colors = ['blue', 'red']
stroke_widths = [2, 1]

hand.write(
    filename='output.svg',
    lines=lines,
    biases=biases,
    styles=styles,
    stroke_colors=stroke_colors,
    stroke_widths=stroke_widths
)
```

**Processing Steps**:
1. Validate input parameters
2. Generate stroke sequences via `_sample()` method
3. Process strokes (alignment, denoising)
4. Render to SVG via `_draw()` function
5. Save to specified filename

---

##### `_sample(self, lines, biases=None, styles=None)`

Generates stroke sequences from text input using the RNN model (internal method).

**Parameters:**
- `lines` (list of str): Text strings to generate
- `biases` (list of float, optional): Bias values for sampling. Default: `None`
- `styles` (list of int, optional): Style indices for priming. Default: `None`

**Returns**: `list of numpy.ndarray` - List of stroke arrays, one per line
  - Each array has shape (N, 3) where columns are [dx, dy, pen_up]

**Purpose**: Runs RNN inference to generate handwriting strokes from text.

**Algorithm**:
1. Encode text to character indices
2. Load style embeddings if styles provided
3. Run RNN in free-running mode (autoregressive generation)
4. Sample from mixture density network outputs
5. Continue until end-of-sequence or max length

**Note**: This is an internal method not intended for direct use. Use `write()` instead.

---

### handwriting_synthesis/hand/_draw.py

**File Location**: `/handwriting_synthesis/hand/_draw.py`

##### `_draw(strokes, lines, filename, stroke_colors=None, stroke_widths=None)`

Renders stroke sequences to SVG file (internal function).

**Parameters:**
- `strokes` (list of numpy.ndarray): List of stroke arrays (one per line)
  - Each array shape: (N, 3) where columns are [dx, dy, pen_up]
- `lines` (list of str): Text strings for length validation
- `filename` (str): Output SVG file path
- `stroke_colors` (list of str, optional): CSS colors per line. Default: `None` (black)
- `stroke_widths` (list of int, optional): Stroke widths in pixels per line. Default: `None`

**Returns**: None

**Purpose**: Creates SVG file with handwriting paths, handles alignment, denoising, and layout.

**Processing Steps**:
1. Convert offsets to absolute coordinates
2. Apply alignment to remove slant
3. Apply denoising to smooth strokes
4. Calculate bounding boxes and layout
5. Generate SVG paths with pen-up handling
6. Write SVG file with proper viewBox and styling

**SVG Features**:
- Scalable vector graphics (resolution-independent)
- Customizable stroke colors and widths
- Proper handling of pen-up/down states
- Multi-line layout with spacing

**Note**: This is an internal function not intended for direct use. Use `Hand.write()` instead.

**Example Output**:
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="...">
  <path d="M..." stroke="black" stroke-width="2" fill="none"/>
  ...
</svg>
```

---

## RNN Module

### handwriting_synthesis/rnn/RNN.py

**File Location**: `/handwriting_synthesis/rnn/RNN.py`

#### Class: `RNN`

Recurrent Neural Network model for handwriting synthesis (extends `BaseModel`).

**Purpose**: Implements sequence-to-sequence model using LSTM with attention and mixture density networks for generating handwriting strokes from text.

**Architecture**:
- 3-layer LSTM with attention mechanism
- Mixture Density Network (MDN) for stroke prediction
- Soft attention over input character sequence
- Style conditioning via learned embeddings

---

#### Methods:

##### `__init__(self, lstm_size, output_mixture_components, attention_mixture_components, **kwargs)`

Initializes RNN model architecture.

**Parameters:**
- `lstm_size` (int): Number of LSTM hidden units per layer
- `output_mixture_components` (int): Number of Gaussian mixture components for stroke prediction (typically 20)
- `attention_mixture_components` (int): Number of mixture components for attention mechanism (typically 10)
- `**kwargs`: Additional parameters passed to `BaseModel` (learning rates, batch sizes, etc.)

**Returns**: None

**Purpose**: Configures model hyperparameters and architecture.

**Default Configuration** (from training):
```python
lstm_size = 400
output_mixture_components = 20
attention_mixture_components = 10
```

---

##### `parse_parameters(self, z, eps=1e-8, sigma_eps=1e-4)`

Parses raw network outputs into mixture density network parameters.

**Parameters:**
- `z` (tf.Tensor): Raw network output tensor of shape (batch, time, features)
- `eps` (float, optional): Small epsilon for numerical stability. Default: `1e-8`
- `sigma_eps` (float, optional): Minimum sigma value to prevent collapse. Default: `1e-4`

**Returns**: Tuple of:
- `pis` (tf.Tensor): Mixture weights (normalized via softmax)
- `mus` (tf.Tensor): Means of Gaussian components
- `sigmas` (tf.Tensor): Standard deviations (constrained positive via exp)
- `rhos` (tf.Tensor): Correlation coefficients (constrained to [-1, 1] via tanh)
- `es` (tf.Tensor): End-of-stroke probabilities (via sigmoid)

**Purpose**: Converts network outputs to probabilistic parameters with proper constraints.

**Constraints Applied**:
- Mixture weights: Softmax normalization
- Means: Unbounded
- Std deviations: exp() + eps
- Correlations: tanh() for [-1, 1] range
- End-of-stroke: sigmoid() for [0, 1] probability

---

##### `nll(y, lengths, pis, mus, sigmas, rho, es, eps=1e-8)` (static method)

Calculates negative log-likelihood loss for mixture density network.

**Parameters:**
- `y` (tf.Tensor): Target stroke data, shape (batch, time, 3)
- `lengths` (tf.Tensor): Sequence lengths, shape (batch,)
- `pis` (tf.Tensor): Mixture weights
- `mus` (tf.Tensor): Gaussian means
- `sigmas` (tf.Tensor): Gaussian standard deviations
- `rho` (tf.Tensor): Correlation coefficients
- `es` (tf.Tensor): End-of-stroke probabilities
- `eps` (float, optional): Numerical stability constant. Default: `1e-8`

**Returns**: Tuple of:
- `sequence_loss` (tf.Tensor): Average loss per sequence, shape (batch,)
- `element_loss` (tf.Tensor): Loss per timestep, shape (batch, time)

**Purpose**: Computes loss for mixture density network training using bivariate Gaussian mixtures.

**Loss Formula**: Negative log-likelihood of 2D Gaussian mixture + binary cross-entropy for end-of-stroke.

---

##### `sample(self, cell)`

Generates handwriting from scratch (unconditional sampling).

**Parameters:**
- `cell` (`LSTMAttentionCell`): LSTM attention cell for generation

**Returns**: Sampled stroke sequences (generated unconditionally)

**Purpose**: Free-form generation without style priming or text conditioning.

**Note**: Primarily used for testing; production uses `primed_sample()`.

---

##### `primed_sample(self, cell)`

Generates handwriting with style priming.

**Parameters:**
- `cell` (`LSTMAttentionCell`): LSTM attention cell for generation

**Returns**: Sampled stroke sequences conditioned on style

**Purpose**: Style-conditioned generation using prime strokes to initialize the model.

**Process**:
1. Load style embedding (prime strokes)
2. Initialize RNN state with style
3. Generate strokes autoregressively
4. Sample from mixture density network at each step

---

##### `calculate_loss(self)`

Builds computational graph and defines loss function.

**Parameters**: None (uses instance attributes)

**Returns**: `tf.Tensor` - Loss tensor for training

**Purpose**: Creates TensorFlow graph for training/inference including:
- Input placeholders
- RNN forward pass
- Mixture density network outputs
- Loss calculation

---

### handwriting_synthesis/rnn/LSTMAttentionCell.py

**File Location**: `/handwriting_synthesis/rnn/LSTMAttentionCell.py`

#### Named Tuple: `LSTMAttentionCellState`

State container for the LSTM attention cell.

**Definition**:
```python
LSTMAttentionCellState = namedtuple(
    'LSTMAttentionCellState',
    ['h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'alpha', 'beta', 'kappa', 'w', 'phi']
)
```

**Fields**:
- `h1`, `c1`: Hidden and cell states for LSTM layer 1
- `h2`, `c2`: Hidden and cell states for LSTM layer 2
- `h3`, `c3`: Hidden and cell states for LSTM layer 3
- `alpha`: Attention mixture weights
- `beta`: Attention mixture widths
- `kappa`: Attention mixture centers
- `w`: Attention window (weighted sum of character embeddings)
- `phi`: Attention weights over input characters

**Purpose**: Encapsulates the complete state of the 3-layer LSTM with attention mechanism.

---

#### Class: `LSTMAttentionCell`

Custom LSTM cell with soft attention mechanism (extends `tf.nn.rnn_cell.RNNCell`).

**Purpose**: Implements 3-layer LSTM with Gaussian mixture attention over input character sequence.

**Architecture**:
- 3 LSTM layers stacked
- Soft attention window over input text
- Mixture density network for output generation
- Supports style-conditioned generation via bias

---

#### Methods:

##### `__init__(self, lstm_size, num_attn_mixture_components, attention_values, attention_values_lengths, num_output_mixture_components, bias, reuse=None)`

Initializes attention-based LSTM cell.

**Parameters:**
- `lstm_size` (int): Hidden state size for each LSTM layer
- `num_attn_mixture_components` (int): Number of attention mixture components (typically 10)
- `attention_values` (tf.Tensor): Character sequence to attend to (batch, max_char_len, char_embedding_dim)
- `attention_values_lengths` (tf.Tensor): Lengths of character sequences (batch,)
- `num_output_mixture_components` (int): Output mixture components for stroke generation (typically 20)
- `bias` (float): Sampling bias for temperature control
- `reuse` (bool, optional): Variable reuse flag for TensorFlow. Default: `None`

**Returns**: None

**Purpose**: Sets up 3-layer LSTM with attention mechanism.

---

##### `state_size` (property)

Returns the structure and sizes of cell state.

**Parameters**: None (property)

**Returns**: `LSTMAttentionCellState` namedtuple with field sizes

**Purpose**: Defines state structure for TensorFlow RNN infrastructure.

---

##### `output_size` (property)

Returns the size of cell output.

**Parameters**: None (property)

**Returns**: `int` - LSTM size (hidden state dimension)

**Purpose**: Specifies output dimensionality for TensorFlow.

---

##### `zero_state(self, batch_size, dtype)`

Creates initial zero state for the cell.

**Parameters:**
- `batch_size` (int): Batch size for state initialization
- `dtype` (tf.DType): Data type for state tensors

**Returns**: `LSTMAttentionCellState` with zero-initialized tensors

**Purpose**: Provides initial state for sequence generation.

---

##### `__call__(self, inputs, state, scope=None)`

Performs one step of LSTM with attention (required by RNNCell interface).

**Parameters:**
- `inputs` (tf.Tensor): Input tensor for current timestep (batch, input_size)
- `state` (`LSTMAttentionCellState`): Previous cell state
- `scope` (str, optional): Variable scope for TensorFlow. Default: `None`

**Returns**: Tuple of:
- `output` (tf.Tensor): Cell output (hidden state of last LSTM layer)
- `new_state` (`LSTMAttentionCellState`): Updated cell state

**Purpose**: Core cell computation with 3 LSTM layers and attention window.

**Computation Flow**:
1. LSTM layer 1: Process input + attention window
2. LSTM layer 2: Process hidden state from layer 1
3. LSTM layer 3: Process hidden state from layer 2
4. Compute attention parameters (alpha, beta, kappa)
5. Calculate attention window over input characters
6. Output final hidden state and updated state

---

##### `output_function(self, state)`

Samples from mixture density network during generation.

**Parameters:**
- `state` (`LSTMAttentionCellState`): Current cell state

**Returns**: Sampled stroke point (x, y, end_of_stroke) as tf.Tensor

**Purpose**: Generates output strokes during inference by sampling from the predicted mixture distribution.

**Sampling Process**:
1. Compute mixture parameters from state
2. Select mixture component based on weights
3. Sample (x, y) from selected bivariate Gaussian
4. Sample end-of-stroke from Bernoulli distribution

---

##### `termination_condition(self, state)`

Determines when to stop sequence generation.

**Parameters:**
- `state` (`LSTMAttentionCellState`): Current cell state

**Returns**: `tf.Tensor` (bool) - Boolean tensor indicating termination

**Purpose**: Stops generation when:
- Attention reaches end of input text (kappa exceeds text length)
- End-of-stroke signal is strong
- Maximum sequence length reached

---

##### `_parse_parameters(self, gmm_params, eps=1e-8, sigma_eps=1e-4)`

Parses GMM parameters with bias adjustment (internal method).

**Parameters:**
- `gmm_params` (tf.Tensor): Raw mixture parameters
- `eps` (float, optional): Numerical stability constant. Default: `1e-8`
- `sigma_eps` (float, optional): Minimum sigma value. Default: `1e-4`

**Returns**: Parsed mixture parameters with applied bias

**Purpose**: Applies bias and constraints to parameters for controlled generation.

**Bias Effect**: Higher bias → more randomness; lower bias → more deterministic.

---

### handwriting_synthesis/rnn/operations.py

**File Location**: `/handwriting_synthesis/rnn/operations.py`

Custom RNN operations for training and inference.

---

#### Functions:

##### `_concat(prefix, suffix, static=False)`

Concatenates tensor dimensions (internal helper).

**Parameters:**
- `prefix`: First dimension value
- `suffix`: Second dimension value
- `static` (bool, optional): Whether to use static shapes. Default: `False`

**Returns**: Concatenated dimension

**Purpose**: Helper for shape manipulation in TensorFlow operations.

**Note**: Internal utility function.

---

##### `raw_rnn(cell, loop_fn, parallel_iterations=None, swap_memory=False, scope=None)`

Custom RNN implementation with arbitrary nested states.

**Parameters:**
- `cell` (RNNCell): RNN cell instance
- `loop_fn` (callable): Function controlling loop logic
- `parallel_iterations` (int, optional): Parallel execution count. Default: `None`
- `swap_memory` (bool, optional): Whether to swap GPU/CPU memory. Default: `False`
- `scope` (str, optional): Variable scope. Default: `None`

**Returns**: Tuple of:
- `states` (list): Cell states at each timestep
- `outputs` (list): Cell outputs at each timestep
- `final_state`: Final cell state

**Purpose**: Enhanced RNN that emits full state at each timestep (adapted from TensorFlow).

**Use Case**: Needed for custom attention mechanisms where full state history is required.

---

##### `rnn_teacher_force(inputs, cell, sequence_length, initial_state, scope='dynamic-rnn-teacher-force')`

Teacher-forced RNN execution for training.

**Parameters:**
- `inputs` (tf.Tensor): Input sequences (batch, time, features)
- `cell` (RNNCell): RNN cell instance
- `sequence_length` (tf.Tensor): Lengths of sequences (batch,)
- `initial_state`: Initial cell state
- `scope` (str, optional): Variable scope. Default: `'dynamic-rnn-teacher-force'`

**Returns**: Tuple of:
- `states`: Cell states at each timestep
- `outputs`: Cell outputs at each timestep
- `final_state`: Final cell state

**Purpose**: Trains RNN using ground truth inputs at each step.

**Teacher Forcing**: At timestep t, use ground truth input[t] (not predicted output[t-1]).

---

##### `rnn_free_run(cell, initial_state, sequence_length, initial_input=None, scope='dynamic-rnn-free-run')`

Free-running RNN for generation/inference.

**Parameters:**
- `cell` (RNNCell): RNN cell instance
- `initial_state`: Initial cell state
- `sequence_length` (int or tf.Tensor): Maximum sequence length to generate
- `initial_input` (tf.Tensor, optional): Optional first input. Default: `None` (uses cell's output_function)
- `scope` (str, optional): Variable scope. Default: `'dynamic-rnn-free-run'`

**Returns**: Tuple of:
- `states`: Cell states at each timestep
- `outputs`: Cell outputs at each timestep
- `final_state`: Final cell state

**Purpose**: Generates sequences by feeding outputs back as inputs (autoregressive).

**Autoregressive**: At timestep t, use predicted output[t-1] as input[t].

---

## TensorFlow Utilities

### handwriting_synthesis/tf/utils.py

#### Functions:

##### `dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None, dropout=None, scope='dense-layer', reuse=False)`
Creates a fully connected neural network layer.
- **Parameters:**
  - `inputs`: Input tensor [batch_size, input_units]
  - `output_units`: Number of output neurons
  - `bias`: Whether to add bias term
  - `activation`: Activation function
  - `batch_norm`: Whether to apply batch normalization
  - `dropout`: Dropout keep probability
  - `scope`: Variable scope name
  - `reuse`: Whether to reuse variables
- **Returns:** Output tensor [batch_size, output_units]
- **Purpose:** Standard dense layer with optional regularization

##### `time_distributed_dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None, dropout=None, scope='time-distributed-dense-layer', reuse=False)`
Applies dense layer independently to each timestep.
- **Parameters:**
  - `inputs`: Input tensor [batch_size, max_seq_len, input_units]
  - `output_units`: Number of output units
  - `bias`: Whether to add bias
  - `activation`: Activation function
  - `batch_norm`: Batch normalization flag
  - `dropout`: Dropout rate
  - `scope`: Variable scope
  - `reuse`: Variable reuse flag
- **Returns:** Output tensor [batch_size, max_seq_len, output_units]
- **Purpose:** Applies same dense layer across all timesteps (weight sharing)

##### `shape(tensor, dim=None)`
Gets tensor shape as list or specific dimension.
- **Parameters:**
  - `tensor`: TensorFlow tensor
  - `dim`: Optional dimension index
- **Returns:** Shape list or single dimension
- **Purpose:** Convenient shape extraction utility

##### `rank(tensor)`
Gets tensor rank (number of dimensions).
- **Parameters:**
  - `tensor`: TensorFlow tensor
- **Returns:** Integer rank
- **Purpose:** Returns dimensionality of tensor

### handwriting_synthesis/tf/BaseModel.py

#### Class: `BaseModel`

Base class for TensorFlow models with training loop, checkpointing, and inference.

##### `__init__(self, reader=None, batch_sizes=None, num_training_steps=20000, learning_rates=None, beta1_decays=None, optimizer='adam', grad_clip=5, regularization_constant=0.0, keep_prob=1.0, patiences=None, warm_start_init_step=0, enable_parameter_averaging=False, min_steps_to_checkpoint=100, log_interval=20, logging_level=logging.INFO, loss_averaging_window=100, validation_batch_size=64, log_dir='logs', checkpoint_dir=checkpoint_path, prediction_dir=prediction_path)`
Initializes base model with training configuration.
- **Parameters:** (extensive, see docstring)
  - Training parameters: batch sizes, learning rates, optimization settings
  - Regularization: grad clipping, dropout, L2
  - Early stopping: patience, restarts
  - Logging: intervals, directories
- **Purpose:** Sets up training infrastructure

##### `update_train_params(self)`
Updates training parameters for current restart.
- **Purpose:** Adjusts hyperparameters between training phases

##### `calculate_loss(self)`
Abstract method for loss calculation (must be implemented by subclasses).
- **Returns:** Loss tensor
- **Purpose:** Defines model-specific loss computation

##### `fit(self)`
Main training loop with validation and early stopping.
- **Purpose:** Trains model with automatic checkpointing, validation monitoring, and learning rate scheduling

##### `predict(self, chunk_size=256)`
Runs inference on test set and saves predictions.
- **Parameters:**
  - `chunk_size`: Batch size for inference
- **Purpose:** Generates and saves predictions to numpy files

##### `save(self, step, averaged=False)`
Saves model checkpoint.
- **Parameters:**
  - `step`: Training step number
  - `averaged`: Whether to save averaged parameters
- **Purpose:** Persists model weights to disk

##### `restore(self, step=None, averaged=False)`
Restores model from checkpoint.
- **Parameters:**
  - `step`: Step to restore (None = latest)
  - `averaged`: Whether to restore averaged parameters
- **Purpose:** Loads model weights from disk

##### `init_logging(self, log_dir)`
Sets up logging to file and console.
- **Parameters:**
  - `log_dir`: Directory for log files
- **Purpose:** Configures Python logging

##### `update_parameters(self, loss)`
Computes gradients and applies parameter updates.
- **Parameters:**
  - `loss`: Loss tensor
- **Purpose:** Defines optimization step with gradient clipping and regularization

##### `get_optimizer(self, learning_rate, beta1_decay)`
Creates TensorFlow optimizer.
- **Parameters:**
  - `learning_rate`: Learning rate
  - `beta1_decay`: Beta1 parameter for Adam/momentum
- **Returns:** TensorFlow optimizer
- **Purpose:** Instantiates Adam, SGD, or RMSProp optimizer

##### `build_graph(self)`
Constructs TensorFlow computational graph.
- **Returns:** TensorFlow graph
- **Purpose:** Builds complete graph with loss, optimizer, and savers

---

## Training Module

### handwriting_synthesis/training/batch_generator.py

##### `batch_generator(batch_size, df, shuffle=True, num_epochs=10000, mode='train')`
Generator that yields training batches.
- **Parameters:**
  - `batch_size`: Samples per batch
  - `df`: DataFrame containing data
  - `shuffle`: Whether to shuffle
  - `num_epochs`: Maximum epochs
  - `mode`: 'train', 'val', or 'test'
- **Yields:** Dictionary batches with keys: x, y, x_len, c, c_len
- **Purpose:** Prepares batches by trimming sequences and adjusting for teacher forcing

### handwriting_synthesis/training/DataReader.py

#### Class: `DataReader`

Handles data loading and batch generation for training.

##### `__init__(self, data_dir)`
Loads and splits data into train/validation/test sets.
- **Parameters:**
  - `data_dir`: Directory containing processed numpy arrays
- **Purpose:** Creates data readers with 95/5 train/val split

##### `train_batch_generator(self, batch_size)`
Creates training batch generator.
- **Parameters:**
  - `batch_size`: Batch size
- **Returns:** Batch generator
- **Purpose:** Provides shuffled training batches

##### `val_batch_generator(self, batch_size)`
Creates validation batch generator.
- **Parameters:**
  - `batch_size`: Batch size
- **Returns:** Batch generator
- **Purpose:** Provides shuffled validation batches

##### `test_batch_generator(self, batch_size)`
Creates test batch generator.
- **Parameters:**
  - `batch_size`: Batch size
- **Returns:** Batch generator
- **Purpose:** Provides unshuffled test batches (single epoch)

### handwriting_synthesis/training/train.py

##### `train()`
Main training entry point.
- **Purpose:** Configures and launches RNN training with specified hyperparameters

### handwriting_synthesis/training/preparation/prepare.py

##### `prepare()`
Preprocesses raw IAM dataset into numpy arrays.
- **Purpose:** Traverses data directory, extracts strokes and transcriptions, applies preprocessing, saves to processed directory

### handwriting_synthesis/training/preparation/operations.py

##### `get_stroke_sequence(filename)`
Extracts and preprocesses stroke data from XML file.
- **Parameters:**
  - `filename`: Path to XML stroke file
- **Returns:** Normalized stroke offsets
- **Purpose:** Parses XML, aligns, denoises, and normalizes strokes

##### `get_ascii_sequences(filename)`
Extracts text transcriptions from file.
- **Parameters:**
  - `filename`: Path to text file
- **Returns:** List of encoded text sequences
- **Purpose:** Parses and encodes text labels

##### `collect_data()`
Traverses IAM dataset and collects file paths and metadata.
- **Returns:** Tuple of (stroke_fnames, transcriptions, writer_ids)
- **Purpose:** Organizes dataset files, filters blacklist, matches strokes to text

---

## Summary

This codebase implements a sequence-to-sequence handwriting synthesis model using:
- **LSTM with Attention**: Generates handwriting strokes from text
- **Mixture Density Networks**: Models probabilistic stroke distributions
- **Style Priming**: Allows conditioning on specific handwriting styles
- **Data Processing Pipeline**: Handles IAM dataset preprocessing
- **SVG Rendering**: Creates scalable vector graphics output

The main user interface is the `Hand` class, which provides a simple `write()` method for generating handwritten text in various styles and with adjustable randomness.
