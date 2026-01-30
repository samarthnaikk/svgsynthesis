# Handwriting Synthesis - Function Documentation

This document provides comprehensive documentation for all functions in the handwriting synthesis codebase.

---

## Table of Contents

1. [Main Module](#main-module)
2. [Configuration](#configuration)
3. [Data Frame Module](#data-frame-module)
4. [Drawing Module](#drawing-module)
5. [Hand Module](#hand-module)
6. [RNN Module](#rnn-module)
7. [TensorFlow Utilities](#tensorflow-utilities)
8. [Training Module](#training-module)

---

## Main Module

### main.py

This is the entry point for the handwriting synthesis system with demo examples.

**Main Script Execution:**
- Demonstrates various handwriting generation scenarios with different biases and styles
- Creates SVG outputs with synthesized handwriting
- Shows examples: usage demo, song lyrics (All Star, Downtown, Never Gonna Give You Up)

---

## Configuration

### config.py

Defines global configuration paths for the project.

**Path Variables:**
- `data_path`: Path to model data directory
- `processed_data_path`: Path to processed data
- `raw_data_path`: Path to raw data
- `ascii_data_path`: Path to ASCII data
- `checkpoint_path`: Path to model checkpoints
- `prediction_path`: Path to model predictions
- `style_path`: Path to style embeddings

---

## Data Frame Module

### handwriting_synthesis/data_frame/DataFrame.py

#### Class: `DataFrame`

A minimal pandas DataFrame analog for handling n-dimensional numpy matrices with support for shuffling, batching, and train/test splitting.

**Methods:**

##### `__init__(self, columns, data)`
Initializes the DataFrame with column names and data matrices.
- **Parameters:**
  - `columns`: List of column names corresponding to data matrices
  - `data`: List of n-dimensional numpy arrays (all must have same first dimension)
- **Purpose:** Creates a data container optimized for batch processing

##### `shapes(self)`
Returns a pandas Series showing the shape of each data matrix.
- **Returns:** Series mapping column names to matrix shapes
- **Purpose:** Quick inspection of data dimensions

##### `dtypes(self)`
Returns a pandas Series showing the data type of each matrix.
- **Returns:** Series mapping column names to numpy dtypes
- **Purpose:** Inspect data types for all columns

##### `shuffle(self)`
Randomly shuffles the order of samples in the DataFrame.
- **Purpose:** Randomizes data order for training

##### `train_test_split(self, train_size, random_state=np.random.randint(1000), stratify=None)`
Splits DataFrame into training and testing sets.
- **Parameters:**
  - `train_size`: Fraction or number of samples for training
  - `random_state`: Random seed for reproducibility
  - `stratify`: Array for stratified splitting
- **Returns:** Tuple of (train_df, test_df)
- **Purpose:** Creates train/test splits while maintaining data structure

##### `batch_generator(self, batch_size, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False)`
Generates batches of data for training.
- **Parameters:**
  - `batch_size`: Number of samples per batch
  - `shuffle`: Whether to shuffle data each epoch
  - `num_epochs`: Maximum number of epochs to generate
  - `allow_smaller_final_batch`: Whether to yield partial final batches
- **Yields:** DataFrame objects containing batch data
- **Purpose:** Provides efficient batch iteration for model training

##### `iterrows(self)`
Iterates over rows (samples) in the DataFrame.
- **Yields:** pandas Series for each sample
- **Purpose:** Row-by-row iteration

##### `mask(self, mask)`
Filters DataFrame using a boolean mask.
- **Parameters:**
  - `mask`: Boolean array for filtering
- **Returns:** New DataFrame with filtered data
- **Purpose:** Subset selection based on conditions

##### `concat(self, other_df)`
Concatenates another DataFrame along the sample axis.
- **Parameters:**
  - `other_df`: DataFrame to concatenate
- **Returns:** New concatenated DataFrame
- **Purpose:** Combines multiple DataFrames

##### `items(self)`
Returns an iterator over (column_name, data) pairs.
- **Returns:** Dictionary items iterator
- **Purpose:** Iterate over columns and their data

##### `__len__(self)`
Returns the number of samples in the DataFrame.
- **Returns:** Integer sample count
- **Purpose:** Get DataFrame size

##### `__getitem__(self, key)`
Accesses data by column name or row index.
- **Parameters:**
  - `key`: Column name (str) or row index (int)
- **Returns:** Data array or pandas Series
- **Purpose:** Data access interface

##### `__setitem__(self, key, value)`
Sets data for a column.
- **Parameters:**
  - `key`: Column name
  - `value`: Data array to set
- **Purpose:** Add or update columns

---

## Drawing Module

### handwriting_synthesis/drawing/operations.py

Contains utility functions for stroke processing, manipulation, and visualization.

**Global Variables:**
- `alphabet`: List of supported characters for handwriting synthesis
- `alpha_to_num`: Dictionary mapping characters to indices
- `num_to_alpha`: Dictionary mapping indices to ASCII codes
- `MAX_STROKE_LEN`: Maximum stroke length (1200)
- `MAX_CHAR_LEN`: Maximum character sequence length (75)

#### Functions:

##### `align(coords)`
Corrects for global slant and offset in handwriting strokes.
- **Parameters:**
  - `coords`: Numpy array of stroke coordinates
- **Returns:** Aligned coordinates
- **Purpose:** Normalizes handwriting by removing slant using linear regression

##### `skew(coords, degrees)`
Skews strokes by a specified angle.
- **Parameters:**
  - `coords`: Stroke coordinates
  - `degrees`: Angle in degrees to skew
- **Returns:** Skewed coordinates
- **Purpose:** Applies horizontal shear transformation

##### `stretch(coords, x_factor, y_factor)`
Stretches strokes along x and y axes.
- **Parameters:**
  - `coords`: Stroke coordinates
  - `x_factor`: Horizontal scaling factor
  - `y_factor`: Vertical scaling factor
- **Returns:** Stretched coordinates
- **Purpose:** Scales handwriting dimensions

##### `add_noise(coords, scale)`
Adds Gaussian noise to strokes.
- **Parameters:**
  - `coords`: Stroke coordinates
  - `scale`: Standard deviation of noise
- **Returns:** Noisy coordinates
- **Purpose:** Data augmentation for training robustness

##### `encode_ascii(ascii_string)`
Encodes ASCII string to array of integer indices.
- **Parameters:**
  - `ascii_string`: Text string to encode
- **Returns:** Numpy array of character indices
- **Purpose:** Converts text to numerical representation for model input

##### `denoise(coords)`
Applies Savitzky-Golay filter to smooth strokes.
- **Parameters:**
  - `coords`: Stroke coordinates
- **Returns:** Smoothed coordinates
- **Purpose:** Reduces noise artifacts from data collection

##### `interpolate(coords, factor=2)`
Interpolates strokes using cubic spline.
- **Parameters:**
  - `coords`: Stroke coordinates
  - `factor`: Interpolation factor (higher = more points)
- **Returns:** Interpolated coordinates
- **Purpose:** Increases stroke resolution for smoother rendering

##### `normalize(offsets)`
Normalizes strokes to median unit norm.
- **Parameters:**
  - `offsets`: Stroke offset vectors
- **Returns:** Normalized offsets
- **Purpose:** Standardizes stroke magnitudes for consistent scaling

##### `coords_to_offsets(coords)`
Converts coordinates to offset representation.
- **Parameters:**
  - `coords`: Absolute coordinates
- **Returns:** Relative offset vectors
- **Purpose:** Transforms from absolute to relative positions (model-friendly)

##### `offsets_to_coords(offsets)`
Converts offsets back to coordinates.
- **Parameters:**
  - `offsets`: Relative offset vectors
- **Returns:** Absolute coordinates
- **Purpose:** Reconstructs absolute positions from relative offsets

##### `draw(offsets, ascii_seq=None, align_strokes=True, denoise_strokes=True, interpolation_factor=None, save_file=None)`
Visualizes handwriting strokes using matplotlib.
- **Parameters:**
  - `offsets`: Stroke offsets to draw
  - `ascii_seq`: Optional text to display as title
  - `align_strokes`: Whether to align strokes
  - `denoise_strokes`: Whether to denoise
  - `interpolation_factor`: Interpolation factor (None = no interpolation)
  - `save_file`: Path to save figure (None = display)
- **Purpose:** Creates matplotlib visualization of handwriting

---

## Hand Module

### handwriting_synthesis/hand/Hand.py

#### Class: `Hand`

Main interface for generating handwritten text.

##### `__init__(self)`
Initializes the handwriting synthesis model.
- **Purpose:** Loads pre-trained RNN model, configures TensorFlow, restores checkpoint

##### `write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None)`
Generates handwritten text and saves to SVG file.
- **Parameters:**
  - `filename`: Output SVG file path
  - `lines`: List of text strings to write
  - `biases`: Optional list of bias values per line (controls randomness)
  - `styles`: Optional list of style indices per line (0-12 available)
  - `stroke_colors`: Optional list of colors per line
  - `stroke_widths`: Optional list of stroke widths per line
- **Purpose:** Main API for generating handwritten SVG output
- **Validation:** Ensures lines are ≤75 chars and contain only valid characters

##### `_sample(self, lines, biases=None, styles=None)`
Generates stroke sequences from text input (internal method).
- **Parameters:**
  - `lines`: List of text strings
  - `biases`: Bias values for sampling
  - `styles`: Style indices for priming
- **Returns:** List of stroke arrays
- **Purpose:** Runs RNN inference to generate handwriting strokes

### handwriting_synthesis/hand/_draw.py

##### `_draw(strokes, lines, filename, stroke_colors=None, stroke_widths=None)`
Renders strokes to SVG file (internal function).
- **Parameters:**
  - `strokes`: List of stroke arrays
  - `lines`: Text strings for validation
  - `filename`: Output SVG path
  - `stroke_colors`: Colors per line
  - `stroke_widths`: Widths per line
- **Purpose:** Creates SVG file with handwriting paths, handles alignment and denoising

---

## RNN Module

### handwriting_synthesis/rnn/RNN.py

#### Class: `RNN`

Recurrent Neural Network model for handwriting synthesis (extends BaseModel).

##### `__init__(self, lstm_size, output_mixture_components, attention_mixture_components, **kwargs)`
Initializes RNN model architecture.
- **Parameters:**
  - `lstm_size`: Number of LSTM hidden units
  - `output_mixture_components`: Number of Gaussian mixture components for stroke prediction
  - `attention_mixture_components`: Number of mixture components for attention mechanism
  - `**kwargs`: Additional parameters passed to BaseModel
- **Purpose:** Configures model hyperparameters and architecture

##### `parse_parameters(self, z, eps=1e-8, sigma_eps=1e-4)`
Parses raw network outputs into mixture density network parameters.
- **Parameters:**
  - `z`: Raw network output tensor
  - `eps`: Small epsilon for numerical stability
  - `sigma_eps`: Minimum sigma value
- **Returns:** Tuple of (pis, mus, sigmas, rhos, es) - mixture parameters
- **Purpose:** Converts network outputs to probabilistic parameters with constraints

##### `nll(y, lengths, pis, mus, sigmas, rho, es, eps=1e-8)` (static method)
Calculates negative log-likelihood loss.
- **Parameters:**
  - `y`: Target stroke data
  - `lengths`: Sequence lengths
  - `pis`, `mus`, `sigmas`, `rho`: Gaussian mixture parameters
  - `es`: End-of-stroke probabilities
  - `eps`: Numerical stability constant
- **Returns:** Tuple of (sequence_loss, element_loss)
- **Purpose:** Computes loss for mixture density network training

##### `sample(self, cell)`
Generates handwriting from scratch (unconditional sampling).
- **Parameters:**
  - `cell`: LSTM attention cell
- **Returns:** Sampled stroke sequences
- **Purpose:** Free-form generation without style priming

##### `primed_sample(self, cell)`
Generates handwriting with style priming.
- **Parameters:**
  - `cell`: LSTM attention cell
- **Returns:** Sampled stroke sequences
- **Purpose:** Style-conditioned generation using prime strokes

##### `calculate_loss(self)`
Builds computational graph and defines loss function.
- **Returns:** Loss tensor
- **Purpose:** Creates TensorFlow graph for training/inference

### handwriting_synthesis/rnn/LSTMAttentionCell.py

#### Class: `LSTMAttentionCell`

Custom LSTM cell with soft attention mechanism (extends RNNCell).

##### `__init__(self, lstm_size, num_attn_mixture_components, attention_values, attention_values_lengths, num_output_mixture_components, bias, reuse=None)`
Initializes attention-based LSTM cell.
- **Parameters:**
  - `lstm_size`: Hidden state size
  - `num_attn_mixture_components`: Number of attention mixture components
  - `attention_values`: Character sequence to attend to
  - `attention_values_lengths`: Lengths of character sequences
  - `num_output_mixture_components`: Output mixture components
  - `bias`: Sampling bias
  - `reuse`: Variable reuse flag
- **Purpose:** Sets up 3-layer LSTM with attention mechanism

##### `state_size` (property)
Returns the structure and sizes of cell state.
- **Returns:** LSTMAttentionCellState namedtuple
- **Purpose:** Defines state structure for TensorFlow

##### `output_size` (property)
Returns the size of cell output.
- **Returns:** LSTM size integer
- **Purpose:** Specifies output dimensionality

##### `zero_state(self, batch_size, dtype)`
Creates initial zero state.
- **Parameters:**
  - `batch_size`: Batch size
  - `dtype`: Data type
- **Returns:** Zero-initialized state
- **Purpose:** Provides initial state for sequence generation

##### `__call__(self, inputs, state, scope=None)`
Performs one step of LSTM with attention.
- **Parameters:**
  - `inputs`: Input tensor for current timestep
  - `state`: Previous cell state
  - `scope`: Variable scope
- **Returns:** Tuple of (output, new_state)
- **Purpose:** Core cell computation with 3 LSTM layers and attention window

##### `output_function(self, state)`
Samples from mixture density network.
- **Parameters:**
  - `state`: Current cell state
- **Returns:** Sampled stroke point (x, y, end_of_stroke)
- **Purpose:** Generates output strokes during inference

##### `termination_condition(self, state)`
Determines when to stop sequence generation.
- **Parameters:**
  - `state`: Current cell state
- **Returns:** Boolean tensor indicating termination
- **Purpose:** Stops generation when reaching end of text or EOS signal

##### `_parse_parameters(self, gmm_params, eps=1e-8, sigma_eps=1e-4)`
Parses GMM parameters with bias adjustment (internal method).
- **Parameters:**
  - `gmm_params`: Raw mixture parameters
  - `eps`, `sigma_eps`: Stability constants
- **Returns:** Parsed mixture parameters
- **Purpose:** Applies bias and constraints to parameters

### handwriting_synthesis/rnn/operations.py

#### Functions:

##### `_concat(prefix, suffix, static=False)`
Concatenates tensor dimensions (internal helper).
- **Parameters:**
  - `prefix`: First dimension
  - `suffix`: Second dimension
  - `static`: Whether to use static shapes
- **Returns:** Concatenated dimension
- **Purpose:** Helper for shape manipulation

##### `raw_rnn(cell, loop_fn, parallel_iterations=None, swap_memory=False, scope=None)`
Custom RNN implementation with arbitrary nested states.
- **Parameters:**
  - `cell`: RNN cell
  - `loop_fn`: Function controlling loop logic
  - `parallel_iterations`: Parallel execution count
  - `swap_memory`: Whether to swap GPU/CPU memory
  - `scope`: Variable scope
- **Returns:** Tuple of (states, outputs, final_state)
- **Purpose:** Enhanced RNN that emits full state at each timestep (adapted from TensorFlow)

##### `rnn_teacher_force(inputs, cell, sequence_length, initial_state, scope='dynamic-rnn-teacher-force')`
Teacher-forced RNN execution for training.
- **Parameters:**
  - `inputs`: Input sequences
  - `cell`: RNN cell
  - `sequence_length`: Lengths of sequences
  - `initial_state`: Initial cell state
  - `scope`: Variable scope
- **Returns:** Tuple of (states, outputs, final_state)
- **Purpose:** Trains RNN using ground truth inputs at each step

##### `rnn_free_run(cell, initial_state, sequence_length, initial_input=None, scope='dynamic-rnn-free-run')`
Free-running RNN for generation/inference.
- **Parameters:**
  - `cell`: RNN cell
  - `initial_state`: Initial cell state
  - `sequence_length`: Maximum sequence length
  - `initial_input`: Optional first input
  - `scope`: Variable scope
- **Returns:** Tuple of (states, outputs, final_state)
- **Purpose:** Generates sequences by feeding outputs back as inputs

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
