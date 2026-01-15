# CASAS Aruba Smart Home - Process Mining Analysis

## 📋 Project Overview

This project implements a complete **process mining pipeline** for analyzing the CASAS Aruba smart home dataset. The solution transforms sensor event data into process models using pm4py, revealing patterns in daily activities through various discovery algorithms.

## 🎯 Objectives

1. Transform smart home sensor data into process mining event logs
2. Apply multiple process discovery algorithms (Alpha, Heuristic, Inductive Miners)
3. Visualize discovered process models
4. Analyze activity patterns, trace variants, and throughput times
5. Provide insights into smart home inhabitant behavior

## 📊 Dataset: CASAS Aruba

**Source**: CASAS Smart Home Project (Washington State University)

**Structure**:
```
date, time, sensor_id, sensor_value
2010-11-04, 02:32:33.351906, Bedroom, ON
2010-11-04, 02:32:38.895958, Bedroom, OFF
...
```

- **Size**: ~1.6 million sensor events
- **Duration**: Several months of continuous monitoring
- **Sensors**: Motion sensors, door sensors, etc. across different rooms
- **Values**: ON/OFF events

## 🏗️ Solution Architecture

### Design Decisions

#### 1. **Case ID Definition**
We use **daily cases** (`YYYY-MM-DD`) where each day represents one process instance. This choice:
- ✅ Captures daily activity patterns
- ✅ Creates manageable case sizes
- ✅ Aligns with natural human activity cycles
- ✅ Provides meaningful aggregation level

**Alternative**: Session-based cases (separated by long inactivity gaps) are also implemented.

#### 2. **Activity Definition**
Activities are defined as `Sensor_ID + Sensor_Value`:
- Example: `Bedroom_ON`, `Kitchen_OFF`
- Rationale: Captures both location and state change
- Creates interpretable process models

#### 3. **Noise Reduction**
Rapid-fire duplicate sensor events (< 1 second apart) are filtered to:
- Reduce process model complexity
- Remove sensor noise
- Focus on meaningful state transitions

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `pm4py`: Process mining library
- `matplotlib`: Visualization
- `seaborn`: Statistical visualization
- `graphviz`: Process model rendering

**Note**: You may need to install Graphviz separately on your system:
- **Windows**: Download from https://graphviz.org/download/
- **Linux**: `sudo apt-get install graphviz`
- **macOS**: `brew install graphviz`

## 📖 Usage

### Basic Usage

Run the complete analysis:

```bash
python process_mining_aruba.py
```

### Configuration Options

Edit the `main()` function in `process_mining_aruba.py`:

```python
# Configuration
FILEPATH = 'aruba.csv'           # Path to your dataset
SAMPLE_SIZE = 50000              # None for full dataset, or number for testing
CASE_STRATEGY = 'daily'          # 'daily' or 'session'
REMOVE_DUPLICATES = True         # Remove rapid-fire sensor events
TIME_THRESHOLD = 1               # Seconds between duplicate events
```

### For Large Datasets

Start with a sample for testing:
```python
SAMPLE_SIZE = 50000  # Use 50K events for quick testing
```

Once validated, use the full dataset:
```python
SAMPLE_SIZE = None  # Process all 1.6M events
```

## 📂 Code Structure

```
process_mining_aruba.py
├── 1. DATA LOADING AND PREPROCESSING
│   ├── load_aruba_data()          # Load CSV file
│   └── preprocess_data()          # Clean and prepare data
│
├── 2. EVENT LOG CREATION
│   ├── create_event_log()         # Create pm4py-compatible log
│   └── convert_to_pm4py_log()     # Convert to EventLog object
│
├── 3. PROCESS DISCOVERY
│   ├── discover_process_alpha_miner()      # Alpha algorithm
│   ├── discover_process_heuristic_miner()  # Heuristic algorithm
│   └── discover_process_inductive_miner()  # Inductive algorithm
│
├── 4. VISUALIZATION
│   ├── visualize_petri_net()      # Visualize Petri nets
│   ├── visualize_heuristics_net() # Visualize heuristics nets
│   └── visualize_process_tree()   # Visualize process trees
│
├── 5. ANALYSIS
│   ├── analyze_activity_frequency()  # Activity statistics
│   ├── analyze_trace_variants()      # Unique process paths
│   ├── analyze_throughput_time()     # Case durations
│   └── analyze_temporal_patterns()   # Time-based patterns
│
└── 6. MAIN EXECUTION
    └── main()                      # Orchestrates entire pipeline
```

## 🔬 Process Discovery Algorithms

### 1. **Alpha Miner**
- **Type**: Petri net discovery
- **Strengths**: 
  - Theoretically sound
  - Discovers concurrency and choices
- **Weaknesses**: 
  - Sensitive to noise
  - Cannot handle loops well
- **Output**: `alpha_miner_model.png`

### 2. **Heuristic Miner**
- **Type**: Heuristics net discovery
- **Strengths**: 
  - Robust to noise
  - Handles frequency information
  - Shows main process paths
- **Weaknesses**: 
  - May miss rare behaviors
- **Output**: `heuristic_miner_model.png`

### 3. **Inductive Miner**
- **Type**: Process tree / Petri net discovery
- **Strengths**: 
  - Guarantees sound models (no deadlocks)
  - Handles incomplete logs
  - Scalable
- **Weaknesses**: 
  - May be less precise
- **Output**: `inductive_miner_tree.png`, `inductive_miner_model.png`

## 📊 Output Files

The script generates the following visualization files:

### Process Models
1. **`alpha_miner_model.png`** - Petri net from Alpha Miner
2. **`heuristic_miner_model.png`** - Heuristics net showing frequent paths
3. **`inductive_miner_tree.png`** - Hierarchical process tree structure
4. **`inductive_miner_model.png`** - Petri net from Inductive Miner

### Analysis Visualizations
5. **`activity_frequency.png`** - Top 20 most frequent activities
6. **`throughput_time_analysis.png`** - Case duration distribution
7. **`temporal_patterns.png`** - Hourly/daily/time-series patterns

## 📈 Analysis Components

### 1. Activity Frequency Analysis
- Identifies most common sensor events
- Shows distribution of activities
- Helps understand inhabitant behavior

### 2. Trace Variant Analysis
- Discovers unique process paths
- Measures process complexity
- Identifies common behavioral patterns

### 3. Throughput Time Analysis
- Calculates case durations (hours per day)
- Shows distribution of daily activity spans
- Identifies outliers and patterns

### 4. Temporal Pattern Analysis
- Hour-of-day activity patterns
- Day-of-week distributions
- Time-series trend analysis

## 🎓 Academic Considerations

### Design Rationale Documentation

**Case ID Selection**:
- Daily cases align with circadian rhythms
- Provides natural process boundaries
- Balances granularity and complexity

**Activity Mapping**:
- `Sensor_ID + Value` creates interpretable activities
- Each activity represents a meaningful state transition
- Maintains traceability to physical sensor events

**Preprocessing Decisions**:
- 1-second threshold removes sensor bounce/noise
- Sorting ensures temporal correctness
- No data is removed arbitrarily; only duplicates

### Expected Insights

From the Aruba dataset, you should observe:
1. **Morning routines**: Kitchen → Bathroom → Bedroom patterns
2. **Sleep cycles**: Extended periods of bedroom activity
3. **Meal preparation**: Kitchen sensor clusters
4. **Activity loops**: Repeated room transitions

### Limitations

1. **Process Complexity**: Smart homes generate highly variable traces
2. **Noise**: Sensor data contains false positives
3. **Granularity**: Daily cases may mask within-day patterns
4. **Activity Labels**: No ground-truth activity labels in basic CSV

## 🔧 Customization

### Changing Case Definition

```python
# In create_event_log() function
case_strategy = 'session'  # Use session-based cases
```

### Adjusting Noise Filtering

```python
# In preprocess_data() function
time_threshold_seconds = 2  # Increase to 2 seconds
remove_duplicates = False   # Disable filtering entirely
```

### Selecting Specific Sensors

```python
# After load_aruba_data()
df = df[df['sensor_id'].isin(['Bedroom', 'Kitchen', 'Bathroom'])]
```

## 📚 References

### CASAS Dataset
- Cook, D. J., & Schmitter-Edgecombe, M. (2009). "Assessing the quality of activities in a smart environment."
- CASAS: http://casas.wsu.edu/

### Process Mining
- van der Aalst, W. M. P. (2016). "Process Mining: Data Science in Action."
- pm4py documentation: https://pm4py.fit.fraunhofer.de/

### Algorithms
- **Alpha Miner**: van der Aalst et al. (2004) "Workflow mining: Discovering process models from event logs"
- **Heuristic Miner**: Weijters & van der Aalst (2003) "Rediscovering workflow models"
- **Inductive Miner**: Leemans et al. (2013) "Discovering block-structured process models"

## ⚠️ Troubleshooting

### Memory Issues
- Use `SAMPLE_SIZE` to test with smaller dataset
- Process data in chunks for very large files

### Graphviz Errors
- Ensure Graphviz is installed system-wide
- Add Graphviz bin directory to PATH

### Slow Processing
- Reduce `SAMPLE_SIZE`
- Increase `TIME_THRESHOLD` to filter more events
- Consider session-based cases for fewer, longer cases

## 🤝 Contributing

This is an academic assignment. For improvements:
1. Fork the repository
2. Create a feature branch
3. Add comprehensive documentation
4. Submit a pull request

## 📄 License

This code is provided for educational purposes as part of a Process Mining course assignment.

## 👤 Author

Process Mining Assignment - January 2026

---

**Note**: This solution prioritizes clarity and educational value over extreme optimization, making it suitable for academic assignments and learning purposes.
