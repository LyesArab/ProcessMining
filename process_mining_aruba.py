"""
Process Mining Solution for CASAS Aruba Smart Home Dataset
============================================================

This script implements a complete process mining pipeline for the Aruba smart home dataset:
1. Data loading and preprocessing
2. Event log creation compatible with pm4py
3. Process discovery using Alpha, Heuristic, and Inductive miners
4. Process model visualization
5. Basic process analysis (activity frequency, trace variants, throughput time)

Dataset Structure:
- Column 1: Date (YYYY-MM-DD)
- Column 2: Time (HH:MM:SS.ffffff)
- Column 3: Sensor ID (location/sensor name)
- Column 4: Sensor Value (ON/OFF)

Author: Process Mining Assignment
Date: January 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# pm4py imports
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.variants.log import get as variants_module
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


# ========================================
# 1. DATA LOADING AND PREPROCESSING
# ========================================

def load_aruba_data(filepath, sample_size=None):
    """
    Load the CASAS Aruba dataset from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the aruba.csv file
    sample_size : int, optional
        Number of rows to load (for testing purposes). None loads all data.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, time, sensor_id, sensor_value
    """
    print("=" * 60)
    print("STEP 1: Loading Aruba Dataset")
    print("=" * 60)
    
    # Column names for CASAS Aruba dataset
    column_names = ['date', 'time', 'sensor_id', 'sensor_value']
    
    # Load data
    if sample_size:
        print(f"Loading {sample_size} rows from {filepath}...")
        df = pd.read_csv(filepath, names=column_names, nrows=sample_size)
    else:
        print(f"Loading all data from {filepath}...")
        df = pd.read_csv(filepath, names=column_names)
    
    print(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
    print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nDataset Info:")
    print(df.info())
    
    return df


def preprocess_data(df, remove_duplicates=True, time_threshold_seconds=1):
    """
    Preprocess the Aruba dataset:
    - Combine date and time into a single timestamp
    - Sort by timestamp
    - Remove duplicate/rapid-fire sensor events (optional)
    - Create activity labels
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe from load_aruba_data()
    remove_duplicates : bool
        Whether to remove consecutive duplicate sensor firings
    time_threshold_seconds : float
        Minimum time between same sensor events (for noise reduction)
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe with timestamp column
    """
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing Data")
    print("=" * 60)
    
    df = df.copy()
    
    # 1. Create combined timestamp
    print("✓ Creating timestamp column...")
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    # 2. Sort by timestamp
    print("✓ Sorting by timestamp...")
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 3. Basic data validation
    print("✓ Validating data...")
    initial_count = len(df)
    df = df.dropna(subset=['timestamp', 'sensor_id', 'sensor_value'])
    print(f"  - Removed {initial_count - len(df)} rows with missing values")
    
    # 4. Remove rapid-fire duplicate events (noise reduction)
    if remove_duplicates:
        print(f"✓ Removing duplicate sensor events (threshold: {time_threshold_seconds}s)...")
        
        # Add time difference column
        df['time_diff'] = df.groupby('sensor_id')['timestamp'].diff().dt.total_seconds()
        
        # Keep first event and events that are far enough apart
        df['keep'] = (df['time_diff'].isna()) | (df['time_diff'] >= time_threshold_seconds)
        
        before_dedup = len(df)
        df = df[df['keep']].copy()
        after_dedup = len(df)
        
        print(f"  - Removed {before_dedup - after_dedup:,} rapid-fire events ({((before_dedup - after_dedup) / before_dedup * 100):.2f}%)")
        
        # Clean up temporary columns
        df = df.drop(['time_diff', 'keep'], axis=1)
    
    # 5. Create activity label (sensor_id + sensor_value)
    print("✓ Creating activity labels...")
    df['activity'] = df['sensor_id'] + '_' + df['sensor_value']
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  - Final dataset: {len(df):,} events")
    print(f"  - Unique sensors: {df['sensor_id'].nunique()}")
    print(f"  - Unique activities: {df['activity'].nunique()}")
    print(f"  - Time span: {df['timestamp'].max() - df['timestamp'].min()}")
    
    # Display sensor statistics
    print(f"\nSensor Statistics:")
    sensor_counts = df['sensor_id'].value_counts().head(10)
    for sensor, count in sensor_counts.items():
        print(f"  {sensor:20s}: {count:7,} events")
    
    return df


# ========================================
# 2. EVENT LOG CREATION
# ========================================

def create_event_log(df, case_strategy='daily', activity_column='activity'):
    """
    Create a pm4py-compatible event log from the preprocessed dataframe.
    
    Case ID Strategy:
    - 'daily': Each day (YYYY-MM-DD) becomes one case
    - 'session': Activity sessions separated by long gaps (e.g., sleep periods)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe with timestamp and activity columns
    case_strategy : str
        Strategy for creating case IDs ('daily' or 'session')
    activity_column : str
        Column name to use as activity
    
    Returns:
    --------
    pd.DataFrame
        Event log with columns: case_id, activity, timestamp
    """
    print("\n" + "=" * 60)
    print("STEP 3: Creating Event Log for pm4py")
    print("=" * 60)
    print(f"Case Strategy: {case_strategy}")
    print(f"Activity Column: {activity_column}")
    
    df = df.copy()
    
    if case_strategy == 'daily':
        # Each day is one case
        print("✓ Creating daily case IDs...")
        df['case_id'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        
    elif case_strategy == 'session':
        # Sessions separated by gaps > 2 hours (likely sleep periods)
        print("✓ Creating session-based case IDs...")
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate time gap between consecutive events
        df['time_gap'] = df['timestamp'].diff().dt.total_seconds()
        
        # New session starts after gap > 2 hours (7200 seconds)
        df['new_session'] = (df['time_gap'] > 7200) | (df['time_gap'].isna())
        df['session_id'] = df['new_session'].cumsum()
        
        # Create case_id as date + session number
        df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        df['case_id'] = df['date_str'] + '_S' + df['session_id'].astype(str)
        
        # Clean up temporary columns
        df = df.drop(['time_gap', 'new_session', 'session_id', 'date_str'], axis=1)
    
    else:
        raise ValueError(f"Unknown case_strategy: {case_strategy}")
    
    # Create the event log with required columns
    event_log = df[['case_id', activity_column, 'timestamp']].copy()
    event_log = event_log.rename(columns={activity_column: 'activity'})
    
    # Sort by case_id and timestamp
    event_log = event_log.sort_values(['case_id', 'timestamp']).reset_index(drop=True)
    
    # Add additional standard attributes for pm4py
    event_log['case:concept:name'] = event_log['case_id']
    event_log['concept:name'] = event_log['activity']
    event_log['time:timestamp'] = event_log['timestamp']
    
    print(f"\n✓ Event log created successfully!")
    print(f"  - Total events: {len(event_log):,}")
    print(f"  - Total cases: {event_log['case_id'].nunique():,}")
    print(f"  - Unique activities: {event_log['activity'].nunique()}")
    print(f"  - Avg events per case: {len(event_log) / event_log['case_id'].nunique():.2f}")
    
    # Display case statistics
    case_lengths = event_log.groupby('case_id').size()
    print(f"\nCase Length Statistics:")
    print(f"  - Min: {case_lengths.min()} events")
    print(f"  - Max: {case_lengths.max()} events")
    print(f"  - Mean: {case_lengths.mean():.2f} events")
    print(f"  - Median: {case_lengths.median():.2f} events")
    
    return event_log


def convert_to_pm4py_log(event_log_df):
    """
    Convert pandas DataFrame to pm4py EventLog object.
    
    Parameters:
    -----------
    event_log_df : pd.DataFrame
        Event log dataframe with pm4py-compatible columns
    
    Returns:
    --------
    pm4py.objects.log.obj.EventLog
        pm4py EventLog object
    """
    print("\n✓ Converting to pm4py EventLog object...")
    
    # Convert to pm4py event log
    parameters = {
        log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'
    }
    
    event_log = log_converter.apply(event_log_df, parameters=parameters,
                                     variant=log_converter.Variants.TO_EVENT_LOG)
    
    print(f"✓ pm4py EventLog created with {len(event_log)} cases")
    
    return event_log


# ========================================
# 3. PROCESS DISCOVERY
# ========================================

def discover_process_alpha_miner(event_log):
    """
    Discover process model using Alpha Miner algorithm.
    
    Alpha Miner discovers a Petri net from the event log based on
    ordering relations between activities.
    
    Parameters:
    -----------
    event_log : pm4py.objects.log.obj.EventLog
        pm4py event log
    
    Returns:
    --------
    tuple
        (net, initial_marking, final_marking)
    """
    print("\n" + "=" * 60)
    print("STEP 4a: Process Discovery - Alpha Miner")
    print("=" * 60)
    print("Alpha Miner: Discovers Petri nets based on ordering relations")
    
    try:
        net, initial_marking, final_marking = alpha_miner.apply(event_log)
        print(f"✓ Alpha Miner completed successfully")
        print(f"  - Places: {len(net.places)}")
        print(f"  - Transitions: {len(net.transitions)}")
        print(f"  - Arcs: {len(net.arcs)}")
        return net, initial_marking, final_marking
    except Exception as e:
        print(f"✗ Alpha Miner failed: {str(e)}")
        return None, None, None


def discover_process_heuristic_miner(event_log):
    """
    Discover process model using Heuristic Miner algorithm.
    
    Heuristic Miner is robust to noise and discovers a Heuristics Net
    that shows the most frequent paths through the process.
    
    Parameters:
    -----------
    event_log : pm4py.objects.log.obj.EventLog
        pm4py event log
    
    Returns:
    --------
    pm4py.objects.heuristics_net.obj.HeuristicsNet
        Heuristics net object
    """
    print("\n" + "=" * 60)
    print("STEP 4b: Process Discovery - Heuristic Miner")
    print("=" * 60)
    print("Heuristic Miner: Robust to noise, shows most frequent paths")
    
    try:
        heu_net = heuristics_miner.apply_heu(event_log)
        print(f"✓ Heuristic Miner completed successfully")
        print(f"  - Activities: {len(heu_net.nodes)}")
        return heu_net
    except Exception as e:
        print(f"✗ Heuristic Miner failed: {str(e)}")
        return None


def discover_process_inductive_miner(event_log):
    """
    Discover process model using Inductive Miner algorithm.
    
    Inductive Miner guarantees sound process models (no deadlocks)
    and discovers a process tree structure.
    
    Parameters:
    -----------
    event_log : pm4py.objects.log.obj.EventLog
        pm4py event log
    
    Returns:
    --------
    tuple
        (process_tree, net, initial_marking, final_marking)
    """
    print("\n" + "=" * 60)
    print("STEP 4c: Process Discovery - Inductive Miner")
    print("=" * 60)
    print("Inductive Miner: Guarantees sound models, discovers process tree")
    
    try:
        # Apply inductive miner - returns (net, im, fm) tuple
        result = inductive_miner.apply(event_log)
        
        # Check if result is a tuple or ProcessTree
        if isinstance(result, tuple) and len(result) == 3:
            net, initial_marking, final_marking = result
            tree = None
        else:
            # Result is a ProcessTree, convert to Petri net
            tree = result
            from pm4py.objects.conversion.process_tree import converter as pt_converter
            net, initial_marking, final_marking = pt_converter.apply(tree)
        
        print(f"[OK] Inductive Miner completed successfully")
        print(f"  - Petri net discovered")
        print(f"  - Places: {len(net.places)}")
        print(f"  - Transitions: {len(net.transitions)}")
        if tree:
            print(f"  - Process tree also available")
        return tree, net, initial_marking, final_marking
    except Exception as e:
        print(f"[FAIL] Inductive Miner failed: {str(e)}")
        return None, None, None, None


# ========================================
# 4. PROCESS MODEL VISUALIZATION
# ========================================

def visualize_petri_net(net, initial_marking, final_marking, 
                        output_file='alpha_miner_model.png', title='Alpha Miner'):
    """
    Visualize a Petri net and save to file.
    
    Parameters:
    -----------
    net : pm4py.objects.petri_net.obj.PetriNet
        Petri net object
    initial_marking : pm4py.objects.petri_net.obj.Marking
        Initial marking
    final_marking : pm4py.objects.petri_net.obj.Marking
        Final marking
    output_file : str
        Output filename
    title : str
        Title for the visualization
    """
    if net is None:
        print(f"✗ Cannot visualize {title} - model is None")
        return
    
    print(f"\n✓ Visualizing {title}...")
    try:
        gviz = pn_visualizer.apply(net, initial_marking, final_marking)
        pn_visualizer.save(gviz, output_file)
        print(f"  - Saved to: {output_file}")
    except Exception as e:
        print(f"✗ Visualization failed: {str(e)}")


def visualize_heuristics_net(heu_net, output_file='heuristic_miner_model.png'):
    """
    Visualize a Heuristics Net and save to file.
    
    Parameters:
    -----------
    heu_net : pm4py.objects.heuristics_net.obj.HeuristicsNet
        Heuristics net object
    output_file : str
        Output filename
    """
    if heu_net is None:
        print(f"✗ Cannot visualize Heuristics Net - model is None")
        return
    
    print(f"\n✓ Visualizing Heuristics Net...")
    try:
        gviz = hn_visualizer.apply(heu_net)
        hn_visualizer.save(gviz, output_file)
        print(f"  - Saved to: {output_file}")
    except Exception as e:
        print(f"✗ Visualization failed: {str(e)}")


def visualize_process_tree(tree, output_file='inductive_miner_tree.png'):
    """
    Visualize a process tree and save to file.
    
    Parameters:
    -----------
    tree : pm4py.objects.process_tree.obj.ProcessTree
        Process tree object
    output_file : str
        Output filename
    """
    if tree is None:
        print(f"✗ Cannot visualize Process Tree - model is None")
        return
    
    print(f"\n✓ Visualizing Process Tree...")
    try:
        gviz = pt_visualizer.apply(tree)
        pt_visualizer.save(gviz, output_file)
        print(f"  - Saved to: {output_file}")
    except Exception as e:
        print(f"✗ Visualization failed: {str(e)}")


# ========================================
# 5. PROCESS ANALYSIS
# ========================================

def analyze_activity_frequency(event_log_df, top_n=20):
    """
    Analyze and visualize activity frequencies.
    
    Parameters:
    -----------
    event_log_df : pd.DataFrame
        Event log dataframe
    top_n : int
        Number of top activities to display
    """
    print("\n" + "=" * 60)
    print("STEP 5a: Activity Frequency Analysis")
    print("=" * 60)
    
    activity_counts = event_log_df['activity'].value_counts()
    
    print(f"\nTop {top_n} Most Frequent Activities:")
    print("-" * 50)
    for i, (activity, count) in enumerate(activity_counts.head(top_n).items(), 1):
        percentage = (count / len(event_log_df)) * 100
        print(f"{i:2d}. {activity:30s}: {count:7,} ({percentage:5.2f}%)")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    top_activities = activity_counts.head(top_n)
    plt.barh(range(len(top_activities)), top_activities.values)
    plt.yticks(range(len(top_activities)), top_activities.index)
    plt.xlabel('Frequency')
    plt.title(f'Top {top_n} Most Frequent Activities in Aruba Smart Home')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('activity_frequency.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Activity frequency plot saved to: activity_frequency.png")
    plt.close()
    
    return activity_counts


def analyze_trace_variants(event_log):
    """
    Analyze trace variants (unique process paths).
    
    Parameters:
    -----------
    event_log : pm4py.objects.log.obj.EventLog
        pm4py event log
    """
    print("\n" + "=" * 60)
    print("STEP 5b: Trace Variant Analysis")
    print("=" * 60)
    
    variants = variants_module.get_variants(event_log)
    
    print(f"\nTrace Variant Statistics:")
    print(f"  - Total cases: {len(event_log)}")
    print(f"  - Unique variants: {len(variants)}")
    print(f"  - Process complexity: {len(variants) / len(event_log):.4f}")
    
    # Sort variants by frequency
    sorted_variants = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nTop 10 Most Frequent Trace Variants:")
    print("-" * 80)
    for i, (variant, cases) in enumerate(sorted_variants[:10], 1):
        percentage = (len(cases) / len(event_log)) * 100
        # Convert variant tuple to readable string
        variant_str = ' → '.join(list(variant)[:5])  # Show first 5 activities
        if len(variant) > 5:
            variant_str += f' ... ({len(variant)} activities total)'
        print(f"{i:2d}. {len(cases):4} cases ({percentage:5.2f}%): {variant_str}")
    
    # Variant coverage analysis
    cumulative_coverage = []
    cumulative_cases = 0
    for i, (variant, cases) in enumerate(sorted_variants, 1):
        cumulative_cases += len(cases)
        coverage = (cumulative_cases / len(event_log)) * 100
        cumulative_coverage.append((i, coverage))
        if coverage >= 80 and len(cumulative_coverage) == i:
            print(f"\n✓ {i} variants cover 80% of all cases")
    
    return variants


def analyze_throughput_time(event_log_df):
    """
    Analyze throughput time (duration) for each case.
    
    Parameters:
    -----------
    event_log_df : pd.DataFrame
        Event log dataframe
    """
    print("\n" + "=" * 60)
    print("STEP 5c: Throughput Time Analysis")
    print("=" * 60)
    
    # Calculate case duration
    case_times = event_log_df.groupby('case_id')['timestamp'].agg(['min', 'max'])
    case_times['duration'] = (case_times['max'] - case_times['min']).dt.total_seconds() / 3600  # hours
    
    print(f"\nCase Duration Statistics (in hours):")
    print("-" * 50)
    print(f"  - Mean:   {case_times['duration'].mean():8.2f} hours")
    print(f"  - Median: {case_times['duration'].median():8.2f} hours")
    print(f"  - Min:    {case_times['duration'].min():8.2f} hours")
    print(f"  - Max:    {case_times['duration'].max():8.2f} hours")
    print(f"  - Std:    {case_times['duration'].std():8.2f} hours")
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = case_times['duration'].quantile(p/100)
        print(f"  - {p:2d}th: {value:8.2f} hours")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(case_times['duration'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Duration (hours)')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of Case Durations')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(case_times['duration'], vert=True)
    plt.ylabel('Duration (hours)')
    plt.title('Case Duration Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('throughput_time_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Throughput time plot saved to: throughput_time_analysis.png")
    plt.close()
    
    return case_times


def analyze_temporal_patterns(event_log_df):
    """
    Analyze temporal patterns in the event log.
    
    Parameters:
    -----------
    event_log_df : pd.DataFrame
        Event log dataframe
    """
    print("\n" + "=" * 60)
    print("STEP 5d: Temporal Pattern Analysis")
    print("=" * 60)
    
    df = event_log_df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['date'] = df['timestamp'].dt.date
    
    # Events per hour
    events_per_hour = df.groupby('hour').size()
    
    print(f"\nEvents by Hour of Day:")
    print("-" * 50)
    for hour in range(24):
        count = events_per_hour.get(hour, 0)
        bar = '█' * int(count / events_per_hour.max() * 50)
        print(f"{hour:02d}:00 - {hour:02d}:59 | {bar} {count:6,}")
    
    # Visualization
    plt.figure(figsize=(14, 8))
    
    # Hour of day
    plt.subplot(2, 2, 1)
    plt.bar(events_per_hour.index, events_per_hour.values, edgecolor='black', alpha=0.7)
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Events')
    plt.title('Event Distribution by Hour of Day')
    plt.grid(True, alpha=0.3)
    
    # Day of week
    plt.subplot(2, 2, 2)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    events_per_day = df['day_of_week'].value_counts().reindex(day_order)
    plt.bar(range(len(events_per_day)), events_per_day.values, edgecolor='black', alpha=0.7)
    plt.xticks(range(len(day_order)), day_order, rotation=45, ha='right')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Events')
    plt.title('Event Distribution by Day of Week')
    plt.grid(True, alpha=0.3)
    
    # Events per day over time
    plt.subplot(2, 1, 2)
    events_per_date = df.groupby('date').size()
    plt.plot(events_per_date.index, events_per_date.values, linewidth=1, alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Number of Events')
    plt.title('Daily Event Activity Over Time')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_patterns.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Temporal pattern plot saved to: temporal_patterns.png")
    plt.close()


# ========================================
# 6. MAIN EXECUTION PIPELINE
# ========================================

def main():
    """
    Main execution pipeline for process mining analysis.
    """
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  CASAS ARUBA SMART HOME - PROCESS MINING ANALYSIS  ".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    
    # Configuration
    FILEPATH = 'aruba.csv'
    SAMPLE_SIZE = 50000  # Set to None to use full dataset, or a number for testing
    CASE_STRATEGY = 'daily'  # 'daily' or 'session'
    REMOVE_DUPLICATES = True
    TIME_THRESHOLD = 1  # seconds
    
    print(f"\nConfiguration:")
    print(f"  - File: {FILEPATH}")
    print(f"  - Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'Full dataset'}")
    print(f"  - Case strategy: {CASE_STRATEGY}")
    print(f"  - Remove duplicates: {REMOVE_DUPLICATES}")
    print(f"  - Time threshold: {TIME_THRESHOLD}s")
    
    try:
        # Step 1: Load data
        df = load_aruba_data(FILEPATH, sample_size=SAMPLE_SIZE)
        
        # Step 2: Preprocess data
        df_clean = preprocess_data(df, 
                                   remove_duplicates=REMOVE_DUPLICATES,
                                   time_threshold_seconds=TIME_THRESHOLD)
        
        # Step 3: Create event log
        event_log_df = create_event_log(df_clean, 
                                        case_strategy=CASE_STRATEGY,
                                        activity_column='activity')
        
        # Convert to pm4py event log
        event_log = convert_to_pm4py_log(event_log_df)
        
        # Step 4: Process Discovery
        print("\n" + "=" * 60)
        print("PROCESS DISCOVERY")
        print("=" * 60)
        
        # Alpha Miner
        alpha_net, alpha_im, alpha_fm = discover_process_alpha_miner(event_log)
        visualize_petri_net(alpha_net, alpha_im, alpha_fm, 
                          'alpha_miner_model.png', 'Alpha Miner')
        
        # Heuristic Miner
        heu_net = discover_process_heuristic_miner(event_log)
        visualize_heuristics_net(heu_net, 'heuristic_miner_model.png')
        
        # Inductive Miner
        tree, ind_net, ind_im, ind_fm = discover_process_inductive_miner(event_log)
        visualize_process_tree(tree, 'inductive_miner_tree.png')
        visualize_petri_net(ind_net, ind_im, ind_fm,
                          'inductive_miner_model.png', 'Inductive Miner')
        
        # Step 5: Process Analysis
        print("\n" + "=" * 60)
        print("PROCESS ANALYSIS")
        print("=" * 60)
        
        activity_counts = analyze_activity_frequency(event_log_df, top_n=20)
        variants = analyze_trace_variants(event_log)
        case_times = analyze_throughput_time(event_log_df)
        analyze_temporal_patterns(event_log_df)
        
        # Summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("\nGenerated Files:")
        print("  1. alpha_miner_model.png - Alpha Miner Petri net")
        print("  2. heuristic_miner_model.png - Heuristic Miner net")
        print("  3. inductive_miner_tree.png - Inductive Miner process tree")
        print("  4. inductive_miner_model.png - Inductive Miner Petri net")
        print("  5. activity_frequency.png - Activity frequency analysis")
        print("  6. throughput_time_analysis.png - Case duration analysis")
        print("  7. temporal_patterns.png - Temporal pattern analysis")
        
        print("\n✓ Process mining analysis completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
