#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Parallel Data Processing for Large-Scale Datasets
Purpose: Efficient processing of billion-scale datasets using parallel computing
Author: Matthew Fong
Skills Demonstrated: Parallel processing, NumPy vectorization, HDF5, memory management

Industry Applications:
- Big data ETL pipelines
- Real-time analytics on streaming data
- Large-scale feature engineering
- Distributed computing for ML training
"""

import numpy as np
import h5py
from multiprocessing import Pool, cpu_count
from typing import Tuple, Dict, List, Optional, Callable, Generator
import os
import time
from functools import partial


class LargeScaleDataProcessor:
    """
    TECHNIQUE: Parallel Processing with Memory-Efficient Chunking
    INDUSTRY APPLICATION: Processing datasets too large to fit in memory,
    ETL pipelines, feature engineering at scale
    KEY SKILL: Distributed computing, memory management, HDF5 data handling
    
    A class for processing large-scale datasets using parallel computing
    and memory-efficient chunking strategies.
    
    Attributes:
        n_workers: Number of parallel workers
        chunk_size: Size of data chunks for processing
        
    Example:
        >>> processor = LargeScaleDataProcessor(n_workers=8, chunk_size=100000)
        >>> results = processor.process_file('large_data.hdf5', transform_func)
    """
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        chunk_size: int = 100000,
        verbose: bool = True
    ):
        """
        Initialize the data processor.
        
        Args:
            n_workers: Number of parallel workers (default: CPU count)
            chunk_size: Number of records per chunk
            verbose: Print progress information
        """
        self.n_workers = n_workers or cpu_count()
        self.chunk_size = chunk_size
        self.verbose = verbose
        
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    @staticmethod
    def _process_chunk(
        args: Tuple[np.ndarray, Callable, Dict]
    ) -> np.ndarray:
        """
        Process a single data chunk.
        
        TECHNIQUE: Stateless chunk processing for parallelization
        INDUSTRY APPLICATION: Map-reduce style data processing
        
        Args:
            args: Tuple of (data_chunk, transform_function, kwargs)
            
        Returns:
            Processed chunk
        """
        data, func, kwargs = args
        return func(data, **kwargs)
    
    def process_array(
        self,
        data: np.ndarray,
        transform_func: Callable,
        **kwargs
    ) -> np.ndarray:
        """
        Process a large array in parallel chunks.
        
        TECHNIQUE: Parallel map over data chunks
        INDUSTRY APPLICATION: Scalable data transformation
        
        Args:
            data: Input array
            transform_func: Function to apply to each chunk
            **kwargs: Additional arguments for transform_func
            
        Returns:
            Processed array
        """
        n_samples = len(data)
        n_chunks = (n_samples + self.chunk_size - 1) // self.chunk_size
        
        self._log(f"Processing {n_samples:,} samples in {n_chunks} chunks...")
        
        # Split data into chunks
        chunks = [
            data[i * self.chunk_size:(i + 1) * self.chunk_size]
            for i in range(n_chunks)
        ]
        
        # Prepare arguments for parallel processing
        args = [(chunk, transform_func, kwargs) for chunk in chunks]
        
        # Process in parallel
        tic = time.time()
        with Pool(self.n_workers) as pool:
            results = pool.map(self._process_chunk, args)
        
        toc = time.time()
        self._log(f"Processing completed in {toc - tic:.2f} seconds")
        self._log(f"Throughput: {n_samples / (toc - tic):,.0f} samples/second")
        
        return np.concatenate(results)
    
    def process_hdf5(
        self,
        input_path: str,
        output_path: str,
        dataset_name: str,
        transform_func: Callable,
        output_dataset_name: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Process an HDF5 file in streaming fashion.
        
        TECHNIQUE: Streaming HDF5 processing
        INDUSTRY APPLICATION: Processing files larger than RAM
        
        Args:
            input_path: Path to input HDF5 file
            output_path: Path to output HDF5 file
            dataset_name: Name of dataset to process
            transform_func: Transformation function
            output_dataset_name: Name for output dataset
            **kwargs: Additional arguments for transform_func
            
        Returns:
            Processing statistics
        """
        output_dataset_name = output_dataset_name or f"{dataset_name}_processed"
        
        self._log(f"Processing {input_path}...")
        
        with h5py.File(input_path, 'r') as f_in:
            dataset = f_in[dataset_name]
            n_samples = dataset.shape[0]
            
            # Determine output shape from first chunk
            first_chunk = dataset[:min(self.chunk_size, n_samples)]
            first_result = transform_func(first_chunk, **kwargs)
            output_shape = (n_samples,) + first_result.shape[1:]
            
            self._log(f"Input shape: {dataset.shape}")
            self._log(f"Output shape: {output_shape}")
            
            with h5py.File(output_path, 'w') as f_out:
                # Create output dataset
                out_dataset = f_out.create_dataset(
                    output_dataset_name,
                    shape=output_shape,
                    dtype=first_result.dtype,
                    chunks=True,
                    compression='gzip'
                )
                
                # Process in chunks
                tic = time.time()
                n_processed = 0
                
                for start in range(0, n_samples, self.chunk_size):
                    end = min(start + self.chunk_size, n_samples)
                    chunk = dataset[start:end]
                    
                    # Process chunk
                    result = transform_func(chunk, **kwargs)
                    out_dataset[start:end] = result
                    
                    n_processed += len(chunk)
                    if self.verbose and n_processed % (self.chunk_size * 10) == 0:
                        progress = 100 * n_processed / n_samples
                        self._log(f"Progress: {progress:.1f}%")
                
                toc = time.time()
        
        stats = {
            'n_samples': n_samples,
            'processing_time': toc - tic,
            'throughput': n_samples / (toc - tic),
            'input_path': input_path,
            'output_path': output_path
        }
        
        self._log(f"Completed in {stats['processing_time']:.2f} seconds")
        self._log(f"Throughput: {stats['throughput']:,.0f} samples/second")
        
        return stats


class VectorizedAggregator:
    """
    TECHNIQUE: Vectorized Binning and Aggregation
    INDUSTRY APPLICATION: Feature engineering, data summarization,
    histogram computation at scale
    KEY SKILL: NumPy vectorization for performance
    
    Efficient aggregation operations using NumPy vectorization.
    """
    
    @staticmethod
    def bin_statistics(
        values: np.ndarray,
        bin_indices: np.ndarray,
        n_bins: int,
        statistic: str = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute statistics within bins using vectorization.
        
        TECHNIQUE: Vectorized binned statistics
        INDUSTRY APPLICATION: Feature aggregation, histogram computation
        
        Args:
            values: Values to aggregate
            bin_indices: Bin assignment for each value
            n_bins: Total number of bins
            statistic: Statistic to compute ('mean', 'sum', 'std', 'count')
            
        Returns:
            Tuple of (bin_statistics, bin_counts)
        """
        # Count per bin
        counts = np.bincount(bin_indices, minlength=n_bins)
        
        if statistic == 'count':
            return counts.astype(float), counts
        
        # Sum per bin
        sums = np.bincount(bin_indices, weights=values, minlength=n_bins)
        
        if statistic == 'sum':
            return sums, counts
        
        # Mean per bin
        with np.errstate(invalid='ignore'):
            means = np.where(counts > 0, sums / counts, 0)
        
        if statistic == 'mean':
            return means, counts
        
        if statistic == 'std':
            # Compute variance using E[X²] - E[X]²
            sum_sq = np.bincount(bin_indices, weights=values**2, minlength=n_bins)
            with np.errstate(invalid='ignore'):
                variance = np.where(
                    counts > 1,
                    (sum_sq / counts) - means**2,
                    0
                )
            return np.sqrt(np.maximum(variance, 0)), counts
        
        raise ValueError(f"Unknown statistic: {statistic}")
    
    @staticmethod
    def bin_2d(
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        x_bins: np.ndarray,
        y_bins: np.ndarray,
        statistic: str = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D binned statistics.
        
        TECHNIQUE: 2D histogram with arbitrary statistics
        INDUSTRY APPLICATION: Heatmap generation, spatial aggregation
        
        Args:
            x: X coordinates
            y: Y coordinates
            values: Values to aggregate
            x_bins: Bin edges for x
            y_bins: Bin edges for y
            statistic: Statistic to compute
            
        Returns:
            Tuple of (2D statistics array, 2D counts array)
        """
        # Digitize coordinates
        x_idx = np.digitize(x, x_bins) - 1
        y_idx = np.digitize(y, y_bins) - 1
        
        # Clip to valid range
        x_idx = np.clip(x_idx, 0, len(x_bins) - 2)
        y_idx = np.clip(y_idx, 0, len(y_bins) - 2)
        
        # Convert to 1D index
        n_x = len(x_bins) - 1
        n_y = len(y_bins) - 1
        flat_idx = y_idx * n_x + x_idx
        
        # Compute statistics
        stats_flat, counts_flat = VectorizedAggregator.bin_statistics(
            values, flat_idx, n_x * n_y, statistic
        )
        
        # Reshape to 2D
        stats_2d = stats_flat.reshape(n_y, n_x)
        counts_2d = counts_flat.reshape(n_y, n_x)
        
        return stats_2d, counts_2d


class DataPipeline:
    """
    TECHNIQUE: Composable Data Pipeline
    INDUSTRY APPLICATION: ETL workflows, feature engineering pipelines
    KEY SKILL: Functional programming, pipeline design
    
    Build composable data processing pipelines.
    """
    
    def __init__(self):
        """Initialize empty pipeline."""
        self.steps = []
        
    def add_step(
        self,
        name: str,
        func: Callable,
        **kwargs
    ) -> 'DataPipeline':
        """
        Add a processing step to the pipeline.
        
        Args:
            name: Step name for logging
            func: Processing function
            **kwargs: Arguments for the function
            
        Returns:
            self for chaining
        """
        self.steps.append({
            'name': name,
            'func': func,
            'kwargs': kwargs
        })
        return self
    
    def run(
        self,
        data: np.ndarray,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Execute the pipeline on input data.
        
        Args:
            data: Input data
            verbose: Print progress
            
        Returns:
            Processed data
        """
        result = data
        
        for i, step in enumerate(self.steps):
            if verbose:
                print(f"Step {i+1}/{len(self.steps)}: {step['name']}")
            
            tic = time.time()
            result = step['func'](result, **step['kwargs'])
            toc = time.time()
            
            if verbose:
                print(f"  Completed in {toc - tic:.2f}s, shape: {result.shape}")
        
        return result
    
    def __repr__(self) -> str:
        """String representation of pipeline."""
        steps_str = '\n'.join([f"  {i+1}. {s['name']}" for i, s in enumerate(self.steps)])
        return f"DataPipeline with {len(self.steps)} steps:\n{steps_str}"


# Common transformation functions
def normalize(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalize data to zero mean and unit variance.
    
    TECHNIQUE: Z-score normalization
    INDUSTRY APPLICATION: Feature scaling for ML
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return np.where(std > 0, (data - mean) / std, 0)


def log_transform(data: np.ndarray, offset: float = 1e-10) -> np.ndarray:
    """
    Apply log transformation with offset for zeros.
    
    TECHNIQUE: Log transformation
    INDUSTRY APPLICATION: Handling skewed distributions
    """
    return np.log10(np.maximum(data, offset))


def clip_outliers(
    data: np.ndarray,
    lower_percentile: float = 1,
    upper_percentile: float = 99
) -> np.ndarray:
    """
    Clip outliers based on percentiles.
    
    TECHNIQUE: Percentile-based outlier removal
    INDUSTRY APPLICATION: Data cleaning, robust statistics
    """
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    return np.clip(data, lower, upper)


if __name__ == "__main__":
    # Demonstration of large-scale data processing
    np.random.seed(42)
    
    # Generate synthetic large dataset
    n_samples = 1_000_000
    n_features = 10
    
    print(f"Generating {n_samples:,} samples with {n_features} features...")
    data = np.random.randn(n_samples, n_features)
    
    # Add some structure
    data[:, 0] = np.abs(data[:, 0]) * 100  # Skewed feature
    data[:, 1] = data[:, 1] + np.random.choice([0, 10], n_samples)  # Bimodal
    
    # Create processing pipeline
    pipeline = DataPipeline()
    pipeline.add_step('Clip outliers', clip_outliers, lower_percentile=1, upper_percentile=99)
    pipeline.add_step('Log transform (col 0)', lambda x: np.column_stack([
        log_transform(x[:, 0]),
        x[:, 1:]
    ]))
    pipeline.add_step('Normalize', normalize)
    
    print(f"\n{pipeline}\n")
    
    # Run pipeline
    processed = pipeline.run(data)
    
    print(f"\nOriginal data stats:")
    print(f"  Mean: {data.mean(axis=0)[:3]}")
    print(f"  Std:  {data.std(axis=0)[:3]}")
    
    print(f"\nProcessed data stats:")
    print(f"  Mean: {processed.mean(axis=0)[:3]}")
    print(f"  Std:  {processed.std(axis=0)[:3]}")
    
    # Demonstrate parallel processing
    print("\n" + "=" * 50)
    print("Parallel Processing Demo")
    print("=" * 50)
    
    processor = LargeScaleDataProcessor(n_workers=4, chunk_size=100000)
    
    def expensive_transform(chunk):
        """Simulate expensive computation."""
        return np.sqrt(np.abs(chunk)) * np.sin(chunk)
    
    result = processor.process_array(data, expensive_transform)
    print(f"Result shape: {result.shape}")
    
    # Demonstrate 2D binning
    print("\n" + "=" * 50)
    print("2D Binning Demo")
    print("=" * 50)
    
    x = np.random.uniform(0, 10, 100000)
    y = np.random.uniform(0, 10, 100000)
    values = np.sin(x) * np.cos(y) + 0.1 * np.random.randn(100000)
    
    x_bins = np.linspace(0, 10, 21)
    y_bins = np.linspace(0, 10, 21)
    
    stats_2d, counts_2d = VectorizedAggregator.bin_2d(
        x, y, values, x_bins, y_bins, statistic='mean'
    )
    
    print(f"2D binned statistics shape: {stats_2d.shape}")
    print(f"Total counts: {counts_2d.sum():,}")
    print(f"Mean value range: [{stats_2d.min():.3f}, {stats_2d.max():.3f}]")
