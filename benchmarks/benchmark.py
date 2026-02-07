#!/usr/bin/env python3
"""
Benchmark script comparing BoltEmbedding against FAISS and scikit-learn.

Usage:
    python benchmark.py
    python benchmark.py --n-embeddings 100000 --embedding-dim 768 --k 10
"""

import argparse
import time
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

# Check for optional dependencies
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: FAISS not installed. Install with: pip install faiss-cpu")

try:
    from sklearn.neighbors import NearestNeighbors

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

from boltembedding import BoltEmbedding, DistanceMetric, Device

# =============================================================================
# Benchmark Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "n_embeddings": [1000, 10000, 100000],
    "embedding_dims": [128, 768],
    "k_values": [1, 10, 100],
    "batch_sizes": [1, 10, 100],
    "n_warmup": 10,
    "n_iterations": 100,
}


# =============================================================================
# Timing Utilities
# =============================================================================


def format_time(nanoseconds: float) -> str:
    """Format time in human-readable units."""
    if nanoseconds < 1e3:
        return f"{nanoseconds:.2f} ns"
    elif nanoseconds < 1e6:
        return f"{nanoseconds / 1e3:.2f} µs"
    elif nanoseconds < 1e9:
        return f"{nanoseconds / 1e6:.2f} ms"
    else:
        return f"{nanoseconds / 1e9:.2f} s"


def benchmark_function(
    fn: Callable, n_warmup: int = 10, n_iterations: int = 100
) -> Tuple[float, float]:
    """
    Benchmark a function and return mean and std of execution time in nanoseconds.

    Returns:
        Tuple of (mean_ns, std_ns)
    """
    # Warmup
    for _ in range(n_warmup):
        fn()

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        result = fn()
        # Handle JAX arrays that need blocking
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        end = time.perf_counter_ns()
        times.append(end - start)

    return np.mean(times), np.std(times)


# =============================================================================
# Library Wrappers
# =============================================================================


class BenchmarkRunner:
    """Runs benchmarks for a specific configuration."""

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        k: int,
        batch_size: int,
        seed: int = 42,
    ):
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.k = k
        self.batch_size = batch_size

        # Generate data
        np.random.seed(seed)
        self.embeddings = np.random.randn(n_embeddings, embedding_dim).astype(
            np.float32
        )
        self.queries = np.random.randn(batch_size, embedding_dim).astype(np.float32)
        self.single_query = self.queries[0]

        # Will hold initialized libraries
        self._bolt = None
        self._faiss_index = None
        self._sklearn_nn = None

    def setup_boltembedding(self) -> float:
        """Setup BoltEmbedding and return setup time in ns."""
        start = time.perf_counter_ns()
        self._bolt = BoltEmbedding(
            embeddings=self.embeddings,
            metric=DistanceMetric.EUCLIDEAN,
            device=Device.CPU,
        )
        self._bolt.index(k=self.k)
        # Warmup JIT
        _ = self._bolt.query(self.single_query)
        end = time.perf_counter_ns()
        return end - start

    def setup_faiss(self) -> float:
        """Setup FAISS and return setup time in ns."""
        if not HAS_FAISS:
            return 0

        start = time.perf_counter_ns()
        self._faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self._faiss_index.add(self.embeddings)
        end = time.perf_counter_ns()
        return end - start

    def setup_sklearn(self) -> float:
        """Setup scikit-learn and return setup time in ns."""
        if not HAS_SKLEARN:
            return 0

        start = time.perf_counter_ns()
        self._sklearn_nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self._sklearn_nn.fit(self.embeddings)
        end = time.perf_counter_ns()
        return end - start

    def benchmark_boltembedding_single(
        self, n_warmup: int, n_iterations: int
    ) -> Tuple[float, float]:
        """Benchmark BoltEmbedding single query."""

        def query_fn():
            return self._bolt.query(self.single_query)

        return benchmark_function(query_fn, n_warmup, n_iterations)

    def benchmark_boltembedding_batch(
        self, n_warmup: int, n_iterations: int
    ) -> Tuple[float, float]:
        """Benchmark BoltEmbedding batch query."""

        def query_fn():
            return self._bolt.query(self.queries)

        return benchmark_function(query_fn, n_warmup, n_iterations)

    def benchmark_faiss_single(
        self, n_warmup: int, n_iterations: int
    ) -> Tuple[float, float]:
        """Benchmark FAISS single query."""
        if not HAS_FAISS or self._faiss_index is None:
            return 0, 0

        query = self.single_query.reshape(1, -1)

        def query_fn():
            return self._faiss_index.search(query, self.k)

        return benchmark_function(query_fn, n_warmup, n_iterations)

    def benchmark_faiss_batch(
        self, n_warmup: int, n_iterations: int
    ) -> Tuple[float, float]:
        """Benchmark FAISS batch query."""
        if not HAS_FAISS or self._faiss_index is None:
            return 0, 0

        def query_fn():
            return self._faiss_index.search(self.queries, self.k)

        return benchmark_function(query_fn, n_warmup, n_iterations)

    def benchmark_sklearn_single(
        self, n_warmup: int, n_iterations: int
    ) -> Tuple[float, float]:
        """Benchmark scikit-learn single query."""
        if not HAS_SKLEARN or self._sklearn_nn is None:
            return 0, 0

        query = self.single_query.reshape(1, -1)

        def query_fn():
            return self._sklearn_nn.kneighbors(query, return_distance=False)

        return benchmark_function(query_fn, n_warmup, n_iterations)

    def benchmark_sklearn_batch(
        self, n_warmup: int, n_iterations: int
    ) -> Tuple[float, float]:
        """Benchmark scikit-learn batch query."""
        if not HAS_SKLEARN or self._sklearn_nn is None:
            return 0, 0

        def query_fn():
            return self._sklearn_nn.kneighbors(self.queries, return_distance=False)

        return benchmark_function(query_fn, n_warmup, n_iterations)


# =============================================================================
# Main Benchmark
# =============================================================================


def run_benchmark(
    n_embeddings: int,
    embedding_dim: int,
    k: int,
    batch_size: int,
    n_warmup: int = 10,
    n_iterations: int = 100,
    verbose: bool = True,
) -> Dict:
    """Run complete benchmark suite for a configuration."""

    if verbose:
        print(f"\n{'='*70}")
        print(
            f"Benchmark: N={n_embeddings:,}, D={embedding_dim}, k={k}, batch={batch_size}"
        )
        print(f"{'='*70}")

    runner = BenchmarkRunner(n_embeddings, embedding_dim, k, batch_size)
    results = {
        "config": {
            "n_embeddings": n_embeddings,
            "embedding_dim": embedding_dim,
            "k": k,
            "batch_size": batch_size,
        },
        "setup": {},
        "single_query": {},
        "batch_query": {},
    }

    # Setup benchmarks
    if verbose:
        print("\n--- Setup Time ---")

    bolt_setup = runner.setup_boltembedding()
    results["setup"]["boltembedding"] = bolt_setup
    if verbose:
        print(f"BoltEmbedding: {format_time(bolt_setup)}")

    if HAS_FAISS:
        faiss_setup = runner.setup_faiss()
        results["setup"]["faiss"] = faiss_setup
        if verbose:
            print(f"FAISS:         {format_time(faiss_setup)}")

    if HAS_SKLEARN:
        sklearn_setup = runner.setup_sklearn()
        results["setup"]["sklearn"] = sklearn_setup
        if verbose:
            print(f"scikit-learn:  {format_time(sklearn_setup)}")

    # Single query benchmarks
    if verbose:
        print("\n--- Single Query Latency ---")

    bolt_single = runner.benchmark_boltembedding_single(n_warmup, n_iterations)
    results["single_query"]["boltembedding"] = bolt_single
    if verbose:
        print(
            f"BoltEmbedding: {format_time(bolt_single[0])} ± {format_time(bolt_single[1])}"
        )

    if HAS_FAISS:
        faiss_single = runner.benchmark_faiss_single(n_warmup, n_iterations)
        results["single_query"]["faiss"] = faiss_single
        if verbose:
            print(
                f"FAISS:         {format_time(faiss_single[0])} ± {format_time(faiss_single[1])}"
            )

    if HAS_SKLEARN:
        sklearn_single = runner.benchmark_sklearn_single(n_warmup, n_iterations)
        results["single_query"]["sklearn"] = sklearn_single
        if verbose:
            print(
                f"scikit-learn:  {format_time(sklearn_single[0])} ± {format_time(sklearn_single[1])}"
            )

    # Batch query benchmarks
    if verbose:
        print(f"\n--- Batch Query Latency (batch_size={batch_size}) ---")

    bolt_batch = runner.benchmark_boltembedding_batch(n_warmup, n_iterations)
    results["batch_query"]["boltembedding"] = bolt_batch
    if verbose:
        per_query = bolt_batch[0] / batch_size
        print(
            f"BoltEmbedding: {format_time(bolt_batch[0])} total, {format_time(per_query)}/query"
        )

    if HAS_FAISS:
        faiss_batch = runner.benchmark_faiss_batch(n_warmup, n_iterations)
        results["batch_query"]["faiss"] = faiss_batch
        if verbose:
            per_query = faiss_batch[0] / batch_size
            print(
                f"FAISS:         {format_time(faiss_batch[0])} total, {format_time(per_query)}/query"
            )

    if HAS_SKLEARN:
        sklearn_batch = runner.benchmark_sklearn_batch(n_warmup, n_iterations)
        results["batch_query"]["sklearn"] = sklearn_batch
        if verbose:
            per_query = sklearn_batch[0] / batch_size
            print(
                f"scikit-learn:  {format_time(sklearn_batch[0])} total, {format_time(per_query)}/query"
            )

    # Speedup summary
    if verbose and HAS_FAISS:
        print("\n--- Speedup vs FAISS ---")
        if faiss_single[0] > 0:
            speedup_single = faiss_single[0] / bolt_single[0]
            print(
                f"Single query: {speedup_single:.2f}x {'faster' if speedup_single > 1 else 'slower'}"
            )
        if faiss_batch[0] > 0:
            speedup_batch = faiss_batch[0] / bolt_batch[0]
            print(
                f"Batch query:  {speedup_batch:.2f}x {'faster' if speedup_batch > 1 else 'slower'}"
            )

    return results


def run_full_benchmark_suite(config: Dict = None, verbose: bool = True) -> List[Dict]:
    """Run complete benchmark suite with multiple configurations."""
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []

    print("\n" + "=" * 70)
    print(" BoltEmbedding Benchmark Suite")
    print("=" * 70)
    print(f"\nLibraries available:")
    print(f"  - BoltEmbedding: ✓")
    print(f"  - FAISS: {'✓' if HAS_FAISS else '✗ (not installed)'}")
    print(f"  - scikit-learn: {'✓' if HAS_SKLEARN else '✗ (not installed)'}")

    # Run benchmarks for different configurations
    for n in config.get("n_embeddings", [10000]):
        for dim in config.get("embedding_dims", [768]):
            for k in config.get("k_values", [10]):
                for batch in config.get("batch_sizes", [1]):
                    try:
                        results = run_benchmark(
                            n_embeddings=n,
                            embedding_dim=dim,
                            k=k,
                            batch_size=batch,
                            n_warmup=config.get("n_warmup", 10),
                            n_iterations=config.get("n_iterations", 100),
                            verbose=verbose,
                        )
                        all_results.append(results)
                    except Exception as e:
                        print(
                            f"\nError in benchmark (N={n}, D={dim}, k={k}, batch={batch}): {e}"
                        )

    return all_results


def print_summary_table(results: List[Dict]) -> None:
    """Print a detailed summary matrix of all benchmark results."""

    # ==========================================================================
    # Setup Times Table
    # ==========================================================================
    print("\n" + "=" * 100)
    print(" SETUP / INDEX BUILD TIME")
    print("=" * 100)

    setup_header = (
        f"{'N':>10} | {'D':>5} | {'k':>4} | "
        f"{'Bolt':>14} | {'FAISS':>14} | {'sklearn':>14} | "
        f"{'vs FAISS':>10} | {'vs sklearn':>10}"
    )
    print(setup_header)
    print("-" * 100)

    for r in results:
        cfg = r["config"]
        bolt_setup = r["setup"].get("boltembedding", 0)
        faiss_setup = r["setup"].get("faiss", 0)
        sklearn_setup = r["setup"].get("sklearn", 0)

        vs_faiss = faiss_setup / bolt_setup if bolt_setup > 0 and faiss_setup > 0 else 0
        vs_sklearn = (
            sklearn_setup / bolt_setup if bolt_setup > 0 and sklearn_setup > 0 else 0
        )

        print(
            f"{cfg['n_embeddings']:>10,} | {cfg['embedding_dim']:>5} | {cfg['k']:>4} | "
            f"{format_time(bolt_setup):>14} | {format_time(faiss_setup) if faiss_setup else 'N/A':>14} | "
            f"{format_time(sklearn_setup) if sklearn_setup else 'N/A':>14} | "
            f"{vs_faiss:>9.2f}x | {vs_sklearn:>9.2f}x"
        )

    # ==========================================================================
    # Single Query Latency Table
    # ==========================================================================
    print("\n" + "=" * 120)
    print(" SINGLE QUERY LATENCY (mean ± std)")
    print("=" * 120)

    single_header = (
        f"{'N':>10} | {'D':>5} | {'k':>4} | "
        f"{'Bolt':>18} | {'FAISS':>18} | {'sklearn':>18} | "
        f"{'vs FAISS':>10} | {'vs sklearn':>10}"
    )
    print(single_header)
    print("-" * 120)

    for r in results:
        cfg = r["config"]
        bolt_mean, bolt_std = r["single_query"].get("boltembedding", (0, 0))
        faiss_mean, faiss_std = r["single_query"].get("faiss", (0, 0))
        sklearn_mean, sklearn_std = r["single_query"].get("sklearn", (0, 0))

        vs_faiss = faiss_mean / bolt_mean if bolt_mean > 0 and faiss_mean > 0 else 0
        vs_sklearn = (
            sklearn_mean / bolt_mean if bolt_mean > 0 and sklearn_mean > 0 else 0
        )

        bolt_str = f"{format_time(bolt_mean)}"
        faiss_str = f"{format_time(faiss_mean)}" if faiss_mean else "N/A"
        sklearn_str = f"{format_time(sklearn_mean)}" if sklearn_mean else "N/A"

        print(
            f"{cfg['n_embeddings']:>10,} | {cfg['embedding_dim']:>5} | {cfg['k']:>4} | "
            f"{bolt_str:>18} | {faiss_str:>18} | {sklearn_str:>18} | "
            f"{vs_faiss:>9.2f}x | {vs_sklearn:>9.2f}x"
        )

    # ==========================================================================
    # Batch Query Latency Table
    # ==========================================================================
    print("\n" + "=" * 140)
    print(" BATCH QUERY PERFORMANCE (total time | per-query latency | throughput)")
    print("=" * 140)

    batch_header = (
        f"{'N':>10} | {'D':>5} | {'k':>4} | {'Batch':>5} | "
        f"{'Bolt Total':>12} | {'Bolt/q':>10} | {'Bolt QPS':>12} | "
        f"{'FAISS Total':>12} | {'FAISS/q':>10} | "
        f"{'Speedup':>8}"
    )
    print(batch_header)
    print("-" * 140)

    for r in results:
        cfg = r["config"]
        batch_size = cfg["batch_size"]

        bolt_mean, _ = r["batch_query"].get("boltembedding", (0, 0))
        faiss_mean, _ = r["batch_query"].get("faiss", (0, 0))

        bolt_per_q = bolt_mean / batch_size if batch_size > 0 else 0
        faiss_per_q = faiss_mean / batch_size if batch_size > 0 else 0

        # Queries per second (QPS)
        bolt_qps = 1e9 * batch_size / bolt_mean if bolt_mean > 0 else 0

        vs_faiss = faiss_mean / bolt_mean if bolt_mean > 0 and faiss_mean > 0 else 0

        print(
            f"{cfg['n_embeddings']:>10,} | {cfg['embedding_dim']:>5} | {cfg['k']:>4} | {batch_size:>5} | "
            f"{format_time(bolt_mean):>12} | {format_time(bolt_per_q):>10} | {bolt_qps:>10,.0f}/s | "
            f"{format_time(faiss_mean) if faiss_mean else 'N/A':>12} | "
            f"{format_time(faiss_per_q) if faiss_per_q else 'N/A':>10} | "
            f"{vs_faiss:>7.2f}x"
        )

    # ==========================================================================
    # Aggregate Statistics
    # ==========================================================================
    if len(results) > 1:
        print("\n" + "=" * 80)
        print(" AGGREGATE STATISTICS")
        print("=" * 80)

        # Collect speedups
        single_speedups_faiss = []
        single_speedups_sklearn = []
        batch_speedups_faiss = []

        for r in results:
            bolt_single = r["single_query"].get("boltembedding", (0, 0))[0]
            faiss_single = r["single_query"].get("faiss", (0, 0))[0]
            sklearn_single = r["single_query"].get("sklearn", (0, 0))[0]
            bolt_batch = r["batch_query"].get("boltembedding", (0, 0))[0]
            faiss_batch = r["batch_query"].get("faiss", (0, 0))[0]

            if bolt_single > 0 and faiss_single > 0:
                single_speedups_faiss.append(faiss_single / bolt_single)
            if bolt_single > 0 and sklearn_single > 0:
                single_speedups_sklearn.append(sklearn_single / bolt_single)
            if bolt_batch > 0 and faiss_batch > 0:
                batch_speedups_faiss.append(faiss_batch / bolt_batch)

        print(f"\nSpeedup vs FAISS (single query):")
        if single_speedups_faiss:
            print(
                f"  Min: {min(single_speedups_faiss):.2f}x | "
                f"Max: {max(single_speedups_faiss):.2f}x | "
                f"Avg: {np.mean(single_speedups_faiss):.2f}x | "
                f"Median: {np.median(single_speedups_faiss):.2f}x"
            )
        else:
            print("  N/A (FAISS not available)")

        print(f"\nSpeedup vs scikit-learn (single query):")
        if single_speedups_sklearn:
            print(
                f"  Min: {min(single_speedups_sklearn):.2f}x | "
                f"Max: {max(single_speedups_sklearn):.2f}x | "
                f"Avg: {np.mean(single_speedups_sklearn):.2f}x | "
                f"Median: {np.median(single_speedups_sklearn):.2f}x"
            )
        else:
            print("  N/A (scikit-learn not available)")

        print(f"\nSpeedup vs FAISS (batch query):")
        if batch_speedups_faiss:
            print(
                f"  Min: {min(batch_speedups_faiss):.2f}x | "
                f"Max: {max(batch_speedups_faiss):.2f}x | "
                f"Avg: {np.mean(batch_speedups_faiss):.2f}x | "
                f"Median: {np.median(batch_speedups_faiss):.2f}x"
            )
        else:
            print("  N/A (FAISS not available)")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BoltEmbedding against FAISS and scikit-learn"
    )
    parser.add_argument(
        "--n-embeddings",
        "-n",
        type=int,
        nargs="+",
        default=[1000, 10000],
        help="Number of embeddings to benchmark",
    )
    parser.add_argument(
        "--embedding-dim",
        "-d",
        type=int,
        nargs="+",
        default=[768],
        help="Embedding dimensions to benchmark",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[10], help="k values to benchmark"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        nargs="+",
        default=[1, 10],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--n-warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--n-iterations", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Only print summary table"
    )

    args = parser.parse_args()

    config = {
        "n_embeddings": args.n_embeddings,
        "embedding_dims": args.embedding_dim,
        "k_values": args.k,
        "batch_sizes": args.batch_size,
        "n_warmup": args.n_warmup,
        "n_iterations": args.n_iterations,
    }

    results = run_full_benchmark_suite(config, verbose=not args.quiet)

    if len(results) > 1:
        print_summary_table(results)

    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
