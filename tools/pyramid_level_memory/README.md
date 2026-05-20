# Pyramid `no_build_levels` Memory Benchmark (`pyramid_level_memory`)

[\[中文\]](README_zh.md)

A benchmark that measures how `no_build_levels` of the **Pyramid** index
trades off memory footprint for build cost and intermediate-layer search
capability. It is intended for VSAG developers who tune Pyramid for
multi-granularity (e.g. day / month / year) path scenarios.

## Background

The Pyramid index splits each `path` (e.g. `"2026/05/19"`) by `/` and
builds an HGraph subgraph per prefix at every level. The
`no_build_levels` parameter is a 0-based list of level indices whose
subgraphs are skipped (vectors are still attached, but the level cannot
serve graph search).

### Level numbering

Pyramid uses an internal **virtual root** as level 0. Path slices then
occupy levels 1, 2, 3, ... So for `"2026/05/19"`:

| Level | Node           |
| ----- | -------------- |
| 0     | virtual root   |
| 1     | `2026` (year)  |
| 2     | `05` (month)   |
| 3     | `19` (day)     |

Therefore "build day only" needs `no_build_levels = [0, 1, 2]`,
**not** `[0, 1]`. This is easy to get wrong; see
`src/algorithm/pyramid.cpp::Build` where `root_->level_` is checked
*before* the path slice loop.

This benchmark answers the practical question:

> If I only ever query at the leaf level, how much RAM do I save by
> skipping the upper levels?

## What It Does

1. Generates 1,000,000 random fp32 vectors of dimension 128.
2. Assigns paths in the form `2026/05/DD` (DD = `01..30`, ~33 333
   vectors per day).
3. Builds the Pyramid index **once** with the configuration selected via
   `--config`.
4. Samples `VmRSS` (from `/proc/self/status`) at four points and prints
   both raw and derived deltas.
5. **Runs one config per process.** A companion script `run_all.sh`
   launches one process per configuration sequentially. This avoids
   glibc allocator pool reuse leaking RSS state across configurations.

Fixed parameters:

| Parameter             | Value           |
| --------------------- | --------------- |
| `dtype`               | `float32`       |
| `dim`                 | 128             |
| `metric_type`         | `l2`            |
| `base_quantization`   | `fp32`          |
| `max_degree`          | 32              |
| `ef_construction`     | 200             |
| `graph_type`          | `nsw`           |
| `build_thread_count`  | 16              |

Compared configurations:

| Tag              | `no_build_levels` | Levels actually built          |
| ---------------- | ----------------- | ------------------------------ |
| `only-day`       | `[0, 1, 2]`       | day                            |
| `month-day`      | `[0, 1]`          | month + day                    |
| `year-month-day` | `[0]`             | year + month + day             |
| `all`            | `[]`              | root + year + month + day      |

## Build And Run

This is a normal CMake target. From the build directory:

```bash
cmake --build . --target pyramid_level_memory
```

Run a single configuration (one config per process is the recommended
way to compare):

```bash
./tools/pyramid_level_memory/pyramid_level_memory --config only-day
./tools/pyramid_level_memory/pyramid_level_memory --config month-day
./tools/pyramid_level_memory/pyramid_level_memory --config year-month-day
./tools/pyramid_level_memory/pyramid_level_memory --config all
```

Or use the helper script to run all four sequentially:

```bash
<source-dir>/tools/pyramid_level_memory/run_all.sh \
    ./tools/pyramid_level_memory/pyramid_level_memory
```

A full run takes roughly 60 minutes on a 16-thread arm64 machine
without AVX/MKL.

## RSS Sample Points And Which Number To Compare

Each run emits four absolute RSS values and two derived deltas:

| Sample point             | What it captures                                           |
| ------------------------ | ---------------------------------------------------------- |
| `RSS before CreateIndex` | baseline + input data already allocated                    |
| `RSS before Build`       | + the empty Pyramid skeleton                               |
| `RSS after  Build`       | + everything Build allocates (graphs, allocators, buffers) |
| `RSS after  ReleaseInput`| input arrays explicitly destroyed; close to **index-only** |

Derived:

| Delta                  | What it means                                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------------------------- |
| `delta (Build only)`   | RSS growth strictly during `Build()`; useful for sanity-checking peak                                     |
| `delta (index-only)`   | `rss_after_release_input - rss_before_create`; **use this for cross-config comparison**                   |

`delta (index-only)` is the most honest cross-configuration metric
because:

- it excludes the ~520 MB of input data (1M × 128 fp32 + ids + path
  strings) that all configurations pay,
- each config runs in a fresh process, so glibc allocator state cannot
  leak across runs,
- it accounts for all allocator-retained memory the configuration
  actually needs to keep the index alive after Build.

## Sample Output

Measured on the arm64 dev container (`vsag-dev-arm64`, 16 build threads,
no AVX/MKL), one process per configuration:

| Configuration    | Build time (s) | RSS after Build (MB) | RSS index-only (MB) | vs `only-day` |
| ---------------- | -------------: | -------------------: | ------------------: | ------------: |
| `only-day`       |          108.4 |              1715.99 |              647.14 |             — |
| `month-day`      |          665.6 |              1921.74 |              853.18 |       +31.8 % |
| `year-month-day` |          807.6 |              2152.38 |             1083.80 |       +67.5 % |
| `all`            |         1046.3 |              2376.48 |             1307.92 |      +102.1 % |

Takeaways:

- Building the leaf layer alone costs ~647 MB on top of the input data.
- Each additional upper level adds another ~200–230 MB. Going from
  leaf-only to a full pyramid roughly **doubles** the index-only RSS.
- The "the vector data is shared, only neighbor lists multiply"
  intuition is qualitatively right but underestimates the cost. At
  `max_degree=32` the neighbor lists alone are ~128 MB per layer of
  1 M nodes, but allocator slack, per-subgraph metadata, and label
  bookkeeping push that to ~200 MB per built upper level.
- Build time grows superlinearly with the number of built levels: the
  upper levels concentrate work onto fewer, larger subgraphs, so they
  are not free either in CPU.

## Caveats

- The benchmark relies on `/proc/self/status` and only runs on Linux.
- Raw build times are not directly comparable across hardware; we care
  about the relative memory differences, not absolute throughput.
- Path distribution is perfectly balanced (`i % 30`). Real workloads
  with skewed path distributions will see different but qualitatively
  similar trends.
- `Pyramid::GetMemoryUsage()` is currently not maintained after Build,
  so this tool falls back to OS-level RSS. If Pyramid grows proper
  memory accounting in the future, this benchmark should switch to
  `index->GetMemoryUsage()`.

## License

This tool is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
