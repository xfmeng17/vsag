# Pyramid `no_build_levels` 内存对比工具 (`pyramid_level_memory`)

[\[EN\]](README.md)

用于度量 **Pyramid** 索引的 `no_build_levels` 参数在内存占用、构建耗时
和中间层搜索能力之间的权衡。面向需要在多粒度(如 日 / 月 / 年)path 场景下
调优 Pyramid 的 VSAG 开发者。

## 背景

Pyramid 索引会把每条 `path`(例如 `"2026/05/19"`)按 `/` 切片,并在每一层
为每个前缀建立一个 HGraph 子图。`no_build_levels` 是 0-based 的层索引数组,
列出的层只挂节点不建图(向量仍存,但该层无法做图搜索)。

### 层级编号

Pyramid 内部有一个**虚拟根节点**,占 level 0。path 切片对应的层级从 1 开始
往下递增。所以 `"2026/05/19"` 的层级映射是:

| Level | 节点              |
| ----- | ----------------- |
| 0     | 虚拟根            |
| 1     | `2026`(年)      |
| 2     | `05`(月)        |
| 3     | `19`(日)        |

因此**"只建 day"对应的 `no_build_levels` 是 `[0, 1, 2]`,而不是 `[0, 1]`**。
这是个容易踩错的细节,根因在 `src/algorithm/pyramid.cpp::Build` 中:`root_->level_`
的判断在 path slice 循环**之前**就被消费掉了。

工具回答一个工程问题:

> 如果只在叶子层查询,把上层都跳过能省多少内存?

## 测试内容

1. 生成 1,000,000 条 fp32 × 128 的随机向量。
2. 给每条向量分配形如 `2026/05/DD` 的 path(DD = `01..30`,每天约 33 333 条)。
3. 用 `--config` 选定的一组配置**只建一次**索引。
4. 在四个采样点读取 `/proc/self/status` 的 `VmRSS`,并打印原始值和两个 delta。
5. **每组配置一个独立进程。** 配套脚本 `run_all.sh` 串行起 4 个进程,每个跑
   一组配置。这样可以避免 glibc 分配器在同进程内复用内存池而污染 RSS 对比。

固定参数:

| 参数                    | 取值       |
| ----------------------- | ---------- |
| `dtype`                 | `float32`  |
| `dim`                   | 128        |
| `metric_type`           | `l2`       |
| `base_quantization`     | `fp32`     |
| `max_degree`            | 32         |
| `ef_construction`       | 200        |
| `graph_type`            | `nsw`      |
| `build_thread_count`    | 16         |

对比的四组配置:

| 标签              | `no_build_levels` | 实际建图层级               |
| ----------------- | ----------------- | -------------------------- |
| `only-day`        | `[0, 1, 2]`       | day                        |
| `month-day`       | `[0, 1]`          | month + day                |
| `year-month-day`  | `[0]`             | year + month + day         |
| `all`             | `[]`              | root + year + month + day  |

## 编译与运行

工具已接入 CMake 构建:

```bash
cmake --build . --target pyramid_level_memory
```

每次只跑一组(推荐做法,这样进程间彻底隔离):

```bash
./tools/pyramid_level_memory/pyramid_level_memory --config only-day
./tools/pyramid_level_memory/pyramid_level_memory --config month-day
./tools/pyramid_level_memory/pyramid_level_memory --config year-month-day
./tools/pyramid_level_memory/pyramid_level_memory --config all
```

或者用脚本一口气串行跑完 4 组:

```bash
<source-dir>/tools/pyramid_level_memory/run_all.sh \
    ./tools/pyramid_level_memory/pyramid_level_memory
```

一次完整运行在 arm64、16 线程、无 AVX/MKL 的环境下大约 60 分钟。

## RSS 采样点和应该用哪个数比较

每一组都会打印 4 个绝对 RSS 和 2 个 delta:

| 采样点                    | 含义                                                              |
| ------------------------- | ----------------------------------------------------------------- |
| `RSS before CreateIndex`  | 进程启动 + 输入数据已分配                                         |
| `RSS before Build`        | 多了一个空的 Pyramid 骨架                                         |
| `RSS after  Build`        | 多了 Build 阶段分配的所有内容(子图 / allocator / buffer 等)    |
| `RSS after  ReleaseInput` | 显式析构输入数组之后,接近**纯索引**占用                          |

派生 delta:

| Delta                  | 含义                                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| `delta (Build only)`   | 严格在 `Build()` 期间增长的 RSS;用于校准峰值                                                         |
| `delta (index-only)`   | `rss_after_release_input - rss_before_create`;**横向对比时用这个**                                    |

之所以推荐 `delta (index-only)`:

- 它扣除了 ~520 MB 的输入数据开销(100w × 128 fp32 + ids + path 字符串),
  这部分在所有配置下都一样;
- 每组跑独立进程,glibc 分配器状态不会跨配置串扰;
- 它正确反映了配置实际需要长期持有的内存(包括 allocator 持有但未还给 OS 的部分)。

## 输出示例

在 arm64 开发容器 (`vsag-dev-arm64`, 16 线程构建, 无 AVX/MKL) 下,每组
配置一个独立进程实测:

| 配置             | 构建耗时 (s) | RSS after Build (MB) | RSS index-only (MB) | 相比 `only-day` |
| ---------------- | -----------: | -------------------: | ------------------: | --------------: |
| `only-day`       |        108.4 |              1715.99 |              647.14 |               — |
| `month-day`      |        665.6 |              1921.74 |              853.18 |        +31.8 %  |
| `year-month-day` |        807.6 |              2152.38 |             1083.80 |        +67.5 %  |
| `all`            |       1046.3 |              2376.48 |             1307.92 |       +102.1 %  |

结论:

- 仅建叶子层就要在输入数据之外再吃 ~647 MB。
- 每多建一层,index-only RSS 会再涨 ~200–230 MB。从 only-day 升到
  完整金字塔,RSS 大致**翻倍**。
- "向量是一份,只是邻居表多两份" 的直觉方向上没错,但低估了实际开销:
  `max_degree=32` 下,1M 节点光邻居表就 ~128 MB / 层,叠加 allocator
  slack、子图元数据、label 簿记,实际每多建一层会涨到 ~200 MB。
- 构建耗时也不是线性增长:上层节点更少更密,subgraph 体量更大,反而
  比叶子层更贵。

## 注意事项

- 工具依赖 `/proc/self/status`,仅能在 Linux 下运行。
- 构建耗时不能直接跨硬件比较;本工具关心的是内存相对差异,不是绝对吞吐。
- Path 分布是严格均衡(`i % 30`)。真实业务里 path 分布不均时,趋势类似
  但具体数值会有差异。
- 当前 `Pyramid::GetMemoryUsage()` 在 Build 之后并未更新,因此本工具退而
  使用 OS 级别的 RSS。如果未来 Pyramid 实现了完整的 memory accounting,
  本工具应该切换到 `index->GetMemoryUsage()`。

## License

该工具根据 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) 授权。
