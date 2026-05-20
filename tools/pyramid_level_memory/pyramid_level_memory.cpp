// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Pyramid no_build_levels 内存对比工具(单配置一次进程)。
//
// 用法:
//   pyramid_level_memory --config <only-day|month-day|year-month-day|all>
//
// 数据规模: 1,000,000 条 fp32 x 128, base_quantization=fp32, max_degree=32。
// path 形态: "2026/05/DD" (DD = 01..30, ~33,333 / 天)。
//
// 重要: Pyramid 的 root_ 节点本身占 level 0(虚拟根),所以
//   path "2026/05/19" 的层级映射为:
//     level 0 = root, level 1 = year(2026), level 2 = month(05), level 3 = day(19)
// 因此 "只建 day" 对应 no_build_levels=[0,1,2],而不是 [0,1]。

#include <vsag/vsag.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int64_t kNumVectors = 1'000'000;
constexpr int64_t kDim = 128;
constexpr int kNumDays = 30;

struct ConfigSpec {
    const char* name;
    const char* no_build_levels_json;
    const char* description;
};

constexpr ConfigSpec kConfigs[] = {
    {"only-day", "[0, 1, 2]", "skip root + year + month, build day only"},
    {"month-day", "[0, 1]", "skip root + year, build month + day"},
    {"year-month-day", "[0]", "skip root, build year + month + day"},
    {"all", "[]", "build root + year + month + day (full pyramid)"},
};

// 读取 /proc/self/status 的 VmRSS(驻留集大小,单位 KB),返回字节数。
uint64_t
read_rss_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            uint64_t kb = 0;
            std::string unit;
            iss >> kb >> unit;
            return kb * 1024UL;
        }
    }
    return 0;
}

std::string
make_index_param(const std::string& no_build_levels) {
    std::ostringstream oss;
    oss << R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": )" << kDim << R"(,
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "alpha": 1.2,
        "ef_construction": 200,
        "graph_type": "nsw",
        "no_build_levels": )" << no_build_levels << R"(,
        "use_reorder": false,
        "support_duplicate": false,
        "index_min_size": 0,
        "build_thread_count": 16
    }
})";
    return oss.str();
}

void
print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --config <name>\n"
              << "Available configs:\n";
    for (const ConfigSpec& cfg : kConfigs) {
        std::cerr << "  " << cfg.name << "  no_build_levels=" << cfg.no_build_levels_json
                  << "  (" << cfg.description << ")\n";
    }
}

const ConfigSpec*
find_config(const std::string& name) {
    for (const ConfigSpec& cfg : kConfigs) {
        if (name == cfg.name) {
            return &cfg;
        }
    }
    return nullptr;
}

double
to_mb(uint64_t bytes) {
    return static_cast<double>(bytes) / 1024.0 / 1024.0;
}

int
run_one_config(const ConfigSpec& cfg) {
    std::cout << "===== " << cfg.name
              << " (no_build_levels=" << cfg.no_build_levels_json << ") =====" << std::endl;
    std::cout << "  description            : " << cfg.description << std::endl;

    std::vector<int64_t> ids(kNumVectors);
    std::vector<float> vectors(kDim * kNumVectors);
    std::vector<std::string> paths(kNumVectors);

    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (int64_t i = 0; i < kNumVectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < kDim * kNumVectors; ++i) {
        vectors[i] = dist(rng);
    }
    char path_buf[16];
    for (int64_t i = 0; i < kNumVectors; ++i) {
        int day = static_cast<int>(i % kNumDays) + 1;
        std::snprintf(path_buf, sizeof(path_buf), "2026/05/%02d", day);
        paths[i] = path_buf;
    }

    vsag::DatasetPtr base = vsag::Dataset::Make();
    base->NumElements(kNumVectors)
        ->Dim(kDim)
        ->Ids(ids.data())
        ->Paths(paths.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    std::string build_param = make_index_param(cfg.no_build_levels_json);

    uint64_t rss_before_create = read_rss_bytes();
    auto index_expected = vsag::Factory::CreateIndex("pyramid", build_param);
    if (not index_expected.has_value()) {
        std::cerr << "CreateIndex failed: " << index_expected.error().message << std::endl;
        return 1;
    }
    std::shared_ptr<vsag::Index> index = index_expected.value();

    uint64_t rss_before_build = read_rss_bytes();
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    auto build_result = index->Build(base);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    double build_sec = std::chrono::duration<double>(t1 - t0).count();
    if (not build_result.has_value()) {
        std::cerr << "Build failed: " << build_result.error().message << std::endl;
        return 1;
    }
    uint64_t rss_after_build = read_rss_bytes();

    // 释放输入数据(base 是 Owner(false),底层数组归三个 std::vector 所有,
    // 这里把它们整体析构掉,然后再采一次 RSS 作为近似的 "index-only" 占用。
    base.reset();
    std::vector<int64_t>().swap(ids);
    std::vector<float>().swap(vectors);
    std::vector<std::string>().swap(paths);
    uint64_t rss_index_only = read_rss_bytes();

    std::cout << "  num_elements           : " << index->GetNumElements() << std::endl;
    std::cout << "  build_time_sec         : " << build_sec << std::endl;
    std::cout << "  RSS before CreateIndex : " << to_mb(rss_before_create) << " MB" << std::endl;
    std::cout << "  RSS before Build       : " << to_mb(rss_before_build) << " MB" << std::endl;
    std::cout << "  RSS after  Build       : " << to_mb(rss_after_build) << " MB" << std::endl;
    std::cout << "  RSS after  ReleaseInput: " << to_mb(rss_index_only) << " MB"
              << "   (approx index-only)" << std::endl;
    std::cout << "  delta (Build only)     : " << to_mb(rss_after_build - rss_before_build)
              << " MB" << std::endl;
    std::cout << "  delta (index-only)     : " << to_mb(rss_index_only - rss_before_create)
              << " MB   (recommended for cross-config compare)" << std::endl;
    std::cout << std::endl;

    return 0;
}

}  // namespace

int
main(int argc, char** argv) {
    std::string config_name;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_name = argv[i + 1];
            ++i;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config_name.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    const ConfigSpec* cfg = find_config(config_name);
    if (cfg == nullptr) {
        std::cerr << "Unknown config: " << config_name << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    return run_one_config(*cfg);
}
