
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

// 生成 hybrid HDF5(dense 单向量粗排 + multi-vector MaxSim 精排)。
// schema 见 data/xfmeng17-dev/task/2026-06-10-subtask1-hdf5-spec.md。
// MaxSim 距离用 vsag::FP32ComputeIP 的 `1 - dot` 约定,与
// src/quantization/multi_vector_computer.cpp 对齐,保证 oracle 不反向。

#include <H5Cpp.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "simd/fp32_simd.h"

namespace {

// 规模常量,与子任务 1 规格一致。
constexpr int64_t kN = 64;   // base doc 数
constexpr int64_t kQ = 8;    // query 数
constexpr int64_t kD = 8;    // 维度(dense dim == multi_vector_dim)
constexpr int64_t kTopK = 10;  // top-K
constexpr uint32_t kQueryTokens = 2;  // 每个 query 固定 2 token
const std::string kDefaultOutDir = "data/xfmeng17-dev/demo/mv_reorder";

// IP per-token 距离,VSAG 约定:越小越好。
float
PerTokenDistIp(const float* q, const float* d) {
    return 1.0F - vsag::FP32ComputeIP(q, d, static_cast<uint64_t>(kD));
}

// MaxSim: dist(Q,D) = sum_{q in Q} min_{d in D} per_token_dist(q,d)。
float
MaxSimDist(const float* query_tokens,
           uint32_t query_len,
           const float* doc_tokens,
           uint32_t doc_len) {
    float total = 0.0F;
    for (uint32_t qi = 0; qi < query_len; ++qi) {
        const float* q = query_tokens + static_cast<int64_t>(qi) * kD;
        float min_dist = std::numeric_limits<float>::max();
        for (uint32_t di = 0; di < doc_len; ++di) {
            const float* d = doc_tokens + static_cast<int64_t>(di) * kD;
            float dist = PerTokenDistIp(q, d);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        total += min_dist;
    }
    return total;
}

// 平铺多向量:flat[(sum_tokens, D)] + counts[(count,)]。
struct FlatMultiVectors {
    std::vector<float> flat;       // 所有 token 平铺,大小 sum_tokens * D
    std::vector<uint32_t> counts;  // 每个文档/query 的 token 数
    std::vector<int64_t> offsets;  // 累积偏移(本地用,不写盘),大小 count+1
};

int64_t
TotalTokens(const FlatMultiVectors& mv) {
    return static_cast<int64_t>(mv.flat.size()) / kD;
}

const float*
DocPtr(const FlatMultiVectors& mv, int64_t idx) {
    return mv.flat.data() + mv.offsets[idx] * kD;
}

void
RebuildOffsets(FlatMultiVectors& mv) {
    mv.offsets.assign(mv.counts.size() + 1, 0);
    for (size_t i = 0; i < mv.counts.size(); ++i) {
        mv.offsets[i + 1] = mv.offsets[i] + mv.counts[i];
    }
}

// dense 单向量 = 该文档所有 token 的 mean-pooling。
std::vector<float>
MeanPool(const FlatMultiVectors& mv) {
    int64_t count = static_cast<int64_t>(mv.counts.size());
    std::vector<float> dense(static_cast<size_t>(count) * kD, 0.0F);
    for (int64_t i = 0; i < count; ++i) {
        const float* doc = DocPtr(mv, i);
        uint32_t len = mv.counts[i];
        float* out = dense.data() + i * kD;
        for (uint32_t t = 0; t < len; ++t) {
            const float* tok = doc + static_cast<int64_t>(t) * kD;
            for (int64_t c = 0; c < kD; ++c) {
                out[c] += tok[c];
            }
        }
        if (len > 0) {
            for (int64_t c = 0; c < kD; ++c) {
                out[c] /= static_cast<float>(len);
            }
        }
    }
    return dense;
}

// 生成随机 token 向量集合,token 数在 [min_len, max_len] 之间。
FlatMultiVectors
GenRandomMultiVectors(int64_t count,
                      std::mt19937& gen,
                      uint32_t min_len,
                      uint32_t max_len) {
    FlatMultiVectors mv;
    mv.counts.resize(count);
    std::uniform_int_distribution<uint32_t> len_dist(min_len, max_len);
    std::uniform_real_distribution<float> val_dist(-1.0F, 1.0F);
    for (int64_t i = 0; i < count; ++i) {
        mv.counts[i] = len_dist(gen);
    }
    RebuildOffsets(mv);
    int64_t total = mv.offsets.back();
    mv.flat.resize(static_cast<size_t>(total) * kD);
    for (auto& v : mv.flat) {
        v = val_dist(gen);
    }
    return mv;
}

// 把 [start, start+len) 个 token 写到 mv 的第 doc_idx 个文档(覆盖式)。
// 仅用于"误排样本"的精细构造。
void
SetDocTokens(FlatMultiVectors& mv,
             int64_t doc_idx,
             const std::vector<std::vector<float>>& tokens) {
    // 重新分配整张平铺表:先改 counts 再 rebuild。
    mv.counts[doc_idx] = static_cast<uint32_t>(tokens.size());
    // 先把旧数据按文档收集出来,再整体重排,保证 offsets 连续。
    std::vector<std::vector<std::vector<float>>> docs(mv.counts.size());
    for (size_t i = 0; i < mv.counts.size(); ++i) {
        if (static_cast<int64_t>(i) == doc_idx) {
            docs[i] = tokens;
            continue;
        }
        const float* doc = DocPtr(mv, static_cast<int64_t>(i));
        // 注意:此时 counts[doc_idx] 已改,offsets 仍是旧的,DocPtr 用旧 offsets。
        // 为避免越界,先在调用前 RebuildOffsets,这里读的是旧布局。
        uint32_t len = mv.counts[i];
        docs[i].resize(len);
        for (uint32_t t = 0; t < len; ++t) {
            docs[i][t].assign(doc + static_cast<int64_t>(t) * kD,
                              doc + static_cast<int64_t>(t + 1) * kD);
        }
    }
    // 重建平铺表。
    RebuildOffsets(mv);
    int64_t total = mv.offsets.back();
    mv.flat.assign(static_cast<size_t>(total) * kD, 0.0F);
    for (size_t i = 0; i < docs.size(); ++i) {
        float* out = mv.flat.data() + mv.offsets[i] * kD;
        for (size_t t = 0; t < docs[i].size(); ++t) {
            std::copy(docs[i][t].begin(), docs[i][t].end(), out + t * kD);
        }
    }
}

// 返回按 score 升序排列的下标(index),而非排序后的值。同 numpy.argsort。
std::vector<int64_t>
ArgsortAsc(const std::vector<float>& scores) {
    std::vector<int64_t> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
        return scores[a] < scores[b];
    });
    return idx;
}

void
WriteFloatMatrix(H5::H5File& file,
                 const std::string& path,
                 const float* data,
                 hsize_t rows,
                 hsize_t cols) {
    hsize_t dims[2] = {rows, cols};
    H5::DataSpace space(2, dims);
    H5::DataSet ds = file.createDataSet(path, H5::PredType::NATIVE_FLOAT, space);
    ds.write(data, H5::PredType::NATIVE_FLOAT);
}

void
WriteUint32Vector(H5::H5File& file,
                  const std::string& path,
                  const uint32_t* data,
                  hsize_t len) {
    hsize_t dims[1] = {len};
    H5::DataSpace space(1, dims);
    H5::DataSet ds = file.createDataSet(path, H5::PredType::NATIVE_UINT32, space);
    ds.write(data, H5::PredType::NATIVE_UINT32);
}

void
WriteInt64Matrix(H5::H5File& file,
                 const std::string& path,
                 const int64_t* data,
                 hsize_t rows,
                 hsize_t cols) {
    hsize_t dims[2] = {rows, cols};
    H5::DataSpace space(2, dims);
    H5::DataSet ds = file.createDataSet(path, H5::PredType::NATIVE_INT64, space);
    ds.write(data, H5::PredType::NATIVE_INT64);
}

void
WriteStringAttr(H5::H5File& file, const std::string& name, const std::string& value) {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = file.createAttribute(name, str_type, scalar);
    attr.write(str_type, value);
}

void
WriteInt64Attr(H5::H5File& file, const std::string& name, int64_t value) {
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = file.createAttribute(name, H5::PredType::NATIVE_INT64, scalar);
    attr.write(H5::PredType::NATIVE_INT64, &value);
}

}  // namespace

int
main(int argc, char** argv) {
    std::string out_dir = kDefaultOutDir;
    if (argc > 1) {
        out_dir = argv[1];
    }
    std::string hdf5_path = out_dir + "/example.hdf5";
    std::string oracle_path = out_dir + "/oracle.json";

    std::mt19937 gen(20260610);

    // 1) 生成 base / query 的多向量。
    FlatMultiVectors train_mv = GenRandomMultiVectors(kN, gen, 2, 4);
    FlatMultiVectors test_mv = GenRandomMultiVectors(kQ, gen, kQueryTokens, kQueryTokens);

    // 2) 构造一个"误排样本":让 query 0 的 dense mean top1 偏向 doc A,
    //    但 MaxSim top1 偏向 doc B。
    //    手法:query 0 用两个互相对立的 token(t_a, t_b);
    //    doc A 的两个 token 都"中庸",mean 与 query mean 很近(粗排赢);
    //    doc B 有一个 token 与 t_b 几乎一致(MaxSim 在 t_b 上拿到极低距离),
    //    但另一个 token 远离,使 mean 被拉偏(粗排输)。
    const int64_t doc_a = 1;
    const int64_t doc_b = 2;
    {
        // query 0 的两个 token。
        std::vector<float> q_ta(kD, 0.0F);
        std::vector<float> q_tb(kD, 0.0F);
        q_ta[0] = 1.0F;  // 指向 +e0
        q_tb[1] = 1.0F;  // 指向 +e1
        std::vector<std::vector<float>> q0_tokens = {q_ta, q_tb};

        // doc A: 两个 token 都接近 query 的 mean 方向 (e0+e1)/sqrt2,
        //        mean 与 query mean 很接近 -> dense 粗排 doc A 靠前。
        std::vector<float> a_t(kD, 0.0F);
        a_t[0] = 0.707F;
        a_t[1] = 0.707F;
        std::vector<std::vector<float>> a_tokens = {a_t, a_t};

        // doc B: token0 与 q_tb 几乎一致(MaxSim 在 q_tb 上 ~0 距离);
        //        token1 指向 -e0 -e1,把 mean 拉偏,dense 粗排吃亏。
        std::vector<float> b_t0(kD, 0.0F);
        b_t0[1] = 1.0F;  // == q_tb
        std::vector<float> b_t1(kD, 0.0F);
        b_t1[0] = -1.0F;
        b_t1[1] = -1.0F;
        std::vector<std::vector<float>> b_tokens = {b_t0, b_t1};

        // 逐个覆盖(SetDocTokens 内部会整表 rebuild,需顺序调用)。
        SetDocTokens(test_mv, 0, q0_tokens);
        SetDocTokens(train_mv, doc_a, a_tokens);
        SetDocTokens(train_mv, doc_b, b_tokens);
    }

    // 3) dense = mean-pooling。
    std::vector<float> train_dense = MeanPool(train_mv);
    std::vector<float> test_dense = MeanPool(test_mv);

    // 4) MaxSim ground truth: 每个 query 对全量 base 算 MaxSim,取升序前 K。
    std::vector<int64_t> neighbors(static_cast<size_t>(kQ) * kTopK);
    std::vector<float> distances(static_cast<size_t>(kQ) * kTopK);
    // 同时算 dense-only top1(用于 oracle 对照与误排自检)。
    std::vector<int64_t> dense_top1(kQ);
    std::vector<int64_t> maxsim_top1(kQ);

    auto dense_dist = [&](const float* q, const float* d) { return PerTokenDistIp(q, d); };

    for (int64_t qi = 0; qi < kQ; ++qi) {
        const float* q_tokens = DocPtr(test_mv, qi);
        uint32_t q_len = test_mv.counts[qi];
        std::vector<float> maxsim_scores(kN);
        std::vector<float> dense_scores(kN);
        for (int64_t bi = 0; bi < kN; ++bi) {
            maxsim_scores[bi] = MaxSimDist(
                q_tokens, q_len, DocPtr(train_mv, bi), train_mv.counts[bi]);
            dense_scores[bi] =
                dense_dist(test_dense.data() + qi * kD, train_dense.data() + bi * kD);
        }
        auto maxsim_order = ArgsortAsc(maxsim_scores);
        auto dense_order = ArgsortAsc(dense_scores);
        maxsim_top1[qi] = maxsim_order[0];
        dense_top1[qi] = dense_order[0];
        for (int64_t k = 0; k < kTopK; ++k) {
            neighbors[qi * kTopK + k] = maxsim_order[k];
            distances[qi * kTopK + k] = maxsim_scores[maxsim_order[k]];
        }
    }

    // 5) 自检 asserts。
    assert(TotalTokens(train_mv) == static_cast<int64_t>(train_mv.offsets.back()));
    {
        int64_t sum_train = std::accumulate(
            train_mv.counts.begin(), train_mv.counts.end(), static_cast<int64_t>(0));
        int64_t sum_test = std::accumulate(
            test_mv.counts.begin(), test_mv.counts.end(), static_cast<int64_t>(0));
        assert(sum_train == TotalTokens(train_mv) &&
               "sum(train_vector_counts) != train_multi_vectors rows");
        assert(sum_test == TotalTokens(test_mv) &&
               "sum(test_vector_counts) != test_multi_vectors rows");
        // Release 下 assert 展开为空,suppress unused variable 警告。
        (void)sum_train;
        (void)sum_test;
    }

    // 统计有多少 query 的 dense 粗排 top1 与 MaxSim 精排 top1 不一致。
    // 固定随机种子下随机数据天然会产生误排,此 assert 依赖种子确定性而非手工构造。
    int mismatch_count = 0;
    std::vector<int64_t> mismatch_queries;
    for (int64_t qi = 0; qi < kQ; ++qi) {
        if (dense_top1[qi] != maxsim_top1[qi]) {
            ++mismatch_count;
            mismatch_queries.push_back(qi);
        }
    }
    assert(mismatch_count >= 1 &&
           "no mis-ranking sample: dense top1 == MaxSim top1 for all queries");

    // 6) 写 HDF5。
    try {
        H5::H5File file(hdf5_path, H5F_ACC_TRUNC);

        WriteStringAttr(file, "type", "dense");
        WriteStringAttr(file, "reorder_type", "multi_vector");
        WriteStringAttr(file, "distance", "ip");
        WriteInt64Attr(file, "multi_vector_dim", kD);

        WriteFloatMatrix(file, "/train", train_dense.data(), kN, kD);
        WriteFloatMatrix(file, "/test", test_dense.data(), kQ, kD);
        WriteFloatMatrix(file,
                         "/train_multi_vectors",
                         train_mv.flat.data(),
                         static_cast<hsize_t>(TotalTokens(train_mv)),
                         kD);
        WriteFloatMatrix(file,
                         "/test_multi_vectors",
                         test_mv.flat.data(),
                         static_cast<hsize_t>(TotalTokens(test_mv)),
                         kD);
        WriteUint32Vector(file, "/train_vector_counts", train_mv.counts.data(), kN);
        WriteUint32Vector(file, "/test_vector_counts", test_mv.counts.data(), kQ);
        WriteInt64Matrix(file, "/neighbors", neighbors.data(), kQ, kTopK);
        WriteFloatMatrix(file, "/distances", distances.data(), kQ, kTopK);
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 write failed: " << e.getDetailMsg() << std::endl;
        return 1;
    }

    // 7) 写 oracle.json(每个 query 的 dense-only topK 与 MaxSim topK)。
    {
        std::ofstream ofs(oracle_path);
        ofs << "{\n";
        ofs << "  \"N\": " << kN << ", \"Q\": " << kQ << ", \"D\": " << kD
            << ", \"K\": " << kTopK << ",\n";
        ofs << "  \"mismatch_queries\": [";
        for (size_t i = 0; i < mismatch_queries.size(); ++i) {
            ofs << mismatch_queries[i];
            if (i + 1 < mismatch_queries.size()) {
                ofs << ", ";
            }
        }
        ofs << "],\n";
        ofs << "  \"queries\": [\n";
        for (int64_t qi = 0; qi < kQ; ++qi) {
            const float* q_tokens = DocPtr(test_mv, qi);
            uint32_t q_len = test_mv.counts[qi];
            std::vector<float> dense_scores(kN);
            for (int64_t bi = 0; bi < kN; ++bi) {
                dense_scores[bi] =
                    dense_dist(test_dense.data() + qi * kD, train_dense.data() + bi * kD);
            }
            (void)q_tokens;
            (void)q_len;
            auto dense_order = ArgsortAsc(dense_scores);
            ofs << "    {\"query\": " << qi << ", \"dense_top1\": " << dense_top1[qi]
                << ", \"maxsim_top1\": " << maxsim_top1[qi] << ", \"dense_topk\": [";
            for (int64_t k = 0; k < kTopK; ++k) {
                ofs << dense_order[k];
                if (k + 1 < kTopK) {
                    ofs << ", ";
                }
            }
            ofs << "], \"maxsim_topk\": [";
            for (int64_t k = 0; k < kTopK; ++k) {
                ofs << neighbors[qi * kTopK + k];
                if (k + 1 < kTopK) {
                    ofs << ", ";
                }
            }
            ofs << "]}";
            if (qi + 1 < kQ) {
                ofs << ",";
            }
            ofs << "\n";
        }
        ofs << "  ]\n}\n";
    }

    std::cout << "OK: wrote " << hdf5_path << " and " << oracle_path << "\n";
    std::cout << "self-check passed. mis-ranking queries (dense top1 != MaxSim top1): ";
    for (size_t i = 0; i < mismatch_queries.size(); ++i) {
        std::cout << mismatch_queries[i] << (i + 1 < mismatch_queries.size() ? " " : "");
    }
    std::cout << "\n";
    for (int64_t qi : mismatch_queries) {
        std::cout << "  query " << qi << ": dense top1=" << dense_top1[qi]
                  << " (粗排), MaxSim top1=" << maxsim_top1[qi] << " (精排纠正)\n";
    }
    return 0;
}
