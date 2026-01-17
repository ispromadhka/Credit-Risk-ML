FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (except turbocat)
RUN pip install --no-cache-dir \
    pandas>=1.5.0 \
    numpy>=1.23.0 \
    scikit-learn>=1.2.0 \
    matplotlib>=3.6.0 \
    seaborn>=0.12.0 \
    catboost>=1.2.0 \
    jupyter>=1.0.0 \
    ipykernel>=6.0.0

# Clone and build TurboCat with missing stub files
RUN git clone https://github.com/ispromadhka/Turbo-Cat.git /tmp/turbocat && \
    mkdir -p /tmp/turbocat/src/metrics /tmp/turbocat/src/preprocessing

# Create missing metrics.cpp stub
RUN echo '#include "turbocat/metrics.hpp"\n\
#include <algorithm>\n\
#include <numeric>\n\
#include <vector>\n\
\n\
namespace turbocat {\n\
\n\
Float Metrics::roc_auc(const Float* y_true, const Float* y_pred, Index n_samples) {\n\
    std::vector<Index> indices(n_samples);\n\
    std::iota(indices.begin(), indices.end(), 0);\n\
    std::sort(indices.begin(), indices.end(), [&](Index a, Index b) {\n\
        return y_pred[a] > y_pred[b];\n\
    });\n\
    Index n_pos = 0, n_neg = 0;\n\
    for (Index i = 0; i < n_samples; ++i) {\n\
        if (y_true[i] > 0.5f) n_pos++; else n_neg++;\n\
    }\n\
    if (n_pos == 0 || n_neg == 0) return 0.5f;\n\
    Float auc = 0.0f, tp = 0.0f;\n\
    for (Index i = 0; i < n_samples; ++i) {\n\
        if (y_true[indices[i]] > 0.5f) tp += 1.0f; else auc += tp;\n\
    }\n\
    return auc / (n_pos * n_neg);\n\
}\n\
\n\
Float Metrics::pr_auc(const Float* y_true, const Float* y_pred, Index n_samples) {\n\
    std::vector<Index> indices(n_samples);\n\
    std::iota(indices.begin(), indices.end(), 0);\n\
    std::sort(indices.begin(), indices.end(), [&](Index a, Index b) {\n\
        return y_pred[a] > y_pred[b];\n\
    });\n\
    Index n_pos = 0;\n\
    for (Index i = 0; i < n_samples; ++i) if (y_true[i] > 0.5f) n_pos++;\n\
    if (n_pos == 0) return 0.0f;\n\
    Float auc = 0.0f, tp = 0.0f, prev_recall = 0.0f;\n\
    for (Index i = 0; i < n_samples; ++i) {\n\
        if (y_true[indices[i]] > 0.5f) tp += 1.0f;\n\
        Float recall = tp / n_pos;\n\
        auc += (tp / (i + 1)) * (recall - prev_recall);\n\
        prev_recall = recall;\n\
    }\n\
    return auc;\n\
}\n\
\n\
Float Metrics::find_optimal_threshold(const Float* y_true, const Float* y_pred, Index n_samples, MetricType metric, int n_thresholds) {\n\
    Float best_threshold = 0.5f, best_score = higher_is_better(metric) ? -1e10f : 1e10f;\n\
    for (int i = 0; i <= n_thresholds; ++i) {\n\
        Float threshold = static_cast<Float>(i) / n_thresholds;\n\
        Float score = f1_score(y_true, y_pred, n_samples, threshold);\n\
        bool is_better = higher_is_better(metric) ? (score > best_score) : (score < best_score);\n\
        if (is_better) { best_score = score; best_threshold = threshold; }\n\
    }\n\
    return best_threshold;\n\
}\n\
\n\
std::pair<Float, Float> Metrics::compute_at_optimal_threshold(const std::vector<Float>& y_true, const std::vector<Float>& y_pred, MetricType metric, int n_thresholds) {\n\
    Float threshold = find_optimal_threshold(y_true.data(), y_pred.data(), y_true.size(), metric, n_thresholds);\n\
    return {threshold, compute(metric, y_true, y_pred, threshold)};\n\
}\n\
\n\
Float Metrics::r2_score(const std::vector<Float>& y_true, const std::vector<Float>& y_pred) {\n\
    if (y_true.empty()) return 0.0f;\n\
    Float mean_y = 0.0f;\n\
    for (auto v : y_true) mean_y += v;\n\
    mean_y /= y_true.size();\n\
    Float ss_tot = 0.0f, ss_res = 0.0f;\n\
    for (size_t i = 0; i < y_true.size(); ++i) {\n\
        ss_tot += (y_true[i] - mean_y) * (y_true[i] - mean_y);\n\
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);\n\
    }\n\
    return ss_tot == 0.0f ? 0.0f : 1.0f - (ss_res / ss_tot);\n\
}\n\
\n\
Float Metrics::compute(MetricType metric, const std::vector<Float>& y_true, const std::vector<Float>& y_pred, Float threshold) {\n\
    switch (metric) {\n\
        case MetricType::LogLoss: return log_loss(y_true, y_pred);\n\
        case MetricType::Accuracy: return accuracy(y_true, y_pred, threshold);\n\
        case MetricType::Precision: return precision(y_true, y_pred, threshold);\n\
        case MetricType::Recall: return recall(y_true, y_pred, threshold);\n\
        case MetricType::F1: return f1_score(y_true, y_pred, threshold);\n\
        case MetricType::ROC_AUC: return roc_auc(y_true, y_pred);\n\
        case MetricType::PR_AUC: return pr_auc(y_true, y_pred);\n\
        case MetricType::MSE: return mse(y_true, y_pred);\n\
        case MetricType::MAE: return mae(y_true, y_pred);\n\
        case MetricType::RMSE: return rmse(y_true, y_pred);\n\
        default: return 0.0f;\n\
    }\n\
}\n\
\n\
}' > /tmp/turbocat/src/metrics/metrics.cpp

# Create missing interactions.cpp stub
RUN echo '#include "turbocat/interactions.hpp"\n\
#include "turbocat/dataset.hpp"\n\
\n\
namespace turbocat {\n\
\n\
std::vector<FeatureInteraction> InteractionDetector::detect(const Dataset& data, const InteractionConfig& config) {\n\
    switch (config.method) {\n\
        case InteractionConfig::DetectionMethod::SplitBased: return detect_split_based(data, config);\n\
        case InteractionConfig::DetectionMethod::MutualInfo: return detect_mutual_info(data, config);\n\
        case InteractionConfig::DetectionMethod::Correlation: return detect_correlation(data, config);\n\
        default: return detect_split_based(data, config);\n\
    }\n\
}\n\
\n\
std::vector<FeatureInteraction> InteractionDetector::detect_split_based(const Dataset&, const InteractionConfig&) { return {}; }\n\
std::vector<FeatureInteraction> InteractionDetector::detect_mutual_info(const Dataset&, const InteractionConfig&) { return {}; }\n\
std::vector<FeatureInteraction> InteractionDetector::detect_correlation(const Dataset&, const InteractionConfig&) { return {}; }\n\
Float InteractionDetector::compute_feature_gain(const Dataset&, FeatureIndex) { return 0.0f; }\n\
Float InteractionDetector::compute_conditional_gain(const Dataset&, FeatureIndex, FeatureIndex) { return 0.0f; }\n\
Float InteractionDetector::compute_mutual_info(const Dataset&, FeatureIndex) { return 0.0f; }\n\
Float InteractionDetector::compute_joint_mutual_info(const Dataset&, FeatureIndex, FeatureIndex) { return 0.0f; }\n\
FeatureIndex InteractionGenerator::generate(Dataset&, const std::vector<FeatureInteraction>&, const InteractionConfig&) { return 0; }\n\
void InteractionGenerator::apply(Dataset&, const Dataset&) {}\n\
void InteractionGenerator::create_product_feature(Dataset&, FeatureIndex, FeatureIndex) {}\n\
void InteractionGenerator::create_ratio_feature(Dataset&, FeatureIndex, FeatureIndex) {}\n\
void InteractionGenerator::create_concat_feature(Dataset&, FeatureIndex, FeatureIndex) {}\n\
\n\
}' > /tmp/turbocat/src/preprocessing/interactions.cpp

# Build and install TurboCat
RUN cd /tmp/turbocat && pip install --no-cache-dir . && rm -rf /tmp/turbocat

# Copy project files
COPY data/ /app/data/
COPY notebooks/ /app/notebooks/
COPY reports/ /app/reports/

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
