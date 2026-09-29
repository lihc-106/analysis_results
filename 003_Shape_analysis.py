from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 1. 全局实验配置
# ============================================================

SUBJECT_ID = "003"

# 修改为实际数据目录
BASE_DIR = Path(r"D:\Codes\Jupyter\data\slow_gamma_data_ed\003")
RESULT_DIR = BASE_DIR / "results"

# True: 使用 003_101.mat 和 003_111.mat
# False: 使用 003_001.mat 和 003_011.mat
SELECT_DATASET = True

SINGLE_FILE = BASE_DIR / (
    f"{SUBJECT_ID}_101.mat" if SELECT_DATASET else f"{SUBJECT_ID}_001.mat"
)
MULTIPLE_FILE = BASE_DIR / (
    f"{SUBJECT_ID}_111.mat" if SELECT_DATASET else f"{SUBJECT_ID}_011.mat"
)

ALPHA = 0.05
CSV_ENCODING = "utf-8-sig"
RANDOM_STATE = 42


ScaleMethod = Literal["none", "standard", "minmax", "normalizer"]


@dataclass(frozen=True)
class PreprocessConfig:
    """数据预处理配置。"""

    clean_invalid: bool = True
    remove_outliers: bool = False
    sigma: float = 8.0
    scale_method: ScaleMethod = "minmax"


@dataclass(frozen=True)
class ComparisonSpec:
    """一次 Mann–Whitney U 检验任务。"""

    name: str
    dataset_1: str
    dataset_2: str
    label_1: int | None = None
    label_2: int | None = None
    select_by_label: bool = True


# ============================================================
# 2. 特征名称
# ============================================================

CHANNELS = list(range(1, 17))
CHANNELS_WITH_AVERAGE: list[int | str] = [*CHANNELS, "Average"]

STATISTICS = [
    "均值",
    "标准差",
    "能量",
    "RMS",
    "MTE",
    "偏度",
    "峰度",
    "HM",
    "HC",
]

ENTROPY_MEASURES = [
    "样本熵",
    "近似熵",
    "模糊熵",
    "谱熵",
]

NETWORKS = ["MI", "coh", "plv", "pli"]

NETWORK_ATTRIBUTES = [
    "平均节点度",
    "平均聚类系数",
    "全局效率",
    "节点度",
    "聚类系数",
    "局部效率",
]

LOCAL_NETWORK_ATTRIBUTES = [
    f"{attribute}{network}"
    for network in NETWORKS
    for attribute in ["节点度", "聚类系数", "局部效率"]
]

SELECTED_GLOBAL_NETWORK_FEATURES = [
    f"{network}_{attribute}"
    for network in NETWORKS
    for attribute in ["平均节点度", "平均聚类系数", "全局效率"]
]


def build_feature_names() -> list[str]:
    """ 421 个特征名称。"""

    feature_names: list[str] = []

    for statistic in STATISTICS:
        for channel in CHANNELS_WITH_AVERAGE:
            feature_names.append(f"Channel_{channel}_{statistic}")

    for entropy_measure in ENTROPY_MEASURES:
        for channel in CHANNELS:
            feature_names.append(f"Channel_{channel}_{entropy_measure}")

    for network in NETWORKS:
        for attribute in NETWORK_ATTRIBUTES:
            if attribute in {"节点度", "聚类系数", "局部效率"}:
                for channel in CHANNELS:
                    feature_names.append(
                        f"Channel_{channel}_{attribute}{network}"
                    )
            else:
                feature_names.append(f"{network}_{attribute}")

    if len(feature_names) != 421:
        raise RuntimeError(
            f"特征名称数量应为 421，当前了 {len(feature_names)} 个。"
        )

    return feature_names


FEATURE_NAMES = build_feature_names()


# ============================================================
# 3. 数据读取与预处理
# ============================================================

def clean_nan_inf(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """删除任意特征包含 NaN 或 Inf 的样本。"""

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels).reshape(-1)

    valid_mask = np.isfinite(features).all(axis=1)
    return features[valid_mask], labels[valid_mask]


def remove_sigma_outliers_sequential(
    features: np.ndarray,
    labels: np.ndarray,
    sigma: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:


    if sigma <= 0:
        raise ValueError("sigma 必须大于 0。")

    features = np.asarray(features, dtype=float).copy()
    labels = np.asarray(labels).reshape(-1).copy()

    for column_index in range(features.shape[1]):
        column = features[:, column_index]
        mean = np.mean(column)
        std = np.std(column)

        if std == 0 or not np.isfinite(std):
            continue

        lower = mean - sigma * std
        upper = mean + sigma * std
        keep_mask = (column >= lower) & (column <= upper)

        features = features[keep_mask]
        labels = labels[keep_mask]

    return features, labels


def scale_features(
    features: np.ndarray,
    method: ScaleMethod,
) -> np.ndarray:
    """对所有特征执行统一的 sklearn 变换。"""

    if method == "none":
        return features.copy()

    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "normalizer": Normalizer,
    }

    try:
        scaler = scalers[method]()
    except KeyError as exc:
        raise ValueError(f"不支持的缩放方法：{method}") from exc

    return scaler.fit_transform(features)


def load_mat_dataset(
    file_path: Path,
    preprocess: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """读取 MATLAB 文件中的 dataset.features 和 dataset.label。"""

    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{file_path}")

    mat_data = loadmat(file_path)

    try:
        dataset = mat_data["dataset"]
        features = np.asarray(dataset["features"][0, 0], dtype=float)
        labels = np.asarray(dataset["label"][0, 0]).reshape(-1)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"{file_path.name} 中未找到预期的 dataset.features 和 dataset.label。"
        ) from exc

    if features.ndim != 2:
        raise ValueError(f"features 应为二维数组，当前形状为 {features.shape}。")

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"样本数和标签数不一致：{features.shape[0]} vs {labels.shape[0]}。"
        )

    if preprocess.clean_invalid:
        features, labels = clean_nan_inf(features, labels)

    if preprocess.remove_outliers:
        features, labels = remove_sigma_outliers_sequential(
            features,
            labels,
            sigma=preprocess.sigma,
        )

    features = scale_features(features, preprocess.scale_method)

    print(
        f"[{file_path.name}] "
        f"features={features.shape}, labels={labels.shape}, "
        f"labels={dict(zip(*np.unique(labels, return_counts=True)))}"
    )

    return features, labels


def create_dataframe(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> pd.DataFrame:
    """将特征矩阵与标签组合成 DataFrame。"""

    if features.shape[1] != len(feature_names):
        raise ValueError(
            f"特征列数为 {features.shape[1]}，"
            f"但特征名称数量为 {len(feature_names)}。"
        )

    frame = pd.DataFrame(features, columns=list(feature_names))
    frame["label"] = np.asarray(labels).reshape(-1)
    return frame


# ============================================================
# 4. 显著性检验
# ============================================================

def select_feature_samples(
    data: pd.DataFrame,
    feature_name: str,
    label: int | None,
    select_by_label: bool,
) -> pd.Series:
    """从一个 DataFrame 中提取待检验的单个同名特征。"""

    if feature_name not in data.columns:
        raise KeyError(f"数据中不存在特征：{feature_name}")

    if not select_by_label:
        return data[feature_name].dropna()

    if label is None:
        raise ValueError("select_by_label=True 时必须提供标签。")

    return data.loc[data["label"] == label, feature_name].dropna()


def mann_whitney_all_features(
    data_1: pd.DataFrame,
    data_2: pd.DataFrame,
    label_1: int | None,
    label_2: int | None,
    select_by_label: bool,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    对两个数据集中的所有同名特征执行 Mann–Whitney U 检验。

    返回每个特征的 U 值、p 值、样本量、中位数及显著性标记。
    """

    feature_names_1 = [column for column in data_1.columns if column != "label"]
    feature_names_2 = {column for column in data_2.columns if column != "label"}

    missing = [name for name in feature_names_1 if name not in feature_names_2]
    if missing:
        raise ValueError(f"第二个数据集缺少 {len(missing)} 个同名特征。")

    records: list[dict[str, object]] = []

    for feature_name in feature_names_1:
        sample_1 = select_feature_samples(
            data_1,
            feature_name,
            label_1,
            select_by_label,
        )
        sample_2 = select_feature_samples(
            data_2,
            feature_name,
            label_2,
            select_by_label,
        )

        if sample_1.empty or sample_2.empty:
            raise ValueError(
                f"{feature_name} 的待比较样本为空："
                f"n1={len(sample_1)}, n2={len(sample_2)}。"
            )

        u_statistic, p_value = mannwhitneyu(
            sample_1,
            sample_2,
            alternative="two-sided",
        )

        records.append(
            {
                "feature": feature_name,
                "n_1": len(sample_1),
                "n_2": len(sample_2),
                "median_1": sample_1.median(),
                "median_2": sample_2.median(),
                "u_statistic": float(u_statistic),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
            }
        )

    result = pd.DataFrame(records)
    return result.sort_values("p_value", ignore_index=True)


def extract_channel_num(feature_name: str) -> int | None:
    """从 Channel_数字_特征名称 中提取通道号。"""

    parts = feature_name.split("_")
    if len(parts) < 3 or parts[0] != "Channel":
        return None

    try:
        return int(parts[1])
    except ValueError:
        return None


def build_significance_matrix(
    test_result: pd.DataFrame,
) -> pd.DataFrame:
    """将显著特征转换为 16 通道 × 特征类别的 0/1 矩阵。"""

    columns = STATISTICS + ENTROPY_MEASURES + LOCAL_NETWORK_ATTRIBUTES
    matrix = pd.DataFrame(0, index=CHANNELS, columns=columns, dtype=int)

    significant_features = test_result.loc[
        test_result["significant"], "feature"
    ]

    for feature_name in significant_features:
        channel = extract_channel_num(feature_name)
        feature_type = feature_name.split("_")[-1]

        if channel is not None and feature_type in matrix.columns:
            matrix.at[channel, feature_type] = 1

    return matrix.reset_index(names="channel")


def run_comparison(
    spec: ComparisonSpec,
    datasets: dict[str, pd.DataFrame],
    result_dir: Path,
) -> pd.DataFrame:
    """执行并导出一次配置化比较。"""

    try:
        data_1 = datasets[spec.dataset_1]
        data_2 = datasets[spec.dataset_2]
    except KeyError as exc:
        raise KeyError(
            f"比较任务 {spec.name} 引用了不存在的数据集：{exc}"
        ) from exc

    result = mann_whitney_all_features(
        data_1=data_1,
        data_2=data_2,
        label_1=spec.label_1,
        label_2=spec.label_2,
        select_by_label=spec.select_by_label,
    )

    detail_path = result_dir / f"{SUBJECT_ID}_{spec.name}_detail.csv"
    matrix_path = result_dir / f"{SUBJECT_ID}_{spec.name}_matrix.csv"

    result.to_csv(detail_path, index=False, encoding=CSV_ENCODING)
    build_significance_matrix(result).to_csv(
        matrix_path,
        index=False,
        encoding=CSV_ENCODING,
    )

    count = int(result["significant"].sum())
    print(
        f"[{spec.name}] 显著特征：{count}/{len(result)}；"
        f"明细：{detail_path.name}"
    )

    return result


# ============================================================
# 5. 数据导出
# ============================================================

def export_feature_data(
    data: pd.DataFrame,
    feature_names: Sequence[str],
    output_path: Path,
) -> None:
    """导出标签和指定特征。"""

    missing = [
        feature_name
        for feature_name in feature_names
        if feature_name not in data.columns
    ]
    if missing:
        raise KeyError(f"待导出的特征不存在：{missing}")

    selected_columns = ["label", *feature_names]
    data.loc[:, selected_columns].to_csv(
        output_path,
        index=False,
        encoding=CSV_ENCODING,
    )

# ============================================================
# 7. 主分析流程
# ============================================================

def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    preprocess = PreprocessConfig(
        clean_invalid=True,
        remove_outliers=False,
        sigma=8.0,
        scale_method="minmax",
    )

    single_features, single_labels = load_mat_dataset(
        SINGLE_FILE,
        preprocess,
    )
    multiple_features, multiple_labels = load_mat_dataset(
        MULTIPLE_FILE,
        preprocess,
    )

    datasets = {
        "single": create_dataframe(single_features, single_labels),
        "multiple": create_dataframe(multiple_features, multiple_labels),
    }

    # 只需修改这张配置表，即可增加、删除或调整比较任务。
    comparison_specs = [
        ComparisonSpec(
            name="O_X",
            dataset_1="single",
            dataset_2="single",
            label_1=0,
            label_2=1,
        ),
        ComparisonSpec(
            name="redO_blueX",
            dataset_1="multiple",
            dataset_2="multiple",
            label_1=0,
            label_2=1,
        ),
        ComparisonSpec(
            name="blueO_redX",
            dataset_1="multiple",
            dataset_2="multiple",
            label_1=2,
            label_2=3,
        ),
        ComparisonSpec(
            name="redO_blueO",
            dataset_1="multiple",
            dataset_2="multiple",
            label_1=0,
            label_2=2,
        ),
        ComparisonSpec(
            name="blueX_redX",
            dataset_1="multiple",
            dataset_2="multiple",
            label_1=1,
            label_2=3,
        ),
        ComparisonSpec(
            name="redO_redX",
            dataset_1="multiple",
            dataset_2="multiple",
            label_1=0,
            label_2=3,
        ),
        ComparisonSpec(
            name="blueX_blueO",
            dataset_1="multiple",
            dataset_2="multiple",
            label_1=1,
            label_2=2,
        ),
        ComparisonSpec(
            name="single_multiple",
            dataset_1="single",
            dataset_2="multiple",
            select_by_label=False,
        ),
    ]

    all_test_results: dict[str, pd.DataFrame] = {}
    for comparison_spec in comparison_specs:
        all_test_results[comparison_spec.name] = run_comparison(
            comparison_spec,
            datasets,
            RESULT_DIR,
        )

    # 导出完整特征数据
    export_feature_data(
        datasets["single"],
        FEATURE_NAMES,
        RESULT_DIR / "single_all_features.csv",
    )
    export_feature_data(
        datasets["multiple"],
        FEATURE_NAMES,
        RESULT_DIR / "multiple_all_features.csv",
    )

    # 导出重点关注特征
    export_feature_data(
        datasets["single"],
        SELECTED_GLOBAL_NETWORK_FEATURES,
        RESULT_DIR / "single_selected_network_features.csv",
    )
    export_feature_data(
        datasets["multiple"],
        SELECTED_GLOBAL_NETWORK_FEATURES,
        RESULT_DIR / "multiple_selected_network_features.csv",
    )

    print(f"\n全部结果已保存到：{RESULT_DIR}")


if __name__ == "__main__":
    main()