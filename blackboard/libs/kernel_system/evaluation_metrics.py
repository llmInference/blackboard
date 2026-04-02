"""
实验评估指标计算

包含：
1. 上下文窗口利用率 (Context Window Utilization)
2. 幻觉率 (Hallucination Rate)
3. 信息检索精度 (Information Retrieval Precision)
"""
from typing import List, Dict, Any, Set
import re


class EvaluationMetrics:
    """评估指标计算器"""

    def __init__(self, max_context_window: int = 8192):
        """
        Args:
            max_context_window: 最大上下文窗口大小（token 数）
        """
        self.max_context_window = max_context_window

    # ========== 1. 上下文窗口利用率 ==========

    def calculate_context_utilization(
        self,
        actual_tokens_used: int,
        max_window: int | None = None
    ) -> float:
        """
        计算上下文窗口利用率

        Args:
            actual_tokens_used: 实际使用的 token 数
            max_window: 最大窗口大小（可选，默认使用初始化时的值）

        Returns:
            利用率百分比 (0-100)

        示例:
            >>> metrics = EvaluationMetrics(max_context_window=8192)
            >>> metrics.calculate_context_utilization(4096)
            50.0
        """
        max_window = max_window or self.max_context_window
        if max_window <= 0:
            return 0.0

        utilization = (actual_tokens_used / max_window) * 100
        return min(utilization, 100.0)  # 最大不超过 100%

    def calculate_average_context_utilization(
        self,
        token_usage_list: List[int]
    ) -> Dict[str, float]:
        """
        计算多次执行的平均上下文窗口利用率

        Args:
            token_usage_list: 每次执行使用的 token 数列表

        Returns:
            包含平均值、最大值、最小值的字典

        示例:
            >>> metrics = EvaluationMetrics(max_context_window=8192)
            >>> metrics.calculate_average_context_utilization([4096, 6144, 2048])
            {'average': 50.0, 'max': 75.0, 'min': 25.0, 'std': 25.0}
        """
        if not token_usage_list:
            return {"average": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}

        utilizations = [
            self.calculate_context_utilization(tokens)
            for tokens in token_usage_list
        ]

        import statistics
        return {
            "average": statistics.mean(utilizations),
            "max": max(utilizations),
            "min": min(utilizations),
            "std": statistics.stdev(utilizations) if len(utilizations) > 1 else 0.0,
        }

    # ========== 2. 幻觉率 ==========

    def calculate_hallucination_rate(
        self,
        generated_facts: List[str],
        ground_truth_facts: List[str],
        similarity_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        计算幻觉率（生成内容中不符合事实的比例）

        Args:
            generated_facts: 生成的事实陈述列表
            ground_truth_facts: 真实的事实陈述列表（参考答案）
            similarity_threshold: 相似度阈值（0-1）

        Returns:
            包含幻觉率、幻觉数量、总数量的字典

        计算方法:
            1. 对于每个生成的事实，检查是否在真实事实中存在
            2. 使用文本相似度匹配（简化版使用关键词匹配）
            3. 幻觉率 = 不匹配的事实数 / 总生成事实数

        示例:
            >>> metrics = EvaluationMetrics()
            >>> generated = ["北京是中国的首都", "上海有1000万人口"]
            >>> ground_truth = ["北京是中国的首都", "上海有2400万人口"]
            >>> metrics.calculate_hallucination_rate(generated, ground_truth)
            {'hallucination_rate': 50.0, 'hallucinated_count': 1, 'total_count': 2}
        """
        if not generated_facts:
            return {
                "hallucination_rate": 0.0,
                "hallucinated_count": 0,
                "total_count": 0,
                "hallucinated_facts": []
            }

        hallucinated_facts = []
        hallucinated_count = 0

        for gen_fact in generated_facts:
            # 检查生成的事实是否在真实事实中有匹配
            is_hallucination = True

            for truth_fact in ground_truth_facts:
                # 简化版：使用关键词匹配
                # 实际应用中可以使用更复杂的语义相似度模型
                similarity = self._calculate_text_similarity(gen_fact, truth_fact)

                if similarity >= similarity_threshold:
                    is_hallucination = False
                    break

            if is_hallucination:
                hallucinated_count += 1
                hallucinated_facts.append(gen_fact)

        hallucination_rate = (hallucinated_count / len(generated_facts)) * 100

        return {
            "hallucination_rate": hallucination_rate,
            "hallucinated_count": hallucinated_count,
            "total_count": len(generated_facts),
            "hallucinated_facts": hallucinated_facts
        }

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（简化版）

        使用 Jaccard 相似度：交集 / 并集

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度 (0-1)
        """
        # 分词（简化版：按字符分割）
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        # Jaccard 相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 移除标点符号，按空格和中文字符分割
        text = re.sub(r'[^\w\s]', '', text)
        # 中文按字符分割，英文按单词分割
        tokens = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                tokens.append(char)
            elif char.isalnum():
                tokens.append(char.lower())
        return tokens

    # ========== 3. 信息检索精度 ==========

    def calculate_retrieval_precision(
        self,
        retrieved_items: List[str],
        relevant_items: List[str]
    ) -> Dict[str, Any]:
        """
        计算信息检索精度（Precision）

        Args:
            retrieved_items: 检索到的项目列表
            relevant_items: 相关的项目列表（真实答案）

        Returns:
            包含精度、召回率、F1 分数的字典

        计算公式:
            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            F1 = 2 * (Precision * Recall) / (Precision + Recall)

            其中:
            - TP (True Positive): 检索到且相关的项目数
            - FP (False Positive): 检索到但不相关的项目数
            - FN (False Negative): 未检索到但相关的项目数

        示例:
            >>> metrics = EvaluationMetrics()
            >>> retrieved = ["doc1", "doc2", "doc3"]
            >>> relevant = ["doc1", "doc2", "doc4"]
            >>> metrics.calculate_retrieval_precision(retrieved, relevant)
            {'precision': 66.67, 'recall': 66.67, 'f1': 66.67, 'tp': 2, 'fp': 1, 'fn': 1}
        """
        retrieved_set = set(retrieved_items)
        relevant_set = set(relevant_items)

        # 计算 TP, FP, FN
        tp = len(retrieved_set & relevant_set)  # 交集
        fp = len(retrieved_set - relevant_set)  # 检索到但不相关
        fn = len(relevant_set - retrieved_set)  # 相关但未检索到

        # 计算精度
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0

        # 计算召回率
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0

        # 计算 F1 分数
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "retrieved_count": len(retrieved_items),
            "relevant_count": len(relevant_items)
        }

    def calculate_retrieval_metrics_at_k(
        self,
        retrieved_items: List[str],
        relevant_items: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[int, Dict[str, Any]]:
        """
        计算不同 K 值下的检索指标（Precision@K, Recall@K）

        Args:
            retrieved_items: 检索到的项目列表（按相关性排序）
            relevant_items: 相关的项目列表
            k_values: K 值列表

        Returns:
            每个 K 值对应的指标字典

        示例:
            >>> metrics = EvaluationMetrics()
            >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            >>> relevant = ["doc1", "doc3", "doc6"]
            >>> metrics.calculate_retrieval_metrics_at_k(retrieved, relevant, [1, 3, 5])
            {1: {'precision': 100.0, 'recall': 33.33}, 3: {'precision': 66.67, 'recall': 66.67}, ...}
        """
        results = {}
        relevant_set = set(relevant_items)

        for k in k_values:
            # 取前 K 个检索结果
            top_k = retrieved_items[:k]
            top_k_set = set(top_k)

            # 计算 Precision@K 和 Recall@K
            tp_at_k = len(top_k_set & relevant_set)

            precision_at_k = (tp_at_k / k * 100) if k > 0 else 0.0
            recall_at_k = (tp_at_k / len(relevant_set) * 100) if len(relevant_set) > 0 else 0.0

            results[k] = {
                "precision": round(precision_at_k, 2),
                "recall": round(recall_at_k, 2),
                "tp": tp_at_k
            }

        return results

    # ========== 综合评估 ==========

    def evaluate_system(
        self,
        token_usage: int,
        generated_facts: List[str],
        ground_truth_facts: List[str],
        retrieved_items: List[str],
        relevant_items: List[str]
    ) -> Dict[str, Any]:
        """
        综合评估系统性能

        Args:
            token_usage: 使用的 token 数
            generated_facts: 生成的事实列表
            ground_truth_facts: 真实事实列表
            retrieved_items: 检索到的项目列表
            relevant_items: 相关项目列表

        Returns:
            包含所有指标的综合报告
        """
        return {
            "context_utilization": self.calculate_context_utilization(token_usage),
            "hallucination": self.calculate_hallucination_rate(
                generated_facts, ground_truth_facts
            ),
            "retrieval": self.calculate_retrieval_precision(
                retrieved_items, relevant_items
            )
        }


# ========== 使用示例 ==========

if __name__ == "__main__":
    metrics = EvaluationMetrics(max_context_window=8192)

    print("=" * 80)
    print("  评估指标计算示例")
    print("=" * 80)

    # 1. 上下文窗口利用率
    print("\n1. 上下文窗口利用率:")
    token_usage = [4096, 6144, 2048, 7000]
    util_result = metrics.calculate_average_context_utilization(token_usage)
    print(f"   平均利用率: {util_result['average']:.2f}%")
    print(f"   最大利用率: {util_result['max']:.2f}%")
    print(f"   最小利用率: {util_result['min']:.2f}%")
    print(f"   标准差: {util_result['std']:.2f}%")

    # 2. 幻觉率
    print("\n2. 幻觉率:")
    generated = [
        "北京是中国的首都",
        "上海有1000万人口",
        "长城建于明朝",
        "中国有56个民族"
    ]
    ground_truth = [
        "北京是中国的首都",
        "上海有2400万人口",
        "长城始建于春秋战国时期",
        "中国有56个民族"
    ]
    hall_result = metrics.calculate_hallucination_rate(generated, ground_truth)
    print(f"   幻觉率: {hall_result['hallucination_rate']:.2f}%")
    print(f"   幻觉数量: {hall_result['hallucinated_count']}/{hall_result['total_count']}")
    if hall_result['hallucinated_facts']:
        print(f"   幻觉内容: {hall_result['hallucinated_facts']}")

    # 3. 信息检索精度
    print("\n3. 信息检索精度:")
    retrieved = ["doc1", "doc2", "doc3", "doc5"]
    relevant = ["doc1", "doc2", "doc4", "doc6"]
    retr_result = metrics.calculate_retrieval_precision(retrieved, relevant)
    print(f"   精度 (Precision): {retr_result['precision']:.2f}%")
    print(f"   召回率 (Recall): {retr_result['recall']:.2f}%")
    print(f"   F1 分数: {retr_result['f1']:.2f}")
    print(f"   TP={retr_result['tp']}, FP={retr_result['fp']}, FN={retr_result['fn']}")

    # 4. Precision@K
    print("\n4. Precision@K 和 Recall@K:")
    retrieved_ranked = ["doc1", "doc2", "doc3", "doc5", "doc7"]
    relevant = ["doc1", "doc2", "doc4", "doc6"]
    k_results = metrics.calculate_retrieval_metrics_at_k(retrieved_ranked, relevant, [1, 3, 5])
    for k, result in k_results.items():
        print(f"   @{k}: Precision={result['precision']:.2f}%, Recall={result['recall']:.2f}%")

    print("\n" + "=" * 80)
