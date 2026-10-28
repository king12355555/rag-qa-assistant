"""命令行问答工具。

用法：
    python -m src.cli "定金不得超过主合同标的额的多少比例？"
    python -m src.cli --data test_samples --rebuild   # 先重建索引再进入交互问答
    python -m src.cli                                 # 交互模式
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build_index import build_index, index_stats
from src.config import DEFAULT_DATA_DIR, K, NUM_MULTI_QUERIES
from src.qa_chain import answer_with_sources, create_qa_system


def print_answer(payload: dict) -> None:
    print("\n" + "=" * 70)
    print(f"问题：{payload['question']}")
    print("-" * 70)
    print(f"回答：{payload['answer']}")
    if payload.get("queries"):
        print(f"\n[MultiQuery] 改写后的检索问题：{payload['queries']}")
    if payload.get("sources"):
        print("\n[来源引用]")
        for i, s in enumerate(payload["sources"], 1):
            print(f"  {i}. {s['source']} ｜ {s['location']}")
            print(f"     {s['excerpt']}…")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 命令行问答")
    parser.add_argument("question", nargs="?", help="要提问的问题；不传则进入交互模式")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="文档目录")
    parser.add_argument("--rebuild", action="store_true", help="先重建索引")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--num-queries", type=int, default=NUM_MULTI_QUERIES)
    args = parser.parse_args()

    if args.rebuild or index_stats() == 0:
        stats = build_index(args.data, rebuild=args.rebuild)
        print(f"[cli] 索引状态：{stats}")

    system = create_qa_system(k=args.k, num_queries=args.num_queries)
    chain, retriever = system["chain"], system["retriever"]

    if args.question:
        try:
            print_answer(answer_with_sources(args.question, chain, retriever))
        except Exception as exc:
            print(f"[ERROR] 问答失败：{exc}")
            print("请确认：① LLM 端点已启动且 .env 中 LLM_BASE_URL 正确；② 已构建索引。")
        return

    print("进入交互模式（输入 q / exit 退出）")
    while True:
        try:
            question = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("q", "quit", "exit", "退出"):
            break
        if not question:
            continue
        try:
            print_answer(answer_with_sources(question, chain, retriever))
        except Exception as exc:
            print(f"[ERROR] 问答失败：{exc}")


if __name__ == "__main__":
    main()
