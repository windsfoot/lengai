import os
import json
import re
import httpx
from sentence_transformers import SentenceTransformer
import chromadb

os.environ["OPENAI_API_KEY"] = "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"

from openai import OpenAI

class JargonExpander:
    def __init__(self, dictionary_path="e:/rag/jargon_dictionary.json"):
        with open(dictionary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.term_to_synonyms = {}
        for entry in data.get("entries", []):
            standard = entry["standard_term"]
            for syn in entry["synonyms"]:
                self.term_to_synonyms[syn] = standard

    def expand(self, query):
        expanded = [query]
        for syn, standard in self.term_to_synonyms.items():
            if syn in query and standard not in expanded:
                expanded.append(standard)
        return expanded

class TridentSearch:
    CONTENT_TYPE_MARKET_LOGIC = ["微观复盘", "周期理论"]
    CONTENT_TYPE_MINDSET = ["心法哲学", "自我反省", "系统思考"]
    CONTENT_TYPE_ANALOGY = ["跨域类比"]

    def __init__(self, collection_name="trading_cognition_atomic_v2"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="e:/rag/stock/chroma_db")
        self.collection = self.client.get_collection(collection_name)
        self.expander = JargonExpander()

    def _recency_score(self, date_str):
        try:
            year, month, day = date_str.split("-")
            return int(year) * 10000 + int(month) * 100 + int(day)
        except:
            return 20190101

    def _deduplicate_by_concepts(self, results):
        """当多个chunk讨论相同core_concepts时，只保留最新的（时间最晚的）。"""
        date_to_best = {}
        for r in results:
            concepts = r.get("core_concepts", [])
            date = r.get("source_date", "")
            key = tuple(sorted(set(concepts)))
            if key not in date_to_best:
                date_to_best[key] = r
            else:
                existing_date = date_to_best[key].get("source_date", "")
                if date > existing_date:
                    date_to_best[key] = r
        return list(date_to_best.values())

    def _rerank(self, results, top_k):
        deduped = self._deduplicate_by_concepts(results)

        for r in deduped:
            recency = r.get("recency_score", 20190101)
            keyword = r.get("keyword_score", 0)
            r["final_score"] = keyword * 5 + recency * 0.001

        deduped.sort(key=lambda x: x["final_score"], reverse=True)
        return deduped[:top_k]

    def _extract_keywords(self, text):
        chinese_words = re.findall(r'[\u4e00-\u9fff]+', text)
        all_chars = set()
        for word in chinese_words:
            for char in word:
                all_chars.add(char)
        return list(all_chars)

    def _keyword_match_score(self, text, keywords):
        text_lower = text.lower()
        score = 0
        for kw in keywords:
            if len(kw) == 1:
                if kw in text_lower:
                    score += 0.5
            elif kw in text_lower:
                score += 2
        return score

    def search(self, query, top_k=10):
        keywords = self._extract_keywords(query)
        expanded = self.expander.expand(query)

        all_results = {}
        for q in expanded:
            embedding = self.model.encode(q).tolist()
            results = self.collection.query(query_embeddings=[embedding], n_results=top_k * 2)

            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                chunk_id = meta.get("chunk_id", "")
                if chunk_id not in all_results:
                    combined = f"{doc} {meta.get('one_line_summary', '')}".lower()
                    all_results[chunk_id] = {
                        "chunk_id": chunk_id,
                        "source_date": meta.get("source_date", ""),
                        "source_title": meta.get("source_title", ""),
                        "content": doc,
                        "content_types": meta.get("content_types", []),
                        "market_phase": meta.get("market_phase", ""),
                        "core_concepts": meta.get("core_concepts", []),
                        "cognitive_stage": meta.get("cognitive_stage", ""),
                        "one_line_summary": meta.get("one_line_summary", ""),
                        "recency_score": self._recency_score(meta.get("source_date", "")),
                        "keyword_score": self._keyword_match_score(combined, keywords)
                    }

        results_list = list(all_results.values())
        return self._rerank(results_list, top_k)

    def search_market_logic(self, query, top_k=5):
        return self._search_by_type(query, self.CONTENT_TYPE_MARKET_LOGIC, top_k)

    def search_mindset(self, query, top_k=5):
        return self._search_by_type(query, self.CONTENT_TYPE_MINDSET, top_k)

    def search_analogy(self, query, top_k=5):
        return self._search_by_type(query, self.CONTENT_TYPE_ANALOGY, top_k)

    def _search_by_type(self, query, content_types, top_k):
        keywords = self._extract_keywords(query)
        embedding = self.model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k * 4)

        filtered = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            meta_types = meta.get("content_types", [])
            if not any(t in meta_types for t in content_types):
                continue

            combined = f"{doc} {meta.get('one_line_summary', '')}".lower()
            filtered.append({
                "chunk_id": meta.get("chunk_id", ""),
                "source_date": meta.get("source_date", ""),
                "source_title": meta.get("source_title", ""),
                "content": doc,
                "content_types": meta_types,
                "market_phase": meta.get("market_phase", ""),
                "core_concepts": meta.get("core_concepts", []),
                "cognitive_stage": meta.get("cognitive_stage", ""),
                "one_line_summary": meta.get("one_line_summary", ""),
                "recency_score": self._recency_score(meta.get("source_date", "")),
                "keyword_score": self._keyword_match_score(combined, keywords)
            })

        return self._rerank(filtered, top_k)

class TradingRAGComplete:
    SYSTEM_PROMPT = """你正在检索《冷爱》交易日记。

核心原则：认知进化论
1. 冷爱的认知是不断进化的过程。同一问题在不同时期可能有不同甚至矛盾的答案。
2. 时间是最重要的过滤器：当观点模糊或矛盾时，以后面时间的为准。
3. 检索结果和回答中，优先呈现最新的成熟认知。
4. 如果引用了早期探索观点，必须标注"早期探索（日期）"，并说明后续认知如何演进。
5. 不要将不同阶段的观点混为一谈，按时间脉络区分。

内容类型参考：
- 微观复盘、周期理论 → 市场逻辑问题
- 心法哲学、自我反省、系统思考 → 心态和认知问题
- 跨域类比 → 思维方式拓展

认知阶段（按时间排序）：
  探索期(19年初)   → 体系构建期(19年中) → 突破期(20年) → 升华期(21年)
  早期，观点正在形成    开始构建体系      认知框架升华    更高维度的认知

当同一概念出现多次，取最新的那个。" """

    def __init__(self):
        self.search = TridentSearch()
        self.llm = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://integrate.api.nvidia.com/v1",
            timeout=httpx.Timeout(120.0, connect=30.0)
        )
        self.model = "meta/llama-3.1-8b-instruct"

    def _organize_by_cognitive_stage(self, cards):
        stages = {
            "探索期(19年初)": [],
            "体系构建期(19年中)": [],
            "突破期(20年)": [],
            "升华期(21年)": []
        }

        for card in cards:
            stage = card.get("cognitive_stage", "")
            if stage in stages:
                stages[stage].append(card)

        organized = []
        for stage, stage_cards in stages.items():
            if stage_cards:
                stage_cards.sort(key=lambda x: x["recency_score"], reverse=True)
                organized.append({
                    "stage": stage,
                    "cards": stage_cards,
                    "count": len(stage_cards)
                })

        return organized

    def format_cards(self, cards):
        output = []
        for i, card in enumerate(cards, 1):
            output.append(f"--- 卡片{i} ---")
            output.append(f"日期: {card['source_date']}")
            output.append(f"阶段: {card['cognitive_stage']}")
            output.append(f"类型: {', '.join(card['content_types'])}")
            output.append(f"市场: {card['market_phase']}")
            output.append(f"摘要: {card['one_line_summary']}")
            output.append(f"内容: {card['content'][:200]}...")
            output.append("")
        return "\n".join(output)

    def generate_answer(self, question, cards, mode="auto"):
        cards_by_stage = self._organize_by_cognitive_stage(cards)

        stage_texts = []
        for org in cards_by_stage:
            stage_texts.append(f"【{org['stage']}】({org['count']}张卡片)")
            for c in org["cards"][:2]:
                stage_texts.append(f"  - {c['source_date']}: {c['one_line_summary']}")

        prompt = f"""{self.SYSTEM_PROMPT}

用户问题: {question}

检索到的思想卡片（按认知阶段组织）:
{chr(10).join(stage_texts)}

请基于以上思想卡片回答用户问题。注意：
1. 以最新认知为主
2. 引用具体日期和卡片
3. 说明观点的演变过程"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )

        return response.choices[0].message.content

    def ask(self, question, top_k=8, mode="auto"):
        if mode == "market_logic":
            cards = self.search.search_market_logic(question, top_k)
        elif mode == "mindset":
            cards = self.search.search_mindset(question, top_k)
        elif mode == "analogy":
            cards = self.search.search_analogy(question, top_k)
        else:
            cards = self.search.search(question, top_k)

        answer = self.generate_answer(question, cards, mode)

        return {
            "question": question,
            "mode": mode,
            "answer": answer,
            "cards": cards,
            "card_count": len(cards)
        }

def main():
    print("=" * 60)
    print("冷爱交易思想RAG系统 - 完整版")
    print("=" * 60)

    rag = TradingRAGComplete()
    print(f"Loaded! Collection: {rag.search.collection.count()} chunks")

    test_questions = [
        ("市场逻辑", "market_logic", "如何判断退潮期？"),
        ("心态检索", "mindset", "如何克服交易焦虑？"),
        ("类比检索", "analogy", "刘备携民渡江与股市人气有什么关系？"),
        ("通用检索", "auto", "分歧第二天的策略是什么？"),
    ]

    for name, mode, q in test_questions:
        print(f"\n{'='*60}")
        print(f"[{name}] {q}")
        print("-"*60)

        result = rag.ask(q, mode=mode, top_k=5)

        print(f"\n检索到 {result['card_count']} 张卡片")
        print(f"\n[回答]:")
        print(result["answer"][:500] + "..." if len(result["answer"]) > 500 else result["answer"])

    print(f"\n{'='*60}")
    print("测试完成!")

if __name__ == "__main__":
    main()
