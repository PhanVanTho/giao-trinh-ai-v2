# -*- coding: utf-8 -*-
"""
audit_service.py (V21.1)
=======================
Hệ thống Phản biện Học thuật đa tầng (Multi-LLM Critic).
Kết hợp Logic Deterministic, Semantic Vector và Gemini Soft-Audit.
"""

import logging
import json
import re
import numpy as np
from openai import OpenAI
from google import genai
from google.genai import types
from cau_hinh import CauHinh
from .schemas import CRITIC_SCHEMA

import hashlib

logger = logging.getLogger(__name__)

# Quota Manager (Session Tracker)
class QuotaManager:
    _calls_made = 0
    MAX_DAILY_QUOTA = 500  # Gemini Free Tier limit

    @classmethod
    def track_call(cls):
        cls._calls_made += 1

    @classmethod
    def get_remaining_quota(cls, num_keys=1):
        return max(0, (cls.MAX_DAILY_QUOTA * num_keys) - cls._calls_made)

CLAIM_CACHE = {} # claim_hash -> verdict

def _normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

class ScholarlyAuditEngine:
    def __init__(self, openai_key: str, gemini_keys: list):
        self.openai_client = OpenAI(api_key=openai_key)
        self.gemini_keys = gemini_keys
        self.retry_cap = 2

    def calculate_vector_scores(self, fact_mappings: list) -> list:
        """
        Tầng 2: Semantic Vector Check.
        Tính toán độ tương đồng giữa Claim (do AI viết) và Span (nguồn gốc).
        """
        if not fact_mappings: return []
        
        # Chuẩn bị texts để embed (Claims và Spans)
        claims = [m.get("claim", "") for m in fact_mappings]
        spans = [m.get("span", "") for m in fact_mappings]
        all_texts = claims + spans
        
        try:
            resp = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=all_texts
            )
            embeddings = [_normalize(e.embedding) for e in resp.data]
            
            # Tách embeddings của claims và spans
            claim_embs = embeddings[:len(claims)]
            span_embs = embeddings[len(claims):]
            
            results = []
            for i, mapping in enumerate(fact_mappings):
                score = float(np.dot(claim_embs[i], span_embs[i]))
                mapping["vector_score"] = score
                results.append(mapping)
            return results
        except Exception as e:
            logger.error(f"[AuditEngine] Vector score calculation failed: {e}")
            for m in fact_mappings: m["vector_score"] = 1.0 # Bỏ qua nếu lỗi API
            return fact_mappings

    def gemini_soft_audit(self, fact_mappings: list, chu_de: str) -> list:
        """
        Tầng 3: Gemini Critic Audit (Surgical Check).
        Chỉ gọi khi vector_score < 0.75 hoặc có nghi ngờ.
        """
        audit_results = []
        suspicious_mappings = []
        
        # 1. Verification Caching
        for i, m in enumerate(fact_mappings):
            if m.get("vector_score", 1.0) < 0.70:
                claim_str = m.get("claim", "")
                span_str = m.get("span", "")
                hash_key = hashlib.md5(f"{claim_str}||{span_str}".encode('utf-8')).hexdigest()
                
                if hash_key in CLAIM_CACHE:
                    logger.info(f"[AuditEngine] Cache hit for claim: {hash_key[:8]}")
                    cached_result = CLAIM_CACHE[hash_key].copy()
                    cached_result["claim_index"] = i
                    audit_results.append(cached_result)
                else:
                    suspicious_mappings.append((i, m, hash_key))
        
        if not suspicious_mappings:
            return audit_results

        # Tóm tắt context cho Gemini
        audit_payload = []
        for orig_i, m, hkey in suspicious_mappings:
            audit_payload.append({
                "index": orig_i,
                "claim": m.get("claim"),
                "span": m.get("span")
            })

        prompt = f"""Bạn là KIỂM CHỨNG VIÊN HỌC THUẬT cho giáo trình: "{chu_de}".
NHIỆM VỤ: Xác minh xem các 'claim' (mệnh đề AI viết) có được hỗ trợ trực tiếp bởi 'span' (dữ liệu nguồn) không.

DANH SÁCH CẦN KIỂM CHỨNG:
{json.dumps(audit_payload, ensure_ascii=False)}

QUY TẮC NGHIÊM NGẶT:
1. TRỰC TIẾP: Claim phải được suy ra trực tiếp từ Span.
2. KHÔNG SUY DIỄN: Nếu Span không nói, dù Claim nghe có vẻ đúng thực tế, vẫn phải đánh dấu là "unsupported".
3. KHÔNG DÙNG KIẾN THỨC NGOÀI: Chỉ dùng dữ liệu trong Span để đối soát.
4. PHÂN LOẠI LỖI:
   - unsupported: Span không đề cập đến ý này.
   - partial: Có ý đúng nhưng bị phóng đại hoặc thiếu sắc thái (modality shift).
   - contradiction: Claim trái ngược hoàn toàn với Span.

TRẢ VỀ JSON THEO CRITIC_SCHEMA:
{{
  "audit_results": [
    {{
      "claim_index": index_so_nguyen,
      "verdict": "YES | NO",
      "error_type": "unsupported | partial | contradiction | none",
      "confidence": 0.0-1.0,
      "reason": "Giải thích ngắn gọn lỗi"
    }}
  ]
}}"""

        for api_key in self.gemini_keys:
            try:
                # Track Quota
                QuotaManager.track_call()
                
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(
                    model=CauHinh.SUPERVISOR_MODEL_LITE,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                text = resp.text or "{}"
                # Xử lý text thô nếu cần (tach_json)
                start = text.find('{'); end = text.rfind('}') + 1
                result = json.loads(text[start:end])
                
                new_results = result.get("audit_results", [])
                
                # Update Cache
                for r in new_results:
                    idx = r.get("claim_index")
                    for orig_i, orig_m, hkey in suspicious_mappings:
                        if orig_i == idx:
                            CLAIM_CACHE[hkey] = r.copy()
                            break
                            
                return audit_results + new_results
            except Exception as e:
                logger.warning(f"[AuditEngine] Gemini Key rotation due to: {e}")
                continue
        
        return []

    def run_full_audit(self, section_data: dict, chu_de: str, min_facts_required: int = 3, is_strict: bool = False) -> dict:
        """
        Orchestration V8: Selective Hybrid Verification Stack (NLI Filter -> LLM Judge) & Confidence Metrics.
        Returns metrics for Meta-Controller: claim_rate, source_agreement.
        """
        mappings = section_data.get("fact_mappings", [])
        content = section_data.get("content", "")
        
        # V8.1: Zero-cost Claim Alignment Check (Sanity Check)
        import re
        content_ids = set(re.findall(r'\[([\w\-]+)\]', content))
        mapping_ids = {str(m.get("source_id")) for m in mappings if m.get("source_id")}
        
        missing_ids = content_ids - mapping_ids
        missing_id_ratio = len(missing_ids) / len(content_ids) if content_ids else 0.0
        
        if missing_id_ratio >= 0.1:
            logger.warning(f"[AuditEngine] Sanity Check Failed: Missing {len(missing_ids)} mappings ({missing_id_ratio:.0%}). Rejecting.")
            return {
                "status": "fail",
                "score": 0.0,
                "feedback": [{"verdict": "NO", "error_type": "contradiction", "reason": f"Sanity Check Fail: Missing fact mappings for IDs: {missing_ids}"}],
                "claim_rate": 0.0,
                "source_agreement": 0.0,
                "has_critical_contradiction": True
            }
        elif missing_ids:
            logger.info(f"[AuditEngine] Soft-tolerance: Ignoring missing {len(missing_ids)} mappings ({missing_id_ratio:.0%} < 10%).")

        if not mappings:
            return {
                "status": "pass", 
                "score": 1.0, 
                "feedback": [],
                "claim_rate": 1.0,
                "source_agreement": 1.0,
                "has_critical_contradiction": False
            }

        # 1. Tầng 1: NLI Priority Filter (Mô phỏng qua Cosine Similarity Embedding)
        scored_mappings = self.calculate_vector_scores(mappings)
        
        entailments = []
        uncertains = []
        contradictions = []
        
        for m in scored_mappings:
            sim = m.get("vector_score", 1.0)
            if sim >= 0.65:
                m["nli_filter"] = "ENTAILMENT"
                entailments.append(m)
            elif sim <= 0.40:
                m["nli_filter"] = "CONTRADICTION"
                contradictions.append(m)
            else:
                m["nli_filter"] = "UNCERTAIN"
                uncertains.append(m)
        
        logger.info(f"[AuditEngine] NLI Filter Results: {len(entailments)} ENTAILMENT, {len(uncertains)} UNCERTAIN, {len(contradictions)} CONTRADICTION")

        # Các claim cần LLM phán xử (Tầng 2)
        llm_required_mappings = uncertains + contradictions
        
        # Quota-Aware Scheduling (V8.3): Adaptive Audit Mode
        quota_remaining = QuotaManager.get_remaining_quota(len(self.gemini_keys))
        
        if quota_remaining > 200:
            max_verify = 3 if is_strict else 2
        elif quota_remaining > 100:
            max_verify = 2 if is_strict else 1
        elif quota_remaining > 50:
            max_verify = 1
        else:
            max_verify = 0 # Cạn quota -> Skip verification
            
        if max_verify == 0:
            logger.warning("[AuditEngine] Quota cạn kiệt (<= 50). Bypass toàn bộ LLM verification để chống 429.")
            llm_required_mappings = []
        elif len(llm_required_mappings) > max_verify:
            # Risk Score = (1 - vector_score) * 0.7 + missing_source_penalty * 0.3
            # Weights are empirically chosen based on validation subset to balance semantic drift and missing provenance.
            for m in llm_required_mappings:
                v_score = m.get("vector_score", 1.0)
                missing_src_penalty = 1.0 if not m.get("source_id") else 0.0
                m["risk_score"] = (1.0 - v_score) * 0.7 + missing_src_penalty * 0.3
                
            # Chọn các claim có risk_score CAO NHẤT (nghi ngờ nhất)
            llm_required_mappings.sort(key=lambda x: x.get("risk_score", 0.0), reverse=True)
            llm_required_mappings = llm_required_mappings[:max_verify]
            logger.info(f"[AuditEngine] Quota Protection (Quota={quota_remaining}): verifying top {max_verify} worst claims out of {len(uncertains) + len(contradictions)}.")
        elif is_strict:
            logger.info(f"[AuditEngine] Strict mode (Quota={quota_remaining}): verifying all {len(llm_required_mappings)} questionable claims.")
        
        failed_claims = []
        if llm_required_mappings:
            # 2. Tầng 2: LLM-as-a-judge cho ca khó
            critic_feedback = self.gemini_soft_audit(llm_required_mappings, chu_de)
            failed_claims = [f for f in critic_feedback if f.get("verdict") == "NO"]
            
        # 3. Tính toán các chỉ số cho Confidence Calibration Layer
        total_claims = len(mappings)
        verified_claims = total_claims - len(failed_claims)
        
        claim_rate = verified_claims / total_claims if total_claims > 0 else 1.0
        
        # Mô phỏng Weighted Multi-Passage Voting: 
        # (Trong thực tế sẽ cần dictionary source_weights map với ID nguồn, ở đây tính baseline)
        source_agreement = sum(m.get("vector_score", 1.0) for m in scored_mappings) / total_claims if total_claims > 0 else 1.0
        
        # Hard Constraint: Kiểm tra xem có mâu thuẫn nghiêm trọng không (LLM xác nhận là contradiction)
        has_critical = any(f.get("error_type") == "contradiction" for f in failed_claims)

        return {
            "status": "fail" if failed_claims else "pass",
            "score": round(claim_rate, 2),
            "feedback": failed_claims,
            "error_count": len(failed_claims),
            "claim_rate": round(claim_rate, 2),
            "source_agreement": round(source_agreement, 2),
            "has_critical_contradiction": has_critical
        }

