import logging
import json
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class AcademicMetaController:
    """
    Nhạc trưởng (Meta-Controller) điều phối vòng lặp Agentic.
    Được thiết kế chuẩn học thuật (Research-Grade) với khả năng đánh giá độ tin cậy
    định lượng và phân xử 3 cấp (3-Tier Resolution).
    """
    
    def __init__(self):
        # Trọng số siêu tham số (Hyperparameters) cho hàm Confidence Calibration
        # Sẽ được điều chỉnh thực nghiệm (empirical tuning)
        self.weights = {
            "w1_claim_rate": 0.4,       # Trọng số tỷ lệ mệnh đề được xác thực
            "w2_source_agreement": 0.3, # Trọng số sự đồng thuận của các nguồn trọng yếu
            "w3_reviewer_score": 0.2,   # Trọng số điểm đánh giá của Reviewer Agent
            "w4_retry_penalty": 0.1     # Điểm phạt khi lặp lại quá nhiều lần
        }
        
        # State tracking (Chỉ dùng cho Intra-run adaptation, chống System Drift)
        self.session_state = {
            "section_failures": {}, 
            "total_retries": 0,
            "global_context": {
                "key_entities": set(),
                "timeline": set(),
                "accepted_facts": []
            }
        }
        
    def reset_state(self):
        """
        Hard Constraint: Scope Boundary.
        Xóa sạch trí nhớ sau mỗi phiên sinh sách (Job/Batch) để ngăn chặn System Drift.
        """
        self.session_state = {
            "section_failures": {},
            "total_retries": 0,
            "global_context": {
                "key_entities": set(),
                "timeline": set(),
                "accepted_facts": []
            }
        }
        logger.info("[MetaController] State has been reset for new session.")

    def calculate_confidence(self, claim_rate: float, source_agreement: float, reviewer_score: float, retry_count: int) -> float:
        """
        Confidence Calibration Layer (Lớp hiệu chuẩn độ tự tin định lượng)
        Công thức: Confidence = w1*ClaimRate + w2*SourceAgreement + w3*ReviewerScore - w4*RetryPenalty
        """
        # Đưa các biến về dải [0, 1] nếu cần. Ở đây giả định claim_rate, source_agreement, reviewer_score đã ở dải [0, 1]
        base_score = (
            self.weights["w1_claim_rate"] * claim_rate +
            self.weights["w2_source_agreement"] * source_agreement +
            self.weights["w3_reviewer_score"] * reviewer_score
        )
        
        # Tính điểm phạt do retries (lặp nhiều chứng tỏ LLM đang loay hoay/ảo giác)
        # Chuẩn hóa penalty, giả sử max_retries = 5 -> penalty max = w4 * 1.0
        normalized_retry = min(retry_count / 5.0, 1.0)
        penalty = self.weights["w4_retry_penalty"] * normalized_retry
        
        # Giới hạn confidence trong dải [0, 1]
        confidence = max(0.0, min(1.0, base_score - penalty))
        logger.debug(f"[MetaController] Confidence Score calculated: {confidence:.3f} (Claim: {claim_rate:.2f}, Source: {source_agreement:.2f}, Rev: {reviewer_score:.2f}, Retries: {retry_count})")
        return confidence

    def evaluate_resolution(self, confidence_score: float, has_critical_contradiction: bool, is_strict_mode: bool) -> Tuple[str, str]:
        """
        3-Tier Resolution Strategy & Hard Constraints Evaluator.
        Quyết định cách hệ thống sẽ phản ứng khi vòng lặp vượt mức giới hạn.
        
        Returns:
            Tuple[Action_Level, Presentation_Format]
        """
        # -------------------------------------------------------------
        # HARD CONSTRAINT 2: Lỗi Mâu thuẫn Nghiêm trọng -> KHÔNG BYPASS
        # -------------------------------------------------------------
        if has_critical_contradiction:
            logger.warning("[MetaController] TIER 3 Triggered: Critical Contradiction detected.")
            # Presentation Layer: Thay vì "XÓA TRẮNG", dùng Academic Neutral Tone
            presentation_text = "Chi tiết về diễn biến sự kiện này vẫn đang có sự phân cực trong các nguồn tư liệu, cần tiếp tục đối chiếu thận trọng."
            return "TIER_3_CRITICAL", presentation_text
            
        # -------------------------------------------------------------
        # HARD CONSTRAINT 1 & TIER 1/2 ROUTING
        # -------------------------------------------------------------
        # Strict mode (Fact lịch sử cốt lõi) yêu cầu Confidence >= 0.85
        # Flexible mode (Phân tích ý nghĩa) yêu cầu Confidence >= 0.65
        threshold = 0.85 if is_strict_mode else 0.65
        
        if confidence_score >= threshold:
            logger.info(f"[MetaController] TIER 1 Triggered: Confidence ({confidence_score:.2f}) >= Threshold ({threshold}). Force Approving minor issue.")
            # Presentation Layer: Không hiển thị cờ, coi như văn bản chuẩn
            return "TIER_1_MINOR", "FORCE_APPROVE"
        else:
            logger.info(f"[MetaController] TIER 2 Triggered: Confidence ({confidence_score:.2f}) < Threshold ({threshold}). Low Confidence.")
            # Presentation Layer: Chuyển cờ kỹ thuật thành Footnote học thuật
            footnote = "Thông tin này đang chờ kiểm chứng chéo từ các nguồn sử liệu độc lập."
            return "TIER_2_UNCERTAIN", footnote

    def update_global_context(self, text_chunk: str, extracted_entities: list):
        """
        Structured Context Passing. Cập nhật bối cảnh toàn cục của phiên hiện tại.
        """
        for entity in extracted_entities:
            self.session_state["global_context"]["key_entities"].add(entity)
            
    def get_structured_context_json(self) -> str:
        """
        Xuất JSON context để nhét vào System Prompt của Editor/Reviewer ở các Thread khác.
        """
        ctx = self.session_state["global_context"]
        structured_data = {
            "global_context": {
                "key_entities": list(ctx["key_entities"]),
                "timeline": list(ctx["timeline"]),
                "accepted_facts": ctx["accepted_facts"]
            }
        }
        return json.dumps(structured_data, ensure_ascii=False)

# Khởi tạo Singleton Controller để quản lý chung trong 1 Job
meta_controller_instance = AcademicMetaController()
