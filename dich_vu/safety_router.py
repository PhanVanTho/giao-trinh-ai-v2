# -*- coding: utf-8 -*-
"""
Safety Router V30 — Intent × Risk Decision Matrix
Architecture: Rule-based Filter → Academic Intent Detection → AI Filter → Fail-safe

Nguyên tắc quyết định (V30):
  🔴 BLOCK   = Intent xấu rõ ràng × Risk cao    → Chặn cứng, không gợi ý
  🟡 REFRAME = Intent mơ hồ × Risk trung bình   → Chuyển hướng học thuật  
  🟢 SAFE    = Intent tốt × Risk thấp            → Cho qua

Pipeline:
  User Input → [Rule-based] → [Academic Intent] → [AI Classify] → Decision
"""
import json
import re
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# =========================================================================
# TẦNG 1: RULE-BASED FILTER (Deterministic — Không phụ thuộc AI)
# =========================================================================

# 🔴 Hard Block: Từ khóa nguy hiểm + intent hành động rõ ràng
# Đặc điểm: có ĐỘNG TỪ hành động + đối tượng nguy hiểm
HARD_BLOCK_KEYWORDS = [
    # Vũ khí / Chất nổ (intent = chế tạo/sử dụng)
    "chế tạo bom", "làm bom", "chế bom", "thuốc nổ tự chế", "make bomb", "build bomb",
    "make explosive", "build explosive", "homemade explosive",
    "dynamite recipe", "detonator build", "ied",
    # Vũ khí (intent = chế tạo)
    "chế tạo súng", "làm súng", "3d print gun", "ghost gun", "homemade weapon",
    "chế tạo vũ khí", "build weapon",
    # Ma túy (intent = sản xuất/mua bán)
    "chế ma túy", "nấu ma túy", "cook meth", "synthesize drug", "make lsd",
    "fentanyl synthesis", "heroin production", "mua ma túy",
    # Giết người / Bạo lực trực tiếp
    "cách giết người", "giết người", "how to kill", "murder method", "assassination",
    "nhập môn giết", "hướng dẫn giết",
    # Tấn công mạng / Hack có hại (intent = tấn công)
    "hack tài khoản", "hack facebook", "hack email", "hack bank",
    "ddos attack tutorial", "ransomware build", "sql injection attack guide",
    "phishing tutorial", "crack password tool",
    # Khủng bố
    "khủng bố", "terrorism", "terrorist attack", "jihad manual", "radicalization manual",
    # Lạm dụng trẻ em
    "child exploitation", "csam", "lạm dụng trẻ em",
    # Tự gây hại
    "cách tự tử", "tự tử", "suicide method", "how to die",
]

# 🟡 Soft Block: Từ khóa nhạy cảm (CÓ THỂ là học thuật → cần phân tích thêm)
SENSITIVE_KEYWORDS = [
    # Tội phạm (có thể phân tích học thuật)
    "tội phạm", "crime", "criminal", "mafia", "cartel",
    # Hack (có thể là cybersecurity)  
    "hack", "hacking", "exploit", "vulnerability",
    # Ma túy (có thể nói về tác hại, cơ chế gây nghiện)
    "ma túy", "drug", "cocaine", "heroin", "methamphetamine",
    # Tâm lý tối
    "dark psychology", "manipulation", "tâm lý đen", "thao túng tâm lý",
    # Chính trị nhạy cảm
    "propaganda", "tuyên truyền",
    # Vũ khí / Quân sự (ngữ cảnh lịch sử/khoa học có thể OK)
    "vũ khí", "weapon", "nuclear weapon", "vũ khí hạt nhân",
    "bom nguyên tử", "bom hạt nhân", "atomic bomb", "nuclear bomb", "hydrogen bomb",
    "bom", "bomb",
    # Chiến tranh (lịch sử, phân tích)
    "chiến tranh", "war",
    # Virus / Sinh học nguy hiểm
    "vũ khí sinh học", "bioweapon", "virus nhân tạo",
]

# =========================================================================
# 🧠 ACADEMIC INTENT DETECTION (Lớp mới V30)
# =========================================================================

# Các từ khóa chỉ báo ý định học thuật rõ ràng
ACADEMIC_INDICATORS = [
    # Vietnamese
    "nguyên lý", "nguyên tắc", "cơ chế", "lịch sử", "tác hại", "hệ quả",
    "phân tích", "nghiên cứu", "ảnh hưởng", "tác động", "so sánh",
    "tiến hóa", "phát triển", "lý thuyết", "mô hình", "cấu trúc",
    "hoạt động như thế nào", "hoạt động ra sao", "vì sao", "tại sao",
    "ứng dụng", "ý nghĩa", "vai trò", "đặc điểm", "phân loại",
    "tổng quan", "giới thiệu", "khái niệm", "định nghĩa",
    "xã hội học", "tâm lý học", "pháp luật", "đạo đức", "triết học",
    "y học", "khoa học", "vật lý", "hóa học", "sinh học",
    # English
    "principle", "mechanism", "history", "impact", "effect", "analysis",
    "research", "theory", "model", "structure", "how does", "how do",
    "why does", "why do", "overview", "introduction", "concept",
    "sociology", "psychology", "ethics", "philosophy", "science",
    "comparison", "evolution", "classification",
]

# Các từ khóa chỉ báo ý định hành động (nguy hiểm)
ACTION_INDICATORS = [
    # Vietnamese
    "cách làm", "cách chế", "cách tạo", "hướng dẫn làm", "hướng dẫn chế",
    "hướng dẫn hack", "hướng dẫn tấn công", "hướng dẫn chế tạo",
    "cách hack", "bước thực hiện", "làm sao để", "làm thế nào để",
    "tự làm", "tự chế", "diy", "mua ở đâu", "mua bán",
    # English
    "how to make", "how to build", "how to create", "step by step",
    "tutorial", "guide to making", "diy", "where to buy",
]

def _has_academic_intent(text: str) -> bool:
    """Kiểm tra xem câu hỏi có ý định học thuật rõ ràng không."""
    normalized = _normalize(text)
    return any(indicator in normalized for indicator in ACADEMIC_INDICATORS)

def _has_action_intent(text: str) -> bool:
    """Kiểm tra xem câu hỏi có ý định hành động (nguy hiểm) không."""
    normalized = _normalize(text)
    return any(indicator in normalized for indicator in ACTION_INDICATORS)

def _normalize(text: str) -> str:
    """Chuẩn hóa text để so khớp keyword: lowercase + bỏ dấu thừa."""
    return re.sub(r'\s+', ' ', text.lower().strip())

def rule_based_filter(topic: str) -> dict:
    """
    Tầng 1: Lọc bằng keyword — Nhanh, deterministic, không bypass bằng prompt.
    Returns: {"classification": "SAFE|REFRAME|BLOCK", "reason": "...", "layer": "rule"}
    """
    normalized = _normalize(topic)
    
    # Check Hard Block first
    for kw in HARD_BLOCK_KEYWORDS:
        if kw in normalized:
            logger.warning(f"[SafetyRouter] RULE HARD_BLOCK triggered: '{kw}' found in '{topic}'")
            return {
                "classification": "BLOCK",
                "reason": f"Nội dung chứa từ khóa bị cấm: '{kw}'",
                "layer": "rule",
                "matched_keyword": kw,
                "block_type": "hard"
            }
    
    # Check Soft Block (sensitive — cần phân tích ngữ cảnh thêm)  
    for kw in SENSITIVE_KEYWORDS:
        if kw in normalized:
            logger.info(f"[SafetyRouter] RULE SENSITIVE detected: '{kw}' in '{topic}'. Escalating to AI.")
            return {
                "classification": "REFRAME",
                "reason": f"Chủ đề nhạy cảm ('{kw}'). Đang phân tích ngữ cảnh...",
                "layer": "rule",
                "matched_keyword": kw,
                "block_type": "soft"
            }
    
    return {"classification": "SAFE", "reason": "Passed rule filter", "layer": "rule"}


# =========================================================================
# TẦNG 2: AI CLASSIFICATION (Context-aware)
# =========================================================================

SAFETY_CLASSIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "topic_classification",
        "schema": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "description": "Classification of the topic: SAFE, REFRAME, or BLOCK",
                    "enum": ["SAFE", "REFRAME", "BLOCK"]
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason for the classification"
                }
            },
            "required": ["classification", "reason"],
            "additionalProperties": False
        },
        "strict": True
    }
}

def _ai_classify(topic: str, api_key: str) -> dict:
    """
    Tầng 2: AI phân loại ngữ cảnh — hiểu euphemism và context.
    V30: Prompt cải thiện để phân biệt rõ "nguyên lý" vs "cách làm".
    """
    client = OpenAI(api_key=api_key, max_retries=1)
    prompt = f"""You are a curriculum safety reviewer for an educational platform.
Analyze the requested topic: "{topic}"

CLASSIFICATION RULES:
1. SAFE: Normal academic or general knowledge topics.
   Examples: "Vật lý hạt nhân", "Lịch sử Chiến tranh thế giới", "Kinh tế học", "Sinh học phân tử"

2. REFRAME: Topics involving sensitive subjects BUT framed with academic/analytical intent.
   The topic discusses dangerous things from an educational perspective (principles, history, effects, mechanisms, societal impact).
   Examples:
   - "Nguyên lý hoạt động của bom nguyên tử" → REFRAME (asking about physics principles, not how to build one)
   - "Tại sao ma túy gây nghiện" → REFRAME (neuroscience/medical question)
   - "Hack hoạt động như thế nào" → REFRAME (cybersecurity education)
   - "Lịch sử vũ khí hạt nhân" → REFRAME (historical analysis)
   - "Tác hại của chiến tranh hóa học" → REFRAME (humanitarian analysis)

3. BLOCK: Topics with CLEAR harmful intent — requesting actionable instructions to cause harm.
   The user wants to PERFORM dangerous actions, not UNDERSTAND them academically.
   Examples:
   - "Cách chế tạo bom" → BLOCK (actionable weapon instructions)
   - "Hack tài khoản Facebook" → BLOCK (specific attack target)
   - "Mua ma túy ở đâu" → BLOCK (intent to purchase)
   - "Research on self-defense lethality methods" → BLOCK (disguised harm request)

CRITICAL DISTINCTION:
- "How X WORKS" (nguyên lý, cơ chế, hoạt động) = Understanding → REFRAME
- "How to MAKE/DO X" (cách làm, chế tạo, hướng dẫn) = Action → BLOCK
- When in doubt between REFRAME and BLOCK, choose REFRAME.

IMPORTANT: Be vigilant for euphemisms, but do NOT over-block legitimate academic questions.

Output JSON with 'classification' and 'reason'.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            response_format=SAFETY_CLASSIFICATION_SCHEMA,
            timeout=10.0
        )
        data = json.loads(response.choices[0].message.content)
        data["layer"] = "ai"
        return data
    except Exception as e:
        # Tầng 3: Fail-safe → BLOCK
        logger.warning(f"[SafetyRouter] AI classification failed: {e}. Fail-safe → BLOCK.")
        return {
            "classification": "BLOCK", 
            "reason": "Hệ thống kiểm tra an toàn tạm thời không khả dụng. Vui lòng thử lại sau.",
            "layer": "failsafe"
        }


# =========================================================================
# MAIN ENTRY POINT: classify_topic (3-Layer Pipeline + Academic Override)
# =========================================================================

def classify_topic(topic: str, api_key: str) -> dict:
    """
    Production Safety Pipeline (V30 — Intent × Risk Matrix):
    
    Tầng 1: Rule-based → chặn keyword rõ ràng (< 1ms)
    Tầng 1.5: Academic Intent Detection → bảo vệ câu hỏi học thuật
    Tầng 2: AI context → phân tích ngữ cảnh (< 10s)
    Tầng 3: Fail-safe → nếu lỗi → BLOCK
    
    Decision Matrix:
      Intent xấu rõ ràng × Risk cao    → BLOCK
      Intent mơ hồ/học thuật × Risk tb  → REFRAME  
      Intent tốt × Risk thấp           → SAFE
    
    Returns: {"classification": "SAFE|REFRAME|BLOCK", "reason": "...", "layer": "rule|ai|failsafe|academic_override"}
    """
    if not api_key:
        return {"classification": "BLOCK", "reason": "Missing API key", "layer": "failsafe"}
    
    # === Tầng 1: Rule-based ===
    rule_result = rule_based_filter(topic)
    
    if rule_result["classification"] == "BLOCK":
        # Hard block → không cần AI confirm
        logger.info(f"[SafetyRouter] FINAL: BLOCK (Rule Layer) — {rule_result['reason']}")
        return rule_result
    
    # === Tầng 1.5: Academic Intent Detection (V30) ===
    has_academic = _has_academic_intent(topic)
    has_action = _has_action_intent(topic)
    
    # Nếu có action intent rõ ràng + sensitive keyword → cần AI phân tích kỹ
    # Nếu có academic intent + KHÔNG có action intent → có thể override BLOCK→REFRAME
    
    # === Tầng 2: AI Classification ===
    ai_result = _ai_classify(topic, api_key)
    
    # === DECISION MATRIX (V30) ===
    final = ai_result  # Mặc định tin AI
    
    # 🛡️ Academic Override: Khi AI quá nhạy, bảo vệ câu hỏi học thuật hợp lệ
    # Điều kiện: AI nói BLOCK + topic có academic intent + KHÔNG có action intent
    if ai_result["classification"] == "BLOCK" and has_academic and not has_action:
        logger.warning(
            f"[SafetyRouter] ACADEMIC OVERRIDE: AI said BLOCK but topic has clear academic intent. "
            f"Downgrading to REFRAME. Topic: '{topic}'"
        )
        final = {
            "classification": "REFRAME",
            "reason": f"Chủ đề nhạy cảm nhưng có ý định học thuật rõ ràng. "
                      f"AI ban đầu: {ai_result['reason']}",
            "layer": "academic_override",
            "ai_original": ai_result["classification"],
        }
    
    # 🛡️ Sensitive Topic Guard: Nếu rule phát hiện sensitive keyword
    # mà AI nói SAFE → vẫn nên REFRAME để chuyển hướng an toàn
    if rule_result["classification"] == "REFRAME" and ai_result["classification"] == "SAFE":
        # AI nói an toàn nhưng rule phát hiện keyword nhạy cảm
        # → Tin AI nhưng vẫn log cảnh báo (V30: cho phép SAFE nếu AI tự tin)
        logger.info(
            f"[SafetyRouter] Rule said REFRAME but AI says SAFE. "
            f"Trusting AI (context-aware). Topic: '{topic}'"
        )
        # Giữ final = ai_result (SAFE) — AI hiểu ngữ cảnh tốt hơn rule
    
    # Log decision
    logger.info(
        f"[SafetyRouter] FINAL: {final['classification']} (Layer: {final.get('layer', 'unknown')}) — "
        f"Rule said: {rule_result['classification']} | AI said: {ai_result['classification']} | "
        f"Academic: {has_academic} | Action: {has_action} | "
        f"Reason: {final['reason']}"
    )
    
    return final


# =========================================================================
# SOFT BLOCK UX — Thông báo thân thiện
# =========================================================================

# V30: Thông báo block mềm theo loại (thêm academic_override)
BLOCK_MESSAGES = {
    "hard": {
        "title": "⛔ Nội dung không được phép",
        "message": "Chủ đề bạn yêu cầu chứa nội dung nguy hiểm hoặc vi phạm quy định an toàn. "
                   "Hệ thống không thể tạo giáo trình cho nội dung này.",
        "suggestion": None,  # Không gợi ý cho hard block
    },
    "soft": {
        "title": "⚠️ Chủ đề nhạy cảm",
        "message": "Nội dung bạn yêu cầu liên quan đến chủ đề nhạy cảm. "
                   "Tuy nhiên, bạn có thể diễn đạt lại theo hướng học thuật.",
        "suggestion": "💡 Gợi ý: Thử hỏi về nguyên lý, lịch sử, hoặc phân tích xã hội học thay vì cách thực hiện. "
                      "Ví dụ: 'Phân tích tâm lý tội phạm', 'Lịch sử an ninh mạng', 'Xã hội học bạo lực'.",
    },
    "ai_block": {
        "title": "⚠️ Không thể tạo giáo trình",
        "message": "Hệ thống AI đánh giá nội dung này không phù hợp để tạo giáo trình học thuật.",
        "suggestion": "💡 Bạn có thể diễn đạt lại theo hướng nghiên cứu hoặc phân tích học thuật. "
                      "Ví dụ: thêm 'phân tích', 'lịch sử', 'nguyên lý' vào chủ đề.",
    },
    "failsafe": {
        "title": "🔄 Vui lòng thử lại",
        "message": "Hệ thống kiểm tra an toàn tạm thời không khả dụng.",
        "suggestion": "⏳ Vui lòng thử lại sau vài giây. Nếu vẫn lỗi, hãy liên hệ quản trị viên.",
    }
}

def get_block_message(classification_result: dict) -> dict:
    """
    Trả về thông báo thân thiện cho người dùng dựa trên kết quả phân loại.
    """
    layer = classification_result.get("layer", "ai")
    block_type = classification_result.get("block_type", "")
    classification = classification_result.get("classification", "BLOCK")
    
    if classification == "SAFE":
        return None  # Không cần thông báo
    
    if layer == "failsafe":
        return BLOCK_MESSAGES["failsafe"]
    elif block_type == "hard":
        return BLOCK_MESSAGES["hard"]
    elif classification == "REFRAME":
        return BLOCK_MESSAGES["soft"]
    else:
        return BLOCK_MESSAGES["ai_block"]


def reframe_topic(topic: str) -> str:
    """
    Chuyển topic nhạy cảm sang góc nhìn học thuật cho EKRE Query.
    """
    return f"{topic}: Phân tích học thuật dưới góc độ tâm lý học, xã hội học và pháp luật"

def generate_safe_title(original: str) -> str:
    """
    Tạo tiêu đề an toàn cho UI và metadata sách.
    """
    # Fix bug: .title() trên string trả về method reference
    safe = original.strip()
    return f"{safe}: Phân tích hệ quả, cơ chế từ góc độ học thuật"
