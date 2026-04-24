# -*- coding: utf-8 -*-
"""
Structured Output Schemas (V18.1)
Dùng cho OpenAI response_format={"type": "json_schema"}.
Tất cả schemas được định nghĩa tập trung tại đây để dễ bảo trì.
"""

# --- Schema cho Micro-Writer (viet_noi_dung_muc) ---
SECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "section_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Tiêu đề mục, không có số thứ tự"
                },
                "fact_mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "span": {"type": "string", "description": "Trích nguyên si đoạn text gốc"},
                            "claim": {"type": "string", "description": "Mệnh đề logic được sinh ra"},
                            "confidence": {"type": "number", "description": "Chấm điểm độ tự tin từ 0 đến 1"}
                        },
                        "required": ["source_id", "span", "claim", "confidence"],
                        "additionalProperties": False
                    }
                },
                "content": {
                    "type": "string",
                    "description": "Nội dung học thuật phân tích chuyên sâu được viết ra DỰA TRÊN CÁC CLAIMS ĐÃ CHỌN. Phải có [ID] cuối mỗi câu."
                },
                "summary": {
                    "type": "string",
                    "description": "Tóm tắt ngắn cho mục tiếp theo"
                }
            },
            "required": ["title", "fact_mappings", "content", "summary"],
            "additionalProperties": False
        }
    }
}

# --- Schema cho Batch Sections (Production-Ready V23.2) ---
BATCH_SECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "batch_section_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Tiêu đề mục (giữ nguyên gốc)"},
                            "content": {"type": "string", "description": "Nội dung học thuật phân tích chuyên sâu kèm trích dẫn [ID]"},
                            "fact_mappings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_id": {"type": "string"},
                                        "span": {"type": "string"},
                                        "claim": {"type": "string"},
                                        "confidence": {"type": "number"}
                                    },
                                    "required": ["source_id", "span", "claim", "confidence"],
                                    "additionalProperties": False
                                }
                            },
                            "summary": {"type": "string"}
                        },
                        "required": ["title", "content", "fact_mappings", "summary"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["sections"],
            "additionalProperties": False
        }
    }
}

# --- Schema cho Chapter Writer (viet_noi_dung_chuong) ---
CHAPTER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "chapter_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "fact_mappings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_id": {"type": "string"},
                                        "span": {"type": "string", "description": "Trích đoạn ngắn từ nguồn"},
                                        "claim": {"type": "string", "description": "Mệnh đề thông tin được sinh ra"}
                                    },
                                    "required": ["source_id", "span", "claim"],
                                    "additionalProperties": False
                                }
                            },
                            "content": {"type": "string"},
                            "citations": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["title", "content", "citations", "fact_mappings"],
                        "additionalProperties": False
                    }
                },
                "used_fact_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["sections", "used_fact_ids"],
            "additionalProperties": False
        }
    }
}

# --- Schema cho Fact-Locked Rewrite (viet_lai_chuong) ---
REWRITE_SCHEMA = CHAPTER_SCHEMA # Reuse chapter schema for rewrite consistency

# --- Schema cho Outline Generator (tao_dan_y) ---
OUTLINE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "outline_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {"type": "string"},
                            "meaning": {"type": "string"}
                        },
                        "required": ["term", "meaning"],
                        "additionalProperties": False
                    }
                },
                "outline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chapter_index": {"type": "integer"},
                            "title": {"type": "string"},
                            "sections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "recommended_pids": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["title", "recommended_pids"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["chapter_index", "title", "sections"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["topic", "terms", "outline"],
            "additionalProperties": False
        }
    }
}

# --- Schema cho Term Extraction (trich_xuat_thuat_ngu_toan_dien) ---
TERM_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "term_extraction_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "core_terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {"type": "string"},
                            "importance_score": {"type": "number"}
                        },
                        "required": ["term", "importance_score"],
                        "additionalProperties": False
                    }
                },
                "supporting_terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {"type": "string"},
                            "importance_score": {"type": "number"}
                        },
                        "required": ["term", "importance_score"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["core_terms", "supporting_terms"],
            "additionalProperties": False
        }
    }
}

# --- Schema cho Fact Extraction (trich_xuat_facts_tu_corpus) ---
FACT_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_extraction_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "fact": {"type": "string"}
                        },
                        "required": ["id", "fact"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["facts"],
            "additionalProperties": False
        }
    }
}

# --- Schema cho Wikipedia Discovery (openai_identify_wiki_titles) ---
WIKI_TITLES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "wiki_titles_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "titles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "lang": {"type": "string", "enum": ["vi", "en"]},
                            "reason": {"type": "string"}
                        },
                        "required": ["title", "lang", "reason"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["titles"],
            "additionalProperties": False
        }
    }
}
# --- Schema cho Học thuật Critic (kiem_chung_hoc_thuat_gemini) ---
CRITIC_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "critic_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "audit_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_index": {"type": "integer", "description": "Vị trí của claim trong danh sách đầu vào"},
                            "verdict": {"type": "string", "enum": ["YES", "NO"]},
                            "error_type": {"type": "string", "enum": ["unsupported", "partial", "contradiction", "none"]},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string", "description": "Giải thích chi tiết lỗi nếu có"}
                        },
                        "required": ["claim_index", "verdict", "error_type", "confidence", "reason"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["audit_results"],
            "additionalProperties": False
        }
    }
}
