from src.offline_demo import RAG_ANSWERS


def test_out_of_scope_questions_refuse_to_guess():
    assert "没有找到依据" in RAG_ANSWERS["任课教师下周三的办公室具体地点在哪里？"]
    assert "没有找到依据" in RAG_ANSWERS["某同学的真实学号是多少？"]


def test_chroma_answer_contains_collection_fields():
    answer = RAG_ANSWERS["Chroma collection 通常保存哪些内容？"]
    assert "document" in answer
    assert "embedding" in answer
    assert "metadata" in answer
