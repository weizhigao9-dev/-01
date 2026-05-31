from src.offline_demo import RAG_ANSWERS


def test_out_of_scope_questions_refuse_to_guess():
    assert "没有找到依据" in RAG_ANSWERS["任课教师下周三的办公室具体地点在哪里？"]
    assert "没有找到依据" in RAG_ANSWERS["某同学的真实学号是多少？"]


def test_scoring_answer_contains_percentages():
    answer = RAG_ANSWERS["本课程实践项目评分标准是什么？"]
    assert "30%" in answer
    assert "20%" in answer
