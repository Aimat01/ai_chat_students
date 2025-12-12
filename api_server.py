"""
Запуск: uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from dotenv import load_dotenv
from school_ai_platform import SchoolAIPlatformV3
import uuid
from datetime import datetime
from flashcard import FlashcardSystem, FlashcardDeckConfig, Flashcard, FlashcardSession


from quiz_system import QuizSystem, QuizConfig, QuizQuestion, QuizResult

load_dotenv()

app = FastAPI(
    title="School AI Platform API",
    description="API для образовательной AI-платформы",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Хранилище платформ для разных пользователей (в памяти)
# В продакшене использовать Redis или баз данных
platforms = {}

sessions = {}

quiz_systems = {}

flashcard_systems = {}

active_quizzes: Dict[str, Dict] = {}

active_decks: Dict[str, Dict] = {}

# База данных квизов (в памяти, можно заменить на SQLite/PostgreSQL)
import json
from pathlib import Path

DB_FOLDER = Path("database")
DB_FOLDER.mkdir(exist_ok=True)

QUIZZES_DB_FILE = DB_FOLDER / "quizzes.json"
QUIZ_RESULTS_DB_FILE = DB_FOLDER / "quiz_results.json"

def load_db(file_path: Path) -> dict:
    """Загрузить базу данных из JSON файла"""
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_db(file_path: Path, data: dict):
    """Сохранить базу данных в JSON файл"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Загрузить существующие квизы при старте
saved_quizzes = load_db(QUIZZES_DB_FILE)
saved_results = load_db(QUIZ_RESULTS_DB_FILE)


class LanguageSelect(BaseModel):
    language: str = "ru"  # en, ru, kk


class ChatMessage(BaseModel):
    user_id: str
    message: str
    language: Optional[str] = "ru"


class ChatResponse(BaseModel):
    user_id: str
    message: str
    response: str
    timestamp: str


class SummaryRequest(BaseModel):
    user_id: str
    topic: str
    language: Optional[str] = "ru"


class UploadMaterialsRequest(BaseModel):
    folder_path: str


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    language: str
    message_count: int
    created_at: str


class QuizGenerateRequest(BaseModel):
    """Запрос на генерацию квиза"""
    user_id: str
    mode: str  # "topic_select", "free_text", "adaptive"
    topic: Optional[str] = None
    num_questions: int = 15
    difficulty: str = "medium"  # easy, medium, hard
    language: str = "ru"


class QuizQuestionResponse(BaseModel):
    """Ответ с вопросом квиза"""
    quiz_id: str
    question_number: int
    total_questions: int
    question: str
    options: List[str]
    topic: str


class QuizAnswerSubmit(BaseModel):
    """Ответ на вопрос квиза"""
    quiz_id: str
    question_number: int
    selected_answer: int 


class QuizAnswerResponse(BaseModel):
    """Результат проверки ответа"""
    is_correct: bool
    correct_answer: int
    explanation: str
    selected_answer: int


class QuizCompleteRequest(BaseModel):
    """Завершение квиза"""
    quiz_id: str
    user_id: str
    answers: List[Dict] 
    time_taken: Optional[int] = None


class QuizFinalResult(BaseModel):
    """Финальный результат квиза"""
    quiz_id: str
    score_percentage: float
    correct_answers: int
    wrong_answers: int
    total_questions: int
    weak_topics: List[str]
    recommendations: List[str]
    detailed_answers: List[Dict]


class TopicInfo(BaseModel):
    """Информация о доступной теме"""
    name: str
    subject: str
    full_name: str
    chunks: int


class FlashcardGenerateRequest(BaseModel):
    user_id: str
    mode: str  # "topic_select", "free_text"
    topic: Optional[str] = None
    num_cards: int = 20
    difficulty: str = "medium"
    language: str = "ru"


class FlashcardResponse(BaseModel):
    term: str
    definition: str
    example: Optional[str]
    topic: str


class FlashcardReviewRequest(BaseModel):
    deck_id: str
    card_index: int
    knew_it: bool  # Знал ли ответ


class DeckProgressResponse(BaseModel):
    deck_id: str
    total_cards: int
    reviewed: int
    known: int
    learning: int
    remaining: int


def get_platform(language: str = "ru"):
    """Получить или создать платформу для языка"""
    if language not in platforms:
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        PINECONE_KEY = os.getenv("PINECONE_API_KEY")
        
        if not OPENAI_KEY or not PINECONE_KEY:
            raise HTTPException(
                status_code=500,
                detail="API keys not configured. Set OPENAI_API_KEY and PINECONE_API_KEY"
            )
        
        platforms[language] = SchoolAIPlatformV3(
            OPENAI_KEY,
            PINECONE_KEY,
            language=language
        )
    
    return platforms[language]


def get_quiz_system(language: str = "ru"):
    """Получить систему квизов для языка"""
    if language not in quiz_systems:
        platform = get_platform(language)
        quiz_systems[language] = QuizSystem(platform)
    return quiz_systems[language]


def get_flashcard_system(language: str = "ru"):
    """Получить систему карточек для языка"""
    if language not in flashcard_systems:
        platform = get_platform(language)
        flashcard_systems[language] = FlashcardSystem(platform)
    return flashcard_systems[language]


def get_or_create_session(user_id: str, language: str = "ru"):
    """Получить или создать сессию с историей диалога"""
    session_key = f"{user_id}_{language}"
    
    if session_key not in sessions:
        sessions[session_key] = {
            "session_id": str(uuid.uuid4()),
            "user_id": user_id,
            "language": language,
            "conversation_history": [],
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
    
    sessions[session_key]["last_activity"] = datetime.now().isoformat()
    return sessions[session_key]


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "School AI Platform API v3.0",
        "docs": "/docs",
        "features": [
            "Multi-language support (en, ru, kk)",
            "Conversation context",
            "EPUB support",
            "Summary generation",
            "Quiz system"
        ]
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "active_languages": list(platforms.keys()),
        "active_sessions": len(sessions),
        "active_quizzes": len(active_quizzes)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Отправить сообщение в чат
    
    Поддерживает контекст диалога - AI помнит предыдущие сообщения!
    """
    try:
        platform = get_platform(message.language)
        
        session = get_or_create_session(message.user_id, message.language)
        
        matches = platform.search_relevant_content(message.message, top_k=5)
        
        response = platform.generate_response_with_context(
            message.message,
            matches,
            session["conversation_history"]
        )
        
        session["conversation_history"].append({
            "role": "user",
            "content": message.message
        })
        session["conversation_history"].append({
            "role": "assistant",
            "content": response
        })
        
        if len(session["conversation_history"]) > 20:
            session["conversation_history"] = session["conversation_history"][-20:]
        
        return ChatResponse(
            user_id=message.user_id,
            message=message.message,
            response=response,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summary")
async def generate_summary(request: SummaryRequest):
    """Сгенерировать конспект по теме"""
    try:
        platform = get_platform(request.language)
        matches = platform.search_relevant_content(request.topic, top_k=10)
        summary = platform.generate_summary(request.topic, matches)
        
        return {
            "user_id": request.user_id,
            "topic": request.topic,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{user_id}", response_model=SessionInfo)
async def get_session_info(user_id: str, language: str = "ru"):
    """Получить информацию о сессии пользователя"""
    session_key = f"{user_id}_{language}"
    
    if session_key not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_key]
    
    return SessionInfo(
        session_id=session["session_id"],
        user_id=session["user_id"],
        language=session["language"],
        message_count=len(session["conversation_history"]) // 2,
        created_at=session["created_at"]
    )


@app.delete("/session/{user_id}")
async def clear_session(user_id: str, language: str = "ru"):
    """Очистить сессию пользователя (начать новый диалог)"""
    session_key = f"{user_id}_{language}"
    
    if session_key in sessions:
        del sessions[session_key]
        return {"message": "Session cleared", "user_id": user_id}
    
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/history/{user_id}")
async def get_history(user_id: str, language: str = "ru", limit: int = 10):
    """Получить историю сообщений пользователя"""
    session_key = f"{user_id}_{language}"
    
    if session_key not in sessions:
        return {"user_id": user_id, "messages": []}
    
    session = sessions[session_key]
    history = session["conversation_history"][-limit*2:]
    
    formatted_history = []
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            formatted_history.append({
                "question": history[i]["content"],
                "answer": history[i+1]["content"]
            })
    
    return {
        "user_id": user_id,
        "language": language,
        "messages": formatted_history
    }


@app.post("/upload_materials")
async def upload_materials(request: UploadMaterialsRequest, background_tasks: BackgroundTasks):
    """
    Загрузить учебные материалы (работает в фоне)
    
    Примечание: В продакшене лучше использовать задачи Celery
    """
    try:
        platform = get_platform("ru")
        
        background_tasks.add_task(
            platform.process_materials_folder,
            request.folder_path
        )
        
        return {
            "message": "Materials upload started",
            "folder": request.folder_path,
            "status": "processing"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics(language: str = "ru"):
    """Получить статистику базы знаний"""
    try:
        platform = get_platform(language)
        stats = platform.index.describe_index_stats()
        topics = platform.load_topics_list()
        
        return {
            "total_chunks": stats.total_vector_count,
            "total_topics": len(topics),
            "active_sessions": len([s for s in sessions.values() if s["language"] == language]),
            "recent_topics": topics[-5:] if topics else []
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subjects")
async def get_subjects(language: str = "ru"):
    """Получить список доступных предметов"""
    try:
        platform = get_platform(language)
        return {
            "language": language,
            "subjects": platform.subjects
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/languages")
async def get_supported_languages():
    """Получить список поддерживаемых языков"""
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "ru", "name": "Русский"},
            {"code": "kk", "name": "Қазақша"}
        ]
    }


@app.get("/quiz/topics", response_model=List[TopicInfo])
async def get_quiz_topics(language: str = "ru"):
    """
    Получить список доступных тем для квиза
    
    Используется для режима "topic_select"
    """
    try:
        quiz_system = get_quiz_system(language)
        topics = quiz_system.get_available_topics()
        
        return topics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/generate")
async def generate_quiz(request: QuizGenerateRequest):
    """
    Сгенерировать новый квиз
    
    Возвращает quiz_id для прохождения
    """
    try:
        quiz_system = get_quiz_system(request.language)
        
        if request.mode not in ["topic_select", "free_text", "adaptive"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Use: topic_select, free_text, or adaptive"
            )
        
        if request.mode in ["topic_select", "free_text"] and not request.topic:
            raise HTTPException(
                status_code=400,
                detail="Topic is required for this mode"
            )
        
        config = QuizConfig(
            mode=request.mode,
            topic=request.topic,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            language=request.language
        )
        
        questions = quiz_system.generate_quiz(config)
        
        if not questions:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate questions"
            )
        
        quiz_id = str(uuid.uuid4())

        quiz_data = {
            "quiz_id": quiz_id,
            "user_id": request.user_id,
            "topic": request.topic or "Информатика",
            "questions": [q.dict() for q in questions],
            "current_question": 0,
            "answers": [],
            "created_at": datetime.now().isoformat(),
            "language": request.language,
            "status": "in_progress",  # in_progress, completed, abandoned
            "mode": request.mode,
            "difficulty": request.difficulty
        }

        # Сохранить в активные квизы (для текущего прохождения)
        active_quizzes[quiz_id] = quiz_data

        # Сохранить в базу данных
        saved_quizzes[quiz_id] = quiz_data.copy()
        save_db(QUIZZES_DB_FILE, saved_quizzes)

        print(f"✅ Квиз {quiz_id} сохранен в БД")

        return {
            "quiz_id": quiz_id,
            "total_questions": len(questions),
            "topic": request.topic or "Информатика",
            "difficulty": request.difficulty,
            "message": "Quiz generated successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/{quiz_id}/question/{question_number}", response_model=QuizQuestionResponse)
async def get_quiz_question(quiz_id: str, question_number: int):
    """
    Получить конкретный вопрос квиза
    
    question_number: 1-based индекс (1, 2, 3...)
    """
    if quiz_id not in active_quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz = active_quizzes[quiz_id]
    questions = quiz["questions"]
    
    idx = question_number - 1
    if idx < 0 or idx >= len(questions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid question number. Valid range: 1-{len(questions)}"
        )
    
    question = questions[idx]
    
    return QuizQuestionResponse(
        quiz_id=quiz_id,
        question_number=question_number,
        total_questions=len(questions),
        question=question["question"],
        options=question["options"],
        topic=question["topic"]
    )


@app.post("/quiz/answer", response_model=QuizAnswerResponse)
async def submit_quiz_answer(answer: QuizAnswerSubmit):
    """
    Отправить ответ на вопрос
    
    Возвращает: правильный ли ответ + объяснение
    """
    if answer.quiz_id not in active_quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz = active_quizzes[answer.quiz_id]
    questions = quiz["questions"]
    
    idx = answer.question_number - 1
    if idx < 0 or idx >= len(questions):
        raise HTTPException(status_code=400, detail="Invalid question number")
    
    question = questions[idx]
    correct_answer = question["correct_answer"]
    is_correct = (answer.selected_answer == correct_answer)
    
    answer_record = {
        "question_number": answer.question_number,
        "question": question["question"],
        "selected_answer": answer.selected_answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "topic": question["topic"],
        "explanation": question["explanation"]
    }
    
    quiz["answers"].append(answer_record)
    
    return QuizAnswerResponse(
        is_correct=is_correct,
        correct_answer=correct_answer,
        explanation=question["explanation"],
        selected_answer=answer.selected_answer
    )


@app.post("/quiz/complete", response_model=QuizFinalResult)
async def complete_quiz(request: QuizCompleteRequest):
    """
    Завершить квиз и получить итоговые результаты
    
    Анализирует ошибки и даёт рекомендации
    """
    if request.quiz_id not in active_quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz = active_quizzes[request.quiz_id]
    quiz_system = get_quiz_system(quiz["language"])
    
    answers = quiz["answers"]
    score = quiz_system.calculate_score(answers)
    
    weak_topics = []
    topic_errors = {}
    
    for answer in answers:
        if not answer["is_correct"]:
            topic = answer["topic"]
            topic_errors[topic] = topic_errors.get(topic, 0) + 1
    
    weak_topics = sorted(
        topic_errors.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    weak_topics = [topic for topic, _ in weak_topics]
    
    recommendations = quiz_system.get_recommendations(weak_topics, quiz["language"])
    
    result = QuizResult(
        quiz_id=request.quiz_id,
        user_id=request.user_id,
        topic=quiz["topic"],
        total_questions=score["total"],
        correct_answers=score["correct"],
        wrong_answers=score["wrong"],
        score_percentage=score["percentage"],
        time_taken=request.time_taken,
        answers=answers,
        weak_topics=weak_topics,
        timestamp=datetime.now().isoformat()
    )
    
    quiz_system.save_result(result)

    # Обновить статус квиза в БД
    if request.quiz_id in saved_quizzes:
        saved_quizzes[request.quiz_id]["status"] = "completed"
        saved_quizzes[request.quiz_id]["completed_at"] = datetime.now().isoformat()
        saved_quizzes[request.quiz_id]["final_score"] = score["percentage"]
        save_db(QUIZZES_DB_FILE, saved_quizzes)

    # Сохранить результат в БД результатов
    result_id = str(uuid.uuid4())
    saved_results[result_id] = {
        "result_id": result_id,
        "quiz_id": request.quiz_id,
        "user_id": request.user_id,
        "topic": quiz["topic"],
        "score_percentage": score["percentage"],
        "correct_answers": score["correct"],
        "wrong_answers": score["wrong"],
        "total_questions": score["total"],
        "weak_topics": weak_topics,
        "answers": answers,
        "timestamp": datetime.now().isoformat()
    }
    save_db(QUIZ_RESULTS_DB_FILE, saved_results)
    print(f"✅ Результат квиза {request.quiz_id} сохранен в БД")

    del active_quizzes[request.quiz_id]

    return QuizFinalResult(
        quiz_id=request.quiz_id,
        score_percentage=score["percentage"],
        correct_answers=score["correct"],
        wrong_answers=score["wrong"],
        total_questions=score["total"],
        weak_topics=weak_topics,
        recommendations=recommendations,
        detailed_answers=answers
    )


@app.get("/quiz/history/{user_id}")
async def get_quiz_history(user_id: str, limit: int = 10):
    """
    Получить историю квизов пользователя
    """
    try:
        quiz_system = get_quiz_system("ru") 
        result_file = quiz_system.results_folder / f"{user_id}_results.json"
        
        if not result_file.exists():
            return {"user_id": user_id, "quizzes": []}
        
        import json
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        quizzes = data.get("quizzes", [])[-limit:]
        
        return {
            "user_id": user_id,
            "total_quizzes": len(data.get("quizzes", [])),
            "recent_quizzes": quizzes
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/stats/{user_id}")
async def get_user_quiz_stats(user_id: str):
    """
    Получить статистику квизов пользователя
    
    Средний балл, сильные/слабые темы, прогресс
    """
    try:
        quiz_system = get_quiz_system("ru")
        result_file = quiz_system.results_folder / f"{user_id}_results.json"
        
        if not result_file.exists():
            return {
                "user_id": user_id,
                "total_quizzes": 0,
                "average_score": 0,
                "weak_topics": []
            }
        
        import json
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        quizzes = data.get("quizzes", [])
        
        if not quizzes:
            return {
                "user_id": user_id,
                "total_quizzes": 0,
                "average_score": 0,
                "weak_topics": []
            }
        
        total_quizzes = len(quizzes)
        avg_score = sum(q["score_percentage"] for q in quizzes) / total_quizzes
        
        weak_topics = quiz_system.get_user_weak_topics(user_id, limit=5)
        
        recent_scores = [q["score_percentage"] for q in quizzes[-5:]]
        
        return {
            "user_id": user_id,
            "total_quizzes": total_quizzes,
            "average_score": round(avg_score, 2),
            "weak_topics": weak_topics,
            "recent_scores": recent_scores,
            "total_questions_answered": sum(q["total_questions"] for q in quizzes)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/flashcards/topics")
async def get_flashcard_topics(language: str = "ru"):
    """
    Получить список доступных тем для карточек
    """
    try:
        fc_system = get_flashcard_system(language)
        topics = fc_system.get_available_topics()
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/flashcards/generate")
async def generate_flashcards(request: FlashcardGenerateRequest):
    """
    Сгенерировать колоду карточек
    
    Возвращает deck_id для изучения
    """
    try:
        fc_system = get_flashcard_system(request.language)
        
        if request.mode not in ["topic_select", "free_text"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Use: topic_select or free_text"
            )
        
        if request.mode in ["topic_select", "free_text"] and not request.topic:
            raise HTTPException(
                status_code=400,
                detail="Topic is required for this mode"
            )
        
        config = FlashcardDeckConfig(
            mode=request.mode,
            topic=request.topic,
            num_cards=request.num_cards,
            difficulty=request.difficulty,
            language=request.language
        )
        
        cards = fc_system.generate_flashcards(config)
        
        if not cards:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate flashcards"
            )
        
        deck_id = str(uuid.uuid4())
        
        active_decks[deck_id] = {
            "deck_id": deck_id,
            "user_id": request.user_id,
            "topic": request.topic or "Информатика",
            "cards": [c.dict() for c in cards],
            "current_index": 0,
            "reviews": {},  
            "created_at": datetime.now().isoformat(),
            "language": request.language
        }
        
        return {
            "deck_id": deck_id,
            "total_cards": len(cards),
            "topic": request.topic or "Информатика",
            "difficulty": request.difficulty,
            "message": "Flashcard deck generated successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flashcards/{deck_id}/card/{card_index}", response_model=FlashcardResponse)
async def get_flashcard(deck_id: str, card_index: int):
    """
    Получить карточку по индексу (0-based)
    
    Возвращает только ТЕРМИН (лицевая сторона)
    """
    if deck_id not in active_decks:
        raise HTTPException(status_code=404, detail="Deck not found")
    
    deck = active_decks[deck_id]
    cards = deck["cards"]
    
    if card_index < 0 or card_index >= len(cards):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid card index. Valid range: 0-{len(cards)-1}"
        )
    
    card = cards[card_index]
    
    return FlashcardResponse(
        term=card["term"],
        definition=card["definition"],
        example=card.get("example"),
        topic=card["topic"]
    )


@app.post("/flashcards/review")
async def review_flashcard(review: FlashcardReviewRequest):
    """
    Отметить знал ли карточку
    
    knew_it: True = знал, False = не знал
    """
    if review.deck_id not in active_decks:
        raise HTTPException(status_code=404, detail="Deck not found")
    
    deck = active_decks[review.deck_id]
    
    if review.card_index not in deck["reviews"]:
        deck["reviews"][review.card_index] = []
    
    deck["reviews"][review.card_index].append({
        "knew_it": review.knew_it,
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "card_index": review.card_index,
        "knew_it": review.knew_it,
        "review_count": len(deck["reviews"][review.card_index])
    }


@app.get("/flashcards/{deck_id}/progress", response_model=DeckProgressResponse)
async def get_deck_progress(deck_id: str):
    """
    Получить прогресс изучения колоды
    """
    if deck_id not in active_decks:
        raise HTTPException(status_code=404, detail="Deck not found")
    
    deck = active_decks[deck_id]
    cards = deck["cards"]
    reviews = deck["reviews"]
    
    reviewed = len(reviews)
    known = 0
    learning = 0
    
    for card_idx, card_reviews in reviews.items():
        if not card_reviews:
            continue
        
        recent = card_reviews[-3:]
        correct = sum(1 for r in recent if r.get("knew_it", False))
        
        if len(recent) >= 2 and correct >= 2:
            known += 1
        else:
            learning += 1
    
    return DeckProgressResponse(
        deck_id=deck_id,
        total_cards=len(cards),
        reviewed=reviewed,
        known=known,
        learning=learning,
        remaining=len(cards) - reviewed
    )


@app.post("/flashcards/{deck_id}/complete")
async def complete_flashcard_session(deck_id: str, user_id: str):
    """
    Завершить сессию изучения карточек
    
    Сохраняет прогресс и возвращает статистику
    """
    if deck_id not in active_decks:
        raise HTTPException(status_code=404, detail="Deck not found")
    
    deck = active_decks[deck_id]
    fc_system = get_flashcard_system(deck["language"])
    
    reviews = deck["reviews"]
    reviewed_count = len(reviews)
    
    known_count = 0
    for card_idx, card_reviews in reviews.items():
        if card_reviews:
            recent = card_reviews[-3:]
            correct = sum(1 for r in recent if r.get("knew_it", False))
            if len(recent) >= 2 and correct >= 2:
                known_count += 1
    
    session = FlashcardSession(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        deck_id=deck_id,
        topic=deck["topic"],
        total_cards=len(deck["cards"]),
        reviewed_cards=reviewed_count,
        known_cards=known_count,
        learning_cards=reviewed_count - known_count,
        cards_data=[
            {
                "term": deck["cards"][idx]["term"],
                "reviews": reviews.get(idx, [])
            }
            for idx in range(len(deck["cards"]))
        ],
        timestamp=datetime.now().isoformat()
    )
    
    fc_system.save_session(session)
    
    del active_decks[deck_id]
    
    mastery_percentage = (known_count / len(deck["cards"]) * 100) if len(deck["cards"]) > 0 else 0
    
    return {
        "session_id": session.session_id,
        "total_cards": len(deck["cards"]),
        "reviewed": reviewed_count,
        "known": known_count,
        "learning": reviewed_count - known_count,
        "mastery_percentage": round(mastery_percentage, 1),
        "message": "Session completed successfully"
    }


@app.get("/flashcards/history/{user_id}")
async def get_flashcard_history(user_id: str, limit: int = 10):
    """
    Получить историю изучения карточек
    """
    try:
        fc_system = get_flashcard_system("ru")
        progress = fc_system.get_user_progress(user_id)
        
        return {
            "user_id": user_id,
            "total_sessions": progress.get("total_sessions", 0),
            "total_cards_reviewed": progress.get("total_cards_reviewed", 0),
            "total_cards_known": progress.get("total_cards_known", 0),
            "topics_studied": progress.get("topics_studied", []),
            "recent_sessions": progress.get("recent_sessions", [])[-limit:]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flashcards/stats/{user_id}")
async def get_flashcard_stats(user_id: str):
    """
    Получить статистику изучения карточек
    """
    try:
        fc_system = get_flashcard_system("ru")
        progress = fc_system.get_user_progress(user_id)
        
        total_sessions = progress.get("total_sessions", 0)
        total_reviewed = progress.get("total_cards_reviewed", 0)
        total_known = progress.get("total_cards_known", 0)
        
        mastery_rate = (total_known / total_reviewed * 100) if total_reviewed > 0 else 0
        
        return {
            "user_id": user_id,
            "total_sessions": total_sessions,
            "total_cards_reviewed": total_reviewed,
            "total_cards_known": total_known,
            "mastery_rate": round(mastery_rate, 1),
            "topics_studied": progress.get("topics_studied", []),
            "average_cards_per_session": round(total_reviewed / total_sessions, 1) if total_sessions > 0 else 0
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)