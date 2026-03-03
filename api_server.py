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
from flashcard import FlashcardSystem, FlashcardDeckConfig


from quiz_system import QuizSystem, QuizConfig

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

import json
from pathlib import Path


class LanguageSelect(BaseModel):
    language: str = "ru"  # en, ru, kk


class TitleRequest(BaseModel):
    message: str
    language: str = "ru"


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



class PlatformQuizGenerateRequest(BaseModel):
    """Запрос на генерацию квиза для образовательной платформы (от школьника)"""
    context: str              # Тема / описание контекста для ИИ
    difficulty: str           # easy, medium, hard — обязателен
    is_private: bool          # обязателен
    num_questions: int        # обязателен
    categories: List[int]     # обязателен
    language: str = "ru"


class TopicInfo(BaseModel):
    """Информация о доступной теме"""
    name: str
    subject: str
    full_name: str
    chunks: int


class PlatformFlashcardGenerateRequest(BaseModel):
    """Запрос на генерацию карточек для образовательной платформы (от школьника)"""
    context: str          # Тема / описание контекста для ИИ — обязателен
    num_cards: int        # Количество карточек — обязателен
    categories: List[int] # Категории — обязателен
    language: str = "ru"



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
        "active_sessions": len(sessions)
    }


@app.post("/generate-title")
async def generate_title(request: TitleRequest):
    """Сгенерировать короткий тайтл для чат-сессии по первому сообщению"""
    try:
        platform = get_platform(request.language)
        response = platform.openai_client.chat.completions.create(
            model=platform.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a short chat title (3-5 words) based on the user's message. The title should reflect the topic. Return ONLY the title text, no quotes, no punctuation at the end."
                },
                {"role": "user", "content": request.message}
            ],
        )
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        return {"title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/quiz/generate-for-platform")
async def generate_quiz_for_platform(request: PlatformQuizGenerateRequest):
    """
    Сгенерировать квиз для образовательной платформы.

    Школьник выбирает: context, difficulty, is_private, num_questions, categories.
    ИИ генерирует вопросы и возвращает JSON в формате Go-бэкенда.
    """
    try:
        quiz_system = get_quiz_system(request.language)

        config = QuizConfig(
            mode="free_text",
            topic=request.context,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            language=request.language,
        )

        questions = quiz_system.generate_quiz(config)

        if not questions:
            raise HTTPException(status_code=500, detail="Failed to generate questions")

        # Конвертируем в формат Go-бэкенда
        formatted_questions = []
        for q in questions:
            options = []
            for i, option_text in enumerate(q.options):
                options.append({
                    "optionText": option_text,
                    "isCorrect": i == q.correct_answer,
                })
            formatted_questions.append({
                "question": q.question,
                "options": options,
            })

        return {
            "title": request.context,
            "description": f"AI-сгенерированный тест по теме: {request.context}",
            "difficulty": request.difficulty,
            "isPrivate": request.is_private,
            "tags": [request.context.lower().replace(" ", "_")],
            "categories": request.categories,
            "questions": formatted_questions,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/flashcards/generate-for-platform")
async def generate_flashcards_for_platform(request: PlatformFlashcardGenerateRequest):
    """
    Сгенерировать карточки для образовательной платформы.

    Школьник выбирает: context, num_cards, categories.
    ИИ генерирует карточки и возвращает JSON в формате Go-бэкенда.
    """
    try:
        fc_system = get_flashcard_system(request.language)

        config = FlashcardDeckConfig(
            mode="free_text",
            topic=request.context,
            num_cards=request.num_cards,
            language=request.language,
        )

        cards = fc_system.generate_flashcards(config)

        if not cards:
            raise HTTPException(status_code=500, detail="Failed to generate flashcards")

        # Конвертируем term/definition → question/answer (формат Go-бэкенда)
        formatted_cards = [
            {"question": c.term, "answer": c.definition}
            for c in cards
        ]

        return {
            "title": request.context,
            "description": f"AI-сгенерированные карточки по теме: {request.context}",
            "tags": [request.context.lower().replace(" ", "_")],
            "categories": request.categories,
            "cards": formatted_cards,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)