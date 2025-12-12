"""
Система карточек (Flashcards) для запоминания терминов
"""

from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
import json
from pathlib import Path
import random


class Flashcard(BaseModel):
    """Модель одной карточки"""
    term: str  # Термин 
    definition: str  # Определение 
    example: Optional[str] = None  
    topic: str  # Тема
    difficulty: str = "medium"  # easy, medium, hard


class FlashcardDeckConfig(BaseModel):
    """Конфигурация колоды карточек"""
    mode: str  # "topic_select", "free_text", "adaptive"
    topic: Optional[str] = None
    num_cards: int = 20
    difficulty: str = "medium"  # easy, medium, hard
    language: str = "ru"


class FlashcardSession(BaseModel):
    """Сессия изучения карточек"""
    session_id: str
    user_id: str
    deck_id: str
    topic: str
    total_cards: int
    reviewed_cards: int
    known_cards: int  
    learning_cards: int  
    cards_data: List[Dict]  
    timestamp: str


class FlashcardSystem:
    """
    Система карточек для запоминания
    
    Функции:
    - Генерация карточек из материалов
    - Spaced repetition (интервальное повторение)
    - Отслеживание прогресса
    - Статистика изучения
    """
    
    def __init__(self, platform):
        """
        platform: SchoolAIPlatformV3 instance
        """
        self.platform = platform
        self.sessions_folder = Path("flashcard_sessions")
        self.sessions_folder.mkdir(exist_ok=True)
        
        self.prompts = {
            "ru": """На основе этого материала создай {num} карточек для запоминания терминов по информатике уровня {difficulty}.

Материал:
{context}

ВАЖНО: Ответь ТОЛЬКО валидным JSON массивом без дополнительного текста!

Формат ответа (строго JSON):
[
  {{
    "term": "Переменная",
    "definition": "Именованная область памяти для хранения данных",
    "example": "x = 5  # x - это переменная",
    "topic": "Python основы"
  }},
  {{
    "term": "Функция",
    "definition": "Блок кода, который выполняет определённую задачу",
    "example": "def hello(): print('Hi')",
    "topic": "Python основы"
  }}
]

Требования:
- Термины должны быть важными для понимания темы
- Определения должны быть КРАТКИМИ (1-2 предложения)
- Примеры должны быть ПРОСТЫМИ и понятными
- Термины должны быть разной сложности ({difficulty})""",
            
            "en": """Based on this material, create {num} flashcards for computer science terms at {difficulty} level.

Material:
{context}

IMPORTANT: Reply ONLY with valid JSON array without additional text!

Response format (strict JSON):
[
  {{
    "term": "Variable",
    "definition": "Named storage location for data",
    "example": "x = 5  # x is a variable",
    "topic": "Python basics"
  }}
]

Requirements:
- Terms must be important for understanding the topic
- Definitions must be BRIEF (1-2 sentences)
- Examples must be SIMPLE and clear
- Terms should vary in difficulty ({difficulty})""",
            
            "kk": """Осы материалға негізделген {num} информатика терминдерін есте сақтауға арналған {difficulty} деңгейіндегі карточкалар жаса.

Материал:
{context}

МАҢЫЗДЫ: Қосымша мәтінсіз тек валидті JSON массивімен жауап бер!

Жауап форматы (қатаң JSON):
[
  {{
    "term": "Айнымалы",
    "definition": "Деректерді сақтауға арналған атаулы жад аймағы",
    "example": "x = 5  # x - айнымалы",
    "topic": "Python негіздері"
  }}
]

Талаптар:
- Терминдер тақырыпты түсіну үшін маңызды болуы керек
- Анықтамалар ҚЫСҚА болуы керек (1-2 сөйлем)
- Мысалдар ҚАРАПАЙЫМ және түсінікті болуы керек
- Терминдер әртүрлі қиындықта болуы керек ({difficulty})"""
        }
    
    def get_available_topics(self) -> List[str]:
        """Получить список доступных тем"""
        topics = self.platform.load_topics_list()
        
        if not topics:
            return []
        
        unique_topics = []
        seen = set()
        
        for topic_data in topics:
            topic_name = topic_data.get("topic", "")
            subject = topic_data.get("subject", "")
            full_name = f"{subject}: {topic_name}"
            
            if full_name not in seen:
                seen.add(full_name)
                unique_topics.append({
                    "name": topic_name,
                    "subject": subject,
                    "full_name": full_name
                })
        
        return unique_topics
    
    def generate_flashcards(self, config: FlashcardDeckConfig) -> List[Flashcard]:
        """
        Генерация карточек по конфигурации
        """
        print(f"\n🃏 Генерация карточек:")
        print(f"   Режим: {config.mode}")
        print(f"   Тема: {config.topic}")
        print(f"   Карточек: {config.num_cards}")
        print(f"   Сложность: {config.difficulty}")
        
        search_query = config.topic or "информатика основы"
        matches = self.platform.search_relevant_content(search_query, top_k=15)
        
        if not matches:
            raise ValueError("Не найдено материалов по этой теме")
        
        context = "\n\n".join([
            f"[{m.metadata.get('topic', 'Материал')}]\n{m.metadata.get('text', '')[:400]}"
            for m in matches[:10]
        ])
        
        prompt_template = self.prompts.get(config.language, self.prompts["ru"])
        prompt = prompt_template.format(
            num=config.num_cards,
            difficulty=config.difficulty,
            context=context
        )
        
        print("   🤖 Генерация карточек через AI...")
        
        response = self.platform.openai_client.chat.completions.create(
            model=self.platform.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You create flashcards for students. Always respond with valid JSON array only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=1
        )
        
        try:
            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            if not response_text.startswith("["):
                start = response_text.find("[")
                if start != -1:
                    response_text = response_text[start:]
            
            cards_data = json.loads(response_text)
            
            if isinstance(cards_data, dict):
                for key in cards_data:
                    if isinstance(cards_data[key], list):
                        cards_data = cards_data[key]
                        break
            
            if not isinstance(cards_data, list):
                raise ValueError("Ответ не является массивом")
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"Ответ AI: {response_text[:200]}...")
            raise ValueError("AI вернул невалидный JSON")
        
        cards = []
        for idx, card_data in enumerate(cards_data):
            try:
                card = Flashcard(
                    term=card_data["term"],
                    definition=card_data["definition"],
                    example=card_data.get("example"),
                    topic=card_data.get("topic", config.topic or "Информатика"),
                    difficulty=config.difficulty
                )
                cards.append(card)
            except Exception as e:
                print(f"   ⚠️ Ошибка в карточке {idx+1}: {e}")
                continue
        
        print(f"   ✅ Создано {len(cards)} карточек\n")
        
        return cards
    
    def save_session(self, session: FlashcardSession):
        """Сохранение сессии изучения"""
        try:
            session_file = self.sessions_folder / f"{session.user_id}_sessions.json"
            
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"user_id": session.user_id, "sessions": []}
            
            data["sessions"].append(session.dict())
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"⚠️ Ошибка сохранения сессии: {e}")
            return False
    
    def get_user_progress(self, user_id: str) -> Dict:
        """Получить прогресс пользователя"""
        try:
            session_file = self.sessions_folder / f"{user_id}_sessions.json"
            
            if not session_file.exists():
                return {
                    "total_sessions": 0,
                    "total_cards_reviewed": 0,
                    "total_cards_known": 0,
                    "topics_studied": []
                }
            
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sessions = data.get("sessions", [])
            
            total_reviewed = sum(s["reviewed_cards"] for s in sessions)
            total_known = sum(s["known_cards"] for s in sessions)
            
            topics = list(set(s["topic"] for s in sessions))
            
            return {
                "total_sessions": len(sessions),
                "total_cards_reviewed": total_reviewed,
                "total_cards_known": total_known,
                "topics_studied": topics,
                "recent_sessions": sessions[-5:]
            }
        
        except Exception as e:
            print(f"⚠️ Ошибка получения прогресса: {e}")
            return {}
    
    def calculate_mastery(self, card_reviews: List[Dict]) -> str:
        """Рассчитать уровень владения карточкой"""
        if not card_reviews:
            return "new"
        
        correct_count = sum(1 for r in card_reviews if r.get("correct", False))
        total = len(card_reviews)
        
        if total < 3:
            return "learning"
        
        percentage = (correct_count / total) * 100
        
        if percentage >= 80:
            return "known"
        elif percentage >= 50:
            return "learning"
        else:
            return "difficult"