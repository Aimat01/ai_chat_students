"""
Система квизов для AI платформы
Поддерживает генерацию тестов по материалам
"""

from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
import json
from pathlib import Path
import random


class QuizQuestion(BaseModel):
    """Модель вопроса квиза"""
    question: str
    options: List[str]  
    correct_answer: int
    explanation: str  
    topic: str  
    difficulty: str = "medium"  # easy, medium, hard


class QuizConfig(BaseModel):
    """Конфигурация квиза"""
    mode: str  # "topic_select", "free_text", "adaptive"
    topic: Optional[str] = None
    num_questions: int = 15
    difficulty: str = "medium"  # easy, medium, hard
    language: str = "ru"


class QuizResult(BaseModel):
    """Результат прохождения квиза"""
    quiz_id: str
    user_id: str
    topic: str
    total_questions: int
    correct_answers: int
    wrong_answers: int
    score_percentage: float
    time_taken: Optional[int] = None 
    answers: List[Dict] 
    weak_topics: List[str]
    timestamp: str


class QuizSystem:
    """
    Система квизов для образовательной платформы
    
    Функции:
    - Генерация вопросов из материалов
    - Multiple choice тесты
    - Анализ слабых мест
    - Рекомендации для улучшения
    """
    
    def __init__(self, platform):
        """
        platform: SchoolAIPlatformV3 instance
        """
        self.platform = platform
        self.results_folder = Path("quiz_results")
        self.results_folder.mkdir(exist_ok=True)
        
        self.prompts = {
            "ru": {
                "generate": """На основе этого материала создай {num} вопросов для теста СТРОГО ПО ТЕМЕ: "{topic}" уровня {difficulty}.

Материал:
{context}

ВАЖНО:
- Ответь ТОЛЬКО валидным JSON массивом без дополнительного текста!
- ВСЕ вопросы должны быть ТОЛЬКО по теме "{topic}". Не отклоняйся от темы

Формат ответа (строго JSON):
[
  {{
    "question": "Текст вопроса по теме {topic}?",
    "options": ["Вариант 1", "Вариант 2", "Вариант 3", "Вариант 4"],
    "correct_answer": 0,
    "explanation": "Почему этот ответ правильный",
    "topic": "{topic}"
  }}
]

Требования:
- ВСЕ вопросы должны быть по теме "{topic}"
- Вопросы должны проверять знание "{topic}"
- Варианты ответов должны быть правдоподобными
- Так же старайся часто делать длины ответов одинаковыми или по крайней мере неправильные ответы длиныыми
- Правильный ответ не должен всегда быть первым
- Используй материал выше для создания точных вопросов""",
                "adaptive": """Проанализируй слабые места ученика и создай {num} целевых вопросов.

Слабые темы: {weak_topics}
Материалы: {context}

Формат ответа (строго JSON):
[
  {{
    "question": "Вопрос по слабой теме?",
    "options": ["Вариант 1", "Вариант 2", "Вариант 3", "Вариант 4"],
    "correct_answer": 0,
    "explanation": "Объяснение с упором на слабую тему",
    "topic": "Конкретная слабая тема"
  }}
]"""
            },
            "en": {
                "generate": """Based on this material, create {num} test questions STRICTLY ABOUT THE TOPIC: "{topic}" at {difficulty} level.

Material:
{context}

IMPORTANT:
- Reply ONLY with valid JSON array without additional text!
- ALL questions must be ONLY about the topic "{topic}"
- DO NOT deviate from the topic "{topic}"

Response format (strict JSON):
[
  {{
    "question": "Question text about {topic}?",
    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
    "correct_answer": 0,
    "explanation": "Why this answer is correct",
    "topic": "{topic}"
  }}
]

Requirements:
- ALL questions must be STRICTLY about the topic "{topic}"
- Questions should test knowledge of "{topic}"
- Answer options should be plausible
- Correct answer shouldn't always be first
- Use the material above to create accurate questions""",
                "adaptive": """Analyze student's weak points and create {num} targeted questions.

Weak topics: {weak_topics}
Materials: {context}

Response format (strict JSON):
[
  {{
    "question": "Question about weak topic?",
    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
    "correct_answer": 0,
    "explanation": "Explanation focusing on weak topic",
    "topic": "Specific weak topic"
  }}
]"""
            },
            "kk": {
                "generate": """Осы материалға негізделген ТАҚЫРЫП БОЙЫНША ҚАТАҢ: "{topic}" {difficulty} деңгейіндегі {num} сұрақтар жаса.

Материал:
{context}

МАҢЫЗДЫ:
- Қосымша мәтінсіз тек валидті JSON массивімен жауап бер!
- БАРЛЫҚ сұрақтар тек "{topic}" тақырыбы бойынша болуы КЕРЕК
- "{topic}" тақырыбынан ауытқымаңыз

Жауап форматы (қатаң JSON):
[
  {{
    "question": "{topic} тақырыбы бойынша сұрақ мәтіні?",
    "options": ["Нұсқа 1", "Нұсқа 2", "Нұсқа 3", "Нұсқа 4"],
    "correct_answer": 0,
    "explanation": "Неліктен бұл жауап дұрыс",
    "topic": "{topic}"
  }}
]

Талаптар:
- БАРЛЫҚ сұрақтар "{topic}" тақырыбы бойынша ҚАТАҢ болуы КЕРЕК
- Сұрақтар "{topic}" білімін тексеруі керек
- Жауап нұсқалары шынайы болуы керек
- Дұрыс жауап әрқашан бірінші болмауы керек
- Дәл сұрақтар жасау үшін жоғарыдағы материалды пайдаланыңыз"""
            }
        }
    
    def get_available_topics(self) -> List[str]:
        """Получить список доступных тем из загруженных материалов"""
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
                    "full_name": full_name,
                    "chunks": topic_data.get("chunks", 0)
                })
        
        return unique_topics
    
    def generate_quiz(self, config: QuizConfig) -> List[QuizQuestion]:
        """
        Генерация квиза по конфигурации
        
        Возвращает список вопросов
        """
        print(f"\n🎯 Генерация квиза:")
        print(f"   Режим: {config.mode}")
        print(f"   Тема: {config.topic}")
        print(f"   Вопросов: {config.num_questions}")
        print(f"   Сложность: {config.difficulty}")
        
        if config.mode == "adaptive":
            matches = self.platform.search_relevant_content(
                config.topic or "информатика",
                top_k=15
            )
        else:
            search_query = config.topic or "информатика основы"
            matches = self.platform.search_relevant_content(search_query, top_k=15)
        
        if not matches:
            raise ValueError("Не найдено материалов по этой теме")
        
        context = "\n\n".join([
            f"[{m.metadata.get('topic', 'Материал')}]\n{m.metadata.get('text', '')}"
            for m in matches[:10]
        ])
        
        prompt_template = self.prompts.get(config.language, self.prompts["ru"])["generate"]

        prompt = prompt_template.format(
            num=config.num_questions,
            difficulty=config.difficulty,
            topic=config.topic or "информатика",
            context=context
        )
        
        print("   🤖 Генерация вопросов...")
        
        response = self.platform.openai_client.chat.completions.create(
            model=self.platform.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a computer science teacher creating test questions. Always respond with valid JSON array only, no additional text."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} if hasattr(self.platform.openai_client, 'response_format') else None
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
            
            questions_data = json.loads(response_text)
            
            if isinstance(questions_data, dict):
                for key in questions_data:
                    if isinstance(questions_data[key], list):
                        questions_data = questions_data[key]
                        break
            
            if not isinstance(questions_data, list):
                raise ValueError("Ответ не является массивом")
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"Ответ AI: {response_text[:200]}...")
            raise ValueError("AI вернул невалидный JSON")
        
        questions = []
        for idx, q_data in enumerate(questions_data):
            try:
                question = QuizQuestion(
                    question=q_data["question"],
                    options=q_data["options"],
                    correct_answer=q_data["correct_answer"],
                    explanation=q_data["explanation"],
                    topic=q_data.get("topic", config.topic or "Информатика"),
                    difficulty=config.difficulty
                )
                questions.append(question)
            except Exception as e:
                print(f"   ⚠️ Ошибка в вопросе {idx+1}: {e}")
                continue
        
        print(f"   ✅ Сгенерировано {len(questions)} вопросов\n")
        
        return questions
    
    def save_result(self, result: QuizResult):
        """Сохранение результата квиза"""
        try:
            result_file = self.results_folder / f"{result.user_id}_results.json"
            
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"user_id": result.user_id, "quizzes": []}
            
            data["quizzes"].append(result.dict())
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"⚠️ Ошибка сохранения результата: {e}")
            return False
    
    def get_user_weak_topics(self, user_id: str, limit: int = 3) -> List[str]:
        """Получить слабые темы пользователя на основе истории"""
        try:
            result_file = self.results_folder / f"{user_id}_results.json"
            
            if not result_file.exists():
                return []
            
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            topic_stats = {}
            
            for quiz in data["quizzes"][-5:]:  
                for answer in quiz.get("answers", []):
                    if not answer.get("is_correct", False):
                        topic = answer.get("topic", "unknown")
                        topic_stats[topic] = topic_stats.get(topic, 0) + 1
            
            weak_topics = sorted(topic_stats.items(), key=lambda x: x[1], reverse=True)
            
            return [topic for topic, _ in weak_topics[:limit]]
        
        except Exception as e:
            print(f"⚠️ Ошибка анализа слабых мест: {e}")
            return []
    
    def get_recommendations(self, weak_topics: List[str], language: str = "ru") -> List[str]:
        """Получить рекомендации для улучшения"""
        recommendations = {
            "ru": [
                f"📚 Изучи материалы по теме: {topic}"
                for topic in weak_topics
            ],
            "en": [
                f"📚 Study materials on topic: {topic}"
                for topic in weak_topics
            ],
            "kk": [
                f"📚 Тақырып бойынша материалдарды оқы: {topic}"
                for topic in weak_topics
            ]
        }
        
        return recommendations.get(language, recommendations["ru"])
    
    def calculate_score(self, answers: List[Dict]) -> Dict:
        """Подсчёт результатов"""
        total = len(answers)
        correct = sum(1 for a in answers if a.get("is_correct", False))
        wrong = total - correct
        percentage = (correct / total * 100) if total > 0 else 0
        
        return {
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "percentage": round(percentage, 2)
        }