"""
AI Платформа для школьников v3.0
Новые возможности:
- Контекст диалога (помнит предыдущие сообщения)
- Мультиязычность (English, Русский, Қазақша)
- Поддержка EPUB книг
"""

import os
from pathlib import Path
from openai import OpenAI
import tiktoken
import time
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from docx import Document
import PyPDF2

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EPUB_SUPPORT = True
except ImportError:
    EPUB_SUPPORT = False
    print("there is no EPUB. pip install ebooklib beautifulsoup4")

load_dotenv()


LANGUAGES = {
    "en": {
        "name": "English",
        "welcome": "🎓 AI Platform for Students",
        "initializing": "🎓 Initializing educational platform...",
        "ready": "✅ Platform ready!",
        "menu": {
            "1": "Load study materials",
            "2": "Start learning (chat with AI)",
            "3": "Show available subjects",
            "4": "View knowledge base statistics",
            "0": "Exit"
        },
        "prompts": {
            "choice": "Choose action: ",
            "user_id": "Your ID or name: ",
            "folder": "Materials folder (default './materials'): ",
            "question": "📝 You: ",
        },
        "commands": {
            "summary": "summary",
            "history": "history",
            "exit": "exit"
        },
        "system_prompt": "You are a friendly AI tutor for students. Explain complex topics in simple language. Remember the conversation context and refer to previous messages when needed.",
        "messages": {
            "processing": "🔄 Processing materials...",
            "thinking": "🤔 Thinking...",
            "creating_summary": "📚 Creating summary...",
            "empty_db": "❌ Knowledge base is empty! Load study materials first.",
            "goodbye": "👋 Good luck with your studies!"
        }
    },
    "ru": {
        "name": "Русский",
        "welcome": "🎓 AI Платформа для Школьников",
        "initializing": "🎓 Инициализация образовательной платформы...",
        "ready": "✅ Платформа готова!",
        "menu": {
            "1": "Загрузить учебные материалы",
            "2": "Начать учиться (чат с AI)",
            "3": "Показать доступные предметы",
            "4": "Посмотреть статистику базы знаний",
            "0": "Выход"
        },
        "prompts": {
            "choice": "Выбери действие: ",
            "user_id": "Твой ID или имя: ",
            "folder": "Папка с материалами (по умолчанию './materials'): ",
            "question": "📝 Ты: ",
        },
        "commands": {
            "summary": "конспект",
            "history": "история",
            "exit": "выход"
        },
        "system_prompt": "Ты дружелюбный AI-репетитор для школьников. Объясняй сложные вещи простым языком. Помни контекст разговора и ссылайся на предыдущие сообщения когда нужно.",
        "messages": {
            "processing": "🔄 Обрабатываю материалы...",
            "thinking": "🤔 Думаю...",
            "creating_summary": "📚 Создаю конспект...",
            "empty_db": "❌ База знаний пуста! Сначала загрузи учебные материалы.",
            "goodbye": "👋 Удачи с учёбой!"
        }
    },
    "kk": {
        "name": "Қазақша",
        "welcome": "🎓 Оқушыларға арналған AI платформасы",
        "initializing": "🎓 Білім беру платформасын іске қосу...",
        "ready": "✅ Платформа дайын!",
        "menu": {
            "1": "Оқу материалдарын жүктеу",
            "2": "Оқуды бастау (AI-мен сөйлесу)",
            "3": "Қолжетімді пәндерді көрсету",
            "4": "Білім базасының статистикасын көру",
            "0": "Шығу"
        },
        "prompts": {
            "choice": "Әрекетті таңда: ",
            "user_id": "Сенің ID немесе атың: ",
            "folder": "Материалдар қалтасы (әдепкі './materials'): ",
            "question": "📝 Сен: ",
        },
        "commands": {
            "summary": "конспект",
            "history": "тарих",
            "exit": "шығу"
        },
        "system_prompt": "Сіз оқушыларға арналған достық AI-репетиторсыз. Күрделі тақырыптарды қарапайым тілмен түсіндіріңіз. Әңгіме контекстін есте сақтап, қажет болса алдыңғы хабарламаларға сілтеме жасаңыз.",
        "messages": {
            "processing": "🔄 Материалдарды өңдеу...",
            "thinking": "🤔 Ойланып жатырмын...",
            "creating_summary": "📚 Конспект жасау...",
            "empty_db": "❌ Білім базасы бос! Алдымен оқу материалдарын жүктеңіз.",
            "goodbye": "👋 Оқуға сәттілік!"
        }
    }
}


class SchoolAIPlatformV3:
    """
    Улучшенная AI-платформа для школьников v3.0
    
    Новое:
    - Контекст диалога
    - Мультиязычность
    - Поддержка EPUB
    """
    
    def __init__(self, openai_api_key, pinecone_api_key, language="ru", index_name="school-topics"):
        self.lang = language
        self.t = LANGUAGES[language]
        
        print(self.t["initializing"])
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-5-mini"
        
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        
        self.topics_list_file = "school_topics.json"
        self.chat_history_folder = Path("chat_history")
        self.chat_history_folder.mkdir(exist_ok=True)
        
        try:
            existing = [idx.name for idx in self.pc.list_indexes()]
        except:
            existing = []
            
        if index_name not in existing:
            print(f"📚 Creating knowledge base: {index_name}")
            try:
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
                )
                time.sleep(20)
            except Exception as e:
                print(f"⚠️ Index creation error: {e}")
        
        self.index = self.pc.Index(index_name)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        self.subjects = {
            "математика": ["алгебра", "геометрия", "тригонометрия"],
            "физика": ["механика", "термодинамика", "электричество"],
            "химия": ["органическая химия", "неорганическая химия"],
            "биология": ["клетка", "генетика", "эволюция"],
            "история": ["древний мир", "средние века"],
            "литература": ["русская литература", "мировая литература"],
        }
        
        print(self.t["ready"] + "\n")
    
    def save_topics_list(self, topics):
        """Сохранение списка обработанных топиков"""
        try:
            with open(self.topics_list_file, 'w', encoding='utf-8') as f:
                json.dump(topics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save list: {e}")
    
    def load_topics_list(self):
        """Загрузка списка топиков"""
        try:
            if Path(self.topics_list_file).exists():
                with open(self.topics_list_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def save_chat_history(self, user_id, messages):
        """Сохранение истории чата"""
        try:
            history_file = self.chat_history_folder / f"user_{user_id}.json"
            
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = {"user_id": user_id, "sessions": []}
            
            session = {
                "timestamp": datetime.now().isoformat(),
                "language": self.lang,
                "messages": messages
            }
            history["sessions"].append(session)
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"⚠️ History save error: {e}")
            return False
    
    def load_chat_history(self, user_id):
        """Загрузка истории чата"""
        try:
            history_file = self.chat_history_folder / f"user_{user_id}.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ History load error: {e}")
        return None
    
    def read_epub(self, path):
        """Чтение EPUB файлов"""
        if not EPUB_SUPPORT:
            print(f"⚠️ EPUB not supported. Install: pip install ebooklib beautifulsoup4")
            return ""
        
        try:
            book = epub.read_epub(path)
            text = []
            
            metadata = {
                'title': book.get_metadata('DC', 'title'),
                'creator': book.get_metadata('DC', 'creator'),
            }
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    try:
                        content = item.get_content()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        text_content = soup.get_text(separator='\n', strip=True)
                        
                        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                        
                        if lines:
                            text.append('\n'.join(lines))
                    except Exception as e:
                        print(f"    ⚠️ Error reading chapter: {e}")
                        continue
            
            result = '\n\n'.join(text)
            print(f"    📊 Extracted {len(result)} characters from EPUB")
            
            return result
        except Exception as e:
            print(f"⚠️ EPUB read error: {e}")
            return ""
    
    def read_file(self, path):
        """Чтение файлов разных форматов"""
        ext = Path(path).suffix.lower()
        
        if ext == '.txt':
            return self._read_txt(path)
        elif ext == '.docx':
            return self._read_docx(path)
        elif ext == '.pdf':
            return self._read_pdf(path)
        elif ext == '.epub':
            return self.read_epub(path)
        return ""
    
    def _read_txt(self, path):
        """Чтение TXT файлов"""
        for enc in ['utf-8', 'cp1251', 'windows-1251']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    return f.read()
            except:
                continue
        return ""
    
    def _read_docx(self, path):
        """Чтение DOCX файлов"""
        try:
            doc = Document(path)
            text = []
            for p in doc.paragraphs:
                if p.text.strip():
                    text.append(p.text)
            return '\n'.join(text)
        except Exception as e:
            print(f"⚠️ DOCX read error: {e}")
            return ""
    
    def _read_pdf(self, path):
        """Чтение PDF файлов"""
        try:
            text = []
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text.append(page_text)
            return '\n'.join(text)
        except Exception as e:
            print(f"⚠️ PDF read error: {e}")
            return ""
    
    def chunk_text(self, text, size=600, overlap=100):
        """Разбиение текста на чанки"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), size - overlap):
            chunk = self.tokenizer.decode(tokens[i:i + size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def create_embeddings(self, texts):
        """Создание эмбеддингов"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def extract_book_title(self, path):
        """Извлечение названия книги из EPUB метаданных"""
        if not EPUB_SUPPORT:
            return None
        
        try:
            book = epub.read_epub(path)
            
            title_meta = book.get_metadata('DC', 'title')
            if title_meta and len(title_meta) > 0:
                title = title_meta[0][0]
                if title and len(title) > 3:
                    title = title.strip()
                    title = ''.join(c for c in title if c.isalnum() or c in ' -_')
                    return title[:50]
            
            return None
        except:
            return None
    
    def process_topic(self, path, topic_id, subject, topic_name):
        """Обработка топика"""
        try:
            if Path(path).suffix.lower() == '.epub':
                smart_title = self.extract_book_title(path)
                if smart_title:
                    topic_name = smart_title
                    print(f"   📖 Book title: {smart_title}")
            
            text = self.read_file(path)
            if len(text) < 50:
                print(f"   ⚠️ Too little text extracted ({len(text)} chars)")
                return False, None
            
            print(f"   📖 Subject: {subject}")
            print(f"   📚 Topic: {topic_name}")
            
            chunks = self.chunk_text(text)
            print(f"   📄 Chunks: {len(chunks)}")
            
            if len(chunks) == 0:
                print(f"   ⚠️ No chunks created from text")
                return False, None
            
            for i in range(0, len(chunks), 50):
                batch = chunks[i:i + 50]
                embeddings = self.create_embeddings(batch)
                
                vectors = []
                for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
                    vectors.append({
                        "id": f"{topic_id}_{i+j}",
                        "values": emb,
                        "metadata": {
                            "subject": subject,
                            "topic": topic_name,
                            "full_name": f"{subject}: {topic_name}",
                            "text": chunk[:800]
                        }
                    })
                
                self.index.upsert(vectors=vectors)
                time.sleep(0.3)
            
            return True, {
                "subject": subject,
                "topic": topic_name,
                "chunks": len(chunks)
            }
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, None
    
    def process_materials_folder(self, folder):
        """Обработка папки с материалами"""
        materials_path = Path(folder)
        if not materials_path.exists():
            print(f"❌ Folder not found: {folder}")
            return
        
        files = list(materials_path.glob("**/*.txt")) + \
                list(materials_path.glob("**/*.docx")) + \
                list(materials_path.glob("**/*.pdf")) + \
                list(materials_path.glob("**/*.epub"))
        
        if not files:
            print(f"❌ No files in: {folder}")
            return
        
        print(f"📚 Found materials: {len(files)}\n")
        
        processed = []
        success = 0
        
        for idx, f in enumerate(files, 1):
            parts = f.parts
            subject = "general"
            if len(parts) > 1:
                subject = parts[-2] if parts[-2] != folder else "general"
            
            topic_name = f.stem
            
            print(f"[{idx}/{len(files)}] {f.name}")
            result, meta = self.process_topic(str(f), f"topic{idx:03d}", subject, topic_name)
            
            if result and meta:
                success += 1
                processed.append({
                    "id": f"topic{idx:03d}",
                    "filename": f.name,
                    "subject": meta["subject"],
                    "topic": meta["topic"],
                    "chunks": meta["chunks"]
                })
                print(f"   ✅ Done\n")
            else:
                print(f"   ⚠️ Error\n")
        
        self.save_topics_list(processed)
        print(f"\n✅ Processed: {success}/{len(files)}")
    
    def search_relevant_content(self, query, top_k=5):
        """Поиск релевантного контента"""
        emb = self.create_embeddings([query])[0]
        results = self.index.query(vector=emb, top_k=top_k, include_metadata=True)
        return results.matches
    
    def generate_response_with_context(self, question, matches, conversation_history):
        """
        Генерация ответа с учётом контекста диалога
        КЛЮЧЕВОЕ УЛУЧШЕНИЕ: AI помнит предыдущие сообщения
        """
        if not matches:
            no_info = {
                "en": "I couldn't find information on this topic. Try rephrasing your question.",
                "ru": "Не нашёл информации по этой теме. Попробуй переформулировать вопрос.",
                "kk": "Бұл тақырып бойынша ақпарат таппадым. Сұрағыңызды басқаша тұжырымдаңыз."
            }
            return no_info.get(self.lang, no_info["en"])
        
        context = "\n\n".join([
            f"[{m.metadata.get('full_name', 'Material')}]\n{m.metadata.get('text', '')}"
            for m in matches
        ])
        
        enhanced_system_prompt = f"""{self.t["system_prompt"]}

КРИТИЧЕСКИ ВАЖНО: 
- Внимательно читай ВСЮ историю разговора перед ответом
- Если вопрос ссылается на предыдущую тему (например: "а это что?", "объясни попроще", "не понял"), ОБЯЗАТЕЛЬНО продолжай ТУ ЖЕ тему
- НЕ переключайся на другую тему, если вопрос - это продолжение предыдущего
- Когда ученик говорит "не понял" или "попроще", объясняй ТУ ЖЕ самую тему проще, а не новую тему"""
        
        messages = [
            {"role": "system", "content": enhanced_system_prompt}
        ]
        
        for msg in conversation_history[-10:]:  
            messages.append(msg)
        
        if len(conversation_history) > 0:
            prompt = f"""Study materials:
{context}

ВАЖНО: Это продолжение нашего разговора. Смотри историю выше!

Student's current question: {question}

Provide a clear explanation. If this question refers to previous messages (like "explain simpler", "I don't understand"), continue explaining THE SAME topic, not a new one."""
        else:
            prompt = f"""Study materials:
{context}

Student's question: {question}

Provide a clear and detailed explanation. Use examples if needed."""
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
        )
        
        return response.choices[0].message.content
    
    def generate_summary(self, topic, matches):
        """Генерация конспекта"""
        if not matches:
            return "Could not find materials for summary."
        
        context = "\n\n".join([m.metadata.get('text', '') for m in matches])
        
        prompt = f"""Create a detailed summary on the topic: {topic}

Materials:
{context}

Summary structure:
1. Key concepts
2. Important formulas/rules (if any)
3. Examples
4. Important points to remember

Write concisely but informatively."""
        
        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You create summaries for students. Structure information clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        return response.choices[0].message.content
    
    def chat_session(self, user_id):
        """Интерактивная сессия с контекстом диалога"""
        print(f"\n👤 User: {user_id}")
        print("="*60)
        
        stats = self.index.describe_index_stats()
        if stats.total_vector_count == 0:
            print(f"\n{self.t['messages']['empty_db']}\n")
            return
        
        print(f"✅ Available materials: {stats.total_vector_count} chunks")
        print(f"\n💬 I'm ready to help with studying!")
        print(f"Commands:")
        print(f"  • '{self.t['commands']['summary']} <topic>' - get summary")
        print(f"  • '{self.t['commands']['history']}' - view history")
        print(f"  • '{self.t['commands']['exit']}' - finish\n")
        
        conversation_history = []
        
        session_messages = []
        
        while True:
            question = input(self.t["prompts"]["question"]).strip()
            
            if not question:
                continue
            
            if question.lower() in [self.t['commands']['exit'], 'exit', 'quit']:
                if session_messages:
                    saved = self.save_chat_history(user_id, session_messages)
                    if saved:
                        print(f"💾 History saved to chat_history/user_{user_id}.json")
                print(f"\n{self.t['messages']['goodbye']}")
                break
            
            if question.lower() == self.t['commands']['history']:
                self.show_history(user_id)
                continue
            
            if question.lower().startswith(self.t['commands']['summary']):
                topic = question[len(self.t['commands']['summary']):].strip()
                if not topic:
                    print(f"🤖: Specify topic. Example: '{self.t['commands']['summary']} derivatives'\n")
                    continue
                
                print(f"\n{self.t['messages']['creating_summary']}\n")
                matches = self.search_relevant_content(topic, top_k=10)
                answer = self.generate_summary(topic, matches)
            else:
                print(f"\n{self.t['messages']['thinking']}\n")
                matches = self.search_relevant_content(question, top_k=5)
                
                answer = self.generate_response_with_context(question, matches, conversation_history)
            
            print(f"🤖: {answer}\n")
            print("-"*60 + "\n")
            
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})
            
            session_messages.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer
            })
            
            if len(session_messages) % 5 == 0:
                self.save_chat_history(user_id, session_messages)
                print(f"💾 Auto-saved ({len(session_messages)} messages)")

    
    def show_history(self, user_id):
        """Показать историю чата"""
        history = self.load_chat_history(user_id)
        if not history or not history.get("sessions"):
            print("\n📭 History is empty\n")
            return
        
        print("\n📜 Chat history:")
        print("="*60)
        
        for idx, session in enumerate(history["sessions"][-5:], 1):
            timestamp = datetime.fromisoformat(session["timestamp"])
            lang = session.get("language", "ru")
            print(f"\nSession {idx} ({timestamp.strftime('%Y-%m-%d %H:%M')}) [{lang}]")
            print(f"Questions: {len(session['messages'])}")
            
            if session["messages"]:
                print("\nLast question:")
                last = session["messages"][-1]
                print(f"  Q: {last['question'][:60]}...")
        
        print("\n" + "="*60 + "\n")
    
    def show_subjects(self):
        """Показать доступные предметы"""
        print("\n📚 Available subjects:")
        print("="*60)
        for subject, topics in self.subjects.items():
            print(f"\n{subject.upper()}")
            print(f"  Topics: {', '.join(topics)}")
        print("\n" + "="*60 + "\n")


def select_language():
    """Выбор языка интерфейса"""
    print("="*60)
    print("SELECT LANGUAGE / ВЫБЕРИТЕ ЯЗЫК / ТІЛДІ ТАҢДАҢЫЗ")
    print("="*60)
    print("1. English")
    print("2. Русский")
    print("3. Қазақша")
    print("="*60)
    
    while True:
        choice = input("Choose (1-3): ").strip()
        if choice == "1":
            return "en"
        elif choice == "2":
            return "ru"
        elif choice == "3":
            return "kk"
        else:
            print("❌ Invalid choice. Try again.")


def main():
    lang = select_language()
    t = LANGUAGES[lang]
    
    print("\n" + "="*60)
    print(t["welcome"])
    print("="*60 + "\n")
    
    load_dotenv()
    
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_KEY = os.getenv("PINECONE_API_KEY")
    
    if not OPENAI_KEY or not PINECONE_KEY:
        print("❌ Set API keys in .env file!")
        print("   OPENAI_API_KEY=your_key")
        print("   PINECONE_API_KEY=your_key")
        return
    
    platform = SchoolAIPlatformV3(OPENAI_KEY, PINECONE_KEY, language=lang)
    
    print(f"\n📚 {t['menu'].get('title', 'Main menu')}:")
    print("="*60)
    for key, value in t["menu"].items():
        print(f"{key} - {value}")
    print("="*60 + "\n")
    
    while True:
        choice = input(t["prompts"]["choice"]).strip()
        
        if choice == "1":
            folder = input(f"\n{t['prompts']['folder']}").strip()
            if not folder:
                folder = "./materials"
            
            if not Path(folder).exists():
                print(f"\n❌ Folder '{folder}' not found!\n")
                continue
            
            print(f"\n{t['messages']['processing']}\n")
            platform.process_materials_folder(folder)
            print("\n✅ Done!\n")
        
        elif choice == "2":
            user_id = input(f"\n{t['prompts']['user_id']}").strip()
            if not user_id:
                user_id = "student"
            
            platform.chat_session(user_id)
            print()
        
        elif choice == "3":
            platform.show_subjects()
        
        elif choice == "4":
            stats = platform.index.describe_index_stats()
            topics = platform.load_topics_list()
            
            print("\n📊 Knowledge base statistics:")
            print("="*60)
            print(f"Total text chunks: {stats.total_vector_count}")
            print(f"Loaded topics: {len(topics)}")
            
            if topics:
                print("\n📚 Last loaded materials:")
                for topic in topics[-5:]:
                    print(f"  • {topic['subject']}: {topic['topic']} ({topic['chunks']} chunks)")
            print("="*60 + "\n")
        
        elif choice == "0":
            print(f"\n{t['messages']['goodbye']}")
            break
        
        else:
            print("❌ Invalid choice\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()