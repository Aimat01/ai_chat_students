"""
Модуль для синхронизации истории чатов с Google Drive
Требует установки: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
"""

import os
import json
from pathlib import Path
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import pickle
import io


class GoogleDriveSync:
    """Класс для синхронизации истории чатов с Google Drive"""
    
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    def __init__(self, credentials_file='credentials.json'):
        """
        Инициализация синхронизации с Google Drive
        
        Args:
            credentials_file: путь к файлу с credentials от Google Cloud Console
        """
        self.credentials_file = credentials_file
        self.token_file = 'token.pickle'
        self.service = None
        self.folder_id = None
        
    def authenticate(self):
        """Аутентификация в Google Drive"""
        creds = None
        
        # Загружаем существующий токен
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # Если токена нет или он невалиден
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    print(f"❌ Файл {self.credentials_file} не найден!")
                    print("Создайте проект в Google Cloud Console и скачайте credentials.json")
                    return False
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Сохраняем токен
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('drive', 'v3', credentials=creds)
        print("✅ Подключено к Google Drive")
        return True
    
    def get_or_create_folder(self, folder_name='SchoolAI_ChatHistory'):
        """Получить или создать папку для истории"""
        try:
            # Ищем существующую папку
            results = self.service.files().list(
                q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            items = results.get('files', [])
            
            if items:
                self.folder_id = items[0]['id']
                print(f"📁 Найдена папка: {folder_name}")
            else:
                # Создаём новую папку
                file_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = self.service.files().create(
                    body=file_metadata,
                    fields='id'
                ).execute()
                self.folder_id = folder.get('id')
                print(f"📁 Создана папка: {folder_name}")
            
            return self.folder_id
        except Exception as e:
            print(f"❌ Ошибка работы с папкой: {e}")
            return None
    
    def upload_file(self, local_path, user_id):
        """
        Загрузить файл истории в Google Drive
        
        Args:
            local_path: путь к локальному файлу
            user_id: ID пользователя
        """
        try:
            if not self.folder_id:
                self.get_or_create_folder()
            
            filename = f"user_{user_id}_history.json"
            
            # Проверяем существует ли файл
            results = self.service.files().list(
                q=f"name='{filename}' and '{self.folder_id}' in parents and trashed=false",
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            items = results.get('files', [])
            
            file_metadata = {
                'name': filename,
                'parents': [self.folder_id]
            }
            media = MediaFileUpload(local_path, mimetype='application/json', resumable=True)
            
            if items:
                # Обновляем существующий файл
                file = self.service.files().update(
                    fileId=items[0]['id'],
                    media_body=media
                ).execute()
                print(f"☁️ История обновлена в Google Drive: {filename}")
            else:
                # Создаём новый файл
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"☁️ История загружена в Google Drive: {filename}")
            
            return file.get('id')
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return None
    
    def download_file(self, user_id, local_path):
        """
        Скачать файл истории из Google Drive
        
        Args:
            user_id: ID пользователя
            local_path: путь для сохранения
        """
        try:
            if not self.folder_id:
                self.get_or_create_folder()
            
            filename = f"user_{user_id}_history.json"
            
            # Ищем файл
            results = self.service.files().list(
                q=f"name='{filename}' and '{self.folder_id}' in parents and trashed=false",
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            items = results.get('files', [])
            
            if not items:
                print(f"📭 История не найдена в Google Drive: {filename}")
                return False
            
            # Скачиваем файл
            request = self.service.files().get_media(fileId=items[0]['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            # Сохраняем локально
            with open(local_path, 'wb') as f:
                f.write(fh.getvalue())
            
            print(f"☁️ История загружена из Google Drive: {filename}")
            return True
        except Exception as e:
            print(f"❌ Ошибка скачивания: {e}")
            return False
    
    def sync_history(self, local_history_folder, user_id):
        """
        Синхронизировать историю между локальным хранилищем и Google Drive
        
        Args:
            local_history_folder: папка с локальной историей
            user_id: ID пользователя
        """
        local_file = Path(local_history_folder) / f"user_{user_id}.json"
        
        try:
            # Если есть локальный файл, загружаем в Drive
            if local_file.exists():
                self.upload_file(str(local_file), user_id)
            else:
                # Если локального нет, пытаемся скачать из Drive
                if self.download_file(user_id, str(local_file)):
                    print("✅ История восстановлена из облака")
                else:
                    print("📝 Начинаем новую историю")
            
            return True
        except Exception as e:
            print(f"❌ Ошибка синхронизации: {e}")
            return False
    
    def list_all_histories(self):
        """Показать все файлы истории в Google Drive"""
        try:
            if not self.folder_id:
                self.get_or_create_folder()
            
            results = self.service.files().list(
                q=f"'{self.folder_id}' in parents and trashed=false",
                spaces='drive',
                fields='files(id, name, modifiedTime)'
            ).execute()
            
            items = results.get('files', [])
            
            if not items:
                print("📭 Нет сохранённых историй")
            else:
                print("\n☁️ Истории в Google Drive:")
                print("="*60)
                for item in items:
                    modified = datetime.fromisoformat(item['modifiedTime'].replace('Z', '+00:00'))
                    print(f"  • {item['name']} (изменён: {modified.strftime('%Y-%m-%d %H:%M')})")
                print("="*60 + "\n")
            
            return items
        except Exception as e:
            print(f"❌ Ошибка получения списка: {e}")
            return []


# Пример использования
if __name__ == "__main__":
    print("="*60)
    print("ТЕСТ СИНХРОНИЗАЦИИ С GOOGLE DRIVE")
    print("="*60 + "\n")
    
    sync = GoogleDriveSync()
    
    if sync.authenticate():
        sync.get_or_create_folder()
        sync.list_all_histories()
        
        # Пример синхронизации
        # sync.sync_history('./chat_history', 'test_user')