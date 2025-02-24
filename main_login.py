from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import mysql.connector
from mysql.connector import Error
import bcrypt
from datetime import datetime

app = FastAPI()

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영환경에서는 구체적인 origin을 지정해야 합니다
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure MySQL 연결 설정
DB_CONFIG = {
    "host": "personalchill-server.mysql.database.azure.com",
    "user": "personalchill",
    "password": "mlproject1!",
    "database": "colorchill",
    "ssl_ca": "~/DigiCertGlobalRootCA.pem"  # Azure MySQL SSL 인증서 경로
}

# 데이터 모델 정의
class UserLogin(BaseModel):
    username: str
    password: str

class UserRegistration(BaseModel):
    username: str
    password: str
    password_confirm: str
    name: str
    phone: str
    birth_date: str

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# 테이블 생성 함수
def create_tables():
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # users 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                name VARCHAR(100) NOT NULL,
                phone VARCHAR(20) NOT NULL,
                birth_date DATE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
    except Error as e:
        print(f"Error creating tables: {str(e)}")
    finally:
        cursor.close()
        connection.close()

# 앱 시작 시 테이블 생성
@app.on_event("startup")
async def startup_event():
    create_tables()

# 로그인 엔드포인트
@app.post("/login")
async def login(user_data: UserLogin):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # 사용자 검색
        cursor.execute(
            "SELECT * FROM users WHERE username = %s",
            (user_data.username,)
        )
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # 비밀번호 검증
        if not bcrypt.checkpw(user_data.password.encode('utf-8'), user['password'].encode('utf-8')):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        return {"message": "Login successful", "user_id": user['id']}
        
    finally:
        cursor.close()
        connection.close()

# 회원가입 엔드포인트
@app.post("/register")
async def register(user_data: UserRegistration):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # 비밀번호 확인
        if user_data.password != user_data.password_confirm:
            raise HTTPException(status_code=400, detail="Passwords do not match")
        
        # 사용자명 중복 검사
        cursor.execute("SELECT username FROM users WHERE username = %s", (user_data.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # 비밀번호 해싱
        hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt())
        
        # 사용자 정보 저장
        cursor.execute("""
            INSERT INTO users (username, password, name, phone, birth_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            user_data.username,
            hashed_password.decode('utf-8'),
            user_data.name,
            user_data.phone,
            user_data.birth_date
        ))
        
        connection.commit()
        return {"message": "Registration successful"}
        
    except Error as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        connection.close()

# 서버 상태 확인 엔드포인트
@app.get("/")
async def read_root():
    return {"status": "Server is running"}