from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import mysql.connector
from mysql.connector import Error
import bcrypt
from contextlib import asynccontextmanager
from datetime import date

# 데이터 모델 정의
class UserLogin(BaseModel):
    userID: str
    password: str

class UserRegistration(BaseModel):
    userID: str
    userName: str
    userPhoneNo: int
    password: str
    password_confirm: str
    personalColor: str = "None"
    birth_date: date  # 생년월일 필드 추가

# Azure MySQL 연결 설정
DB_CONFIG = {
    "host": "personalchill-server.mysql.database.azure.com",
    "user": "personalchill",
    "password": "mlproject1!",
    "database": "colorchill",
    "ssl_ca": "~/DigiCertGlobalRootCA.pem"
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# 테이블 스키마 수정을 위한 SQL 명령
ALTER_TABLE_SQL = """
ALTER TABLE users 
ADD COLUMN birth_date DATE;
"""

def alter_table():
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(ALTER_TABLE_SQL)
        connection.commit()
    except Error as e:
        print(f"Note: {str(e)}")  # 이미 컬럼이 존재하는 경우 무시
    finally:
        cursor.close()
        connection.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행
    print("Application startup")
    alter_table()  # 테이블 수정 실행
    yield
    # 앱 종료 시 실행
    print("Application shutdown")

app = FastAPI(lifespan=lifespan)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로그인 엔드포인트
@app.post("/login")
async def login(user_data: UserLogin):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute(
            "SELECT * FROM users WHERE userID = %s",
            (user_data.userID,)
        )
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid userID or password")
        
        if not bcrypt.checkpw(user_data.password.encode('utf-8'), user['userPasswordHash'].encode('utf-8')):
            raise HTTPException(status_code=401, detail="Invalid userID or password")
        
        return {
            "message": "Login successful",
            "userID": user['userID'],
            "userName": user['userName'],
            "personalColor": user['personalColor'],
            "birth_date": user['birth_date'].isoformat() if user['birth_date'] else None
        }
        
    finally:
        cursor.close()
        connection.close()

# 회원가입 엔드포인트
@app.post("/register")
async def register(user_data: UserRegistration):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        if user_data.password != user_data.password_confirm:
            raise HTTPException(status_code=400, detail="Passwords do not match")
        
        cursor.execute("SELECT userID FROM users WHERE userID = %s", (user_data.userID,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="UserID already exists")
        
        hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt())
        
        cursor.execute("""
            INSERT INTO users (userID, userName, userPhoneNo, userPasswordHash, personalColor, birth_date)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            user_data.userID,
            user_data.userName,
            user_data.userPhoneNo,
            hashed_password.decode('utf-8'),
            user_data.personalColor,
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