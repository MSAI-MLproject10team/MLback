import pandas as pd
import pymysql

# Azure MySQL 연결 설정
DB_CONFIG = {
    "host": "personalchill-server.mysql.database.azure.com",
    "user": "personalchill",
    "password": "mlproject1!",
    "database": "colorchill",
    "ssl_ca": "~/DigiCertGlobalRootCA.pem"  # 절대경로 사용
}

# CSV 파일 불러오기
csv_file_path = "~/backend/clothimage_fin_final.csv"  # CSV 파일 경로 수정
df = pd.read_csv(csv_file_path, encoding='utf-8')

# MySQL 연결
conn = pymysql.connect(**DB_CONFIG)
cursor = conn.cursor()

# 데이터 삽입 쿼리 실행
for _, row in df.iterrows():
    sql = """
    INSERT INTO newclothes (ID, brand, itemName, productLink, imageFileName, imageLinkBlob, colorHex, personalCol, category, price)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, tuple(row))

# 변경 사항 커밋
conn.commit()

# 연결 종료
cursor.close()
conn.close()

print("✅ 데이터 삽입 완료!")
