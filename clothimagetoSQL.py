import os
import pandas as pd
import pymysql

# Azure MySQL 연결 설정
DB_CONFIG = {
    "host": "personalchill-server.mysql.database.azure.com",
    "user": "personalchill",
    "password": "mlproject1!",
    "database": "colorchill",
    "ssl_ca": "/home/azureuser/DigiCertGlobalRootCA.pem"  # 절대경로 사용
}

# CSV 파일 경로 변환 (절대경로로 변경)
csv_file_path = os.path.expanduser("~/backend/clothimage_fin_final.csv")

# CSV 파일 불러오기
df = pd.read_csv(csv_file_path, encoding='utf-8')

# NaN 값을 None으로 변환 (SQL에서 NULL로 인식)
df = df.where(pd.notna(df), None)

# MySQL 연결
try:
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 데이터 삽입 쿼리 실행
    for _, row in df.iterrows():
        sql = """
        INSERT INTO newclothes 
        (ID, brand, itemName, productLink, imageFileName, imageLinkBlob, colorHex, personalCol, category, price)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            cursor.execute(sql, tuple(row))
        except Exception as e:
            print(f"❌ 오류 발생 (ID: {row['ID']}): {e}")

    # 변경 사항 커밋
    conn.commit()
    print("✅ 데이터 삽입 완료!")

except Exception as e:
    print(f"❌ MySQL 연결 또는 데이터 삽입 실패: {e}")

finally:
    # 연결 종료
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()
