import pandas as pd
import pymysql

# CSV 파일 로드
df = pd.read_csv('/clothimage_fin_final.csv')

# MySQL 연결
conn = pymysql.connect(host='your_host', user='your_user', password='your_password', database='your_db')
cursor = conn.cursor()

# 데이터 삽입
for _, row in df.iterrows():
    sql = """
    INSERT INTO newclothes (ID, brand, itemName, productLink, imageFileName, imageLinkBlob, colorHex, personalCol, category, price)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, tuple(row))

conn.commit()
cursor.close()
conn.close()
