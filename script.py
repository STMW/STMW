import pyodbc
import sys

connection = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-N5MFSCA\CITADEL;DATABASE=Test;Trusted_Connection=yes;')


cursor = connection.cursor()
cursor.execute(f"INSERT INTO test(NUMBER) VALUES ({sys.argv[1]})")

connection.commit()

#while 1:
#    row = cursor.fetchone()
#    if not row:
#        break
#    print(row)

cursor.close()
connection.close()


