import pypyodbc
import sys

#connection = pyodbc.connect(
#        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-N5MFSCA\CITADEL;DATABASE=Test;Trusted_Connection=yes;')

#connection = pyodbc.connect('Server=tcp:s8server.database.windows.net,1433;Initial Catalog=BDD_S8;'
#                            'Persist Security Info=False;User ID=marcel.songo@groupe-esigelec.org@s8server;'
#                            'Password=17G2432Qq4LsS8R;MultipleActiveResultSets=False;Encrypt=True;'
#                            'TrustServerCertificate=False;Connection Timeout=30;')

server = 's8server.database.windows.net'
database = 'BDD_S8'
username = 'marcel.songo@groupe-esigelec.org@s8server'
password = '17G2432Qq4LsS8R'
driver= '{ODBC Driver 17 for SQL Server}'

with pypyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password) as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT TOP 3 name, collation_name FROM sys.databases")
        row = cursor.fetchone()
        while row:
            print (str(row[0]) + " " + str(row[1]))
            row = cursor.fetchone()


#cursor.execute(f"INSERT INTO test(NUMBER) VALUES ({sys.argv[1]})")

#connection.commit()

#while 1:
#    row = cursor.fetchone()
#    if not row:
#        break
#    print(row)



