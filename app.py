# -*- coding: utf-8 -*-
from flask import Flask, jsonify
import pypyodbc
import subprocess

app = Flask(__name__)

@app.route("/")
def hello():
    server = 's8server.database.windows.net'
    database = 'BDD_S8'
    username = 'marcel.songo@groupe-esigelec.org@s8server'
    password = '17G2432Qq4LsS8R'
    driver = '{ODBC Driver 17 for SQL Server}'

    connection = pypyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = connection.cursor()
    
    cursor.execute("SELECT TOP 3 name, collation_name FROM sys.databases")
    row = cursor.fetchone()
    
    return str(row[0]) + " " + str(row[1])

@app.route("/api/test")
def test():
    '''
    # Trusted Connection to Named Instance
    connection = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-N5MFSCA\CITADEL;DATABASE=Test;Trusted_Connection=yes;')

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM test")

    while 1:
        row = cursor.fetchone()
        if not row:
            break
        print(row)

    cursor.close()
    connection.close()
    '''

    p = subprocess.run(args = ['python', './script.py'], universal_newlines = True, stdout = subprocess.PIPE)
    #print('return code: ', p.returncode)

    # Trusted Connection to Named Instance
    #connection = pyodbc.connect(
        #'DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-N5MFSCA\CITADEL;DATABASE=Test;Trusted_Connection=yes;')

    #cursor = connection.cursor()
    #cursor.execute("SELECT * FROM test")

    #while 1:
        #row = cursor.fetchone()
        #if not row:
            #break
        #print(row)

    #cursor.close()
    #connection.close()

    dictionnaire = {
        'type': 'Prévision de température',
        'valeurs': [24, 24, 25, 26, 27, 28],
        'unite': "degrés Celcius",
        'return_code': p.returncode
    }
    return jsonify(p.stdout.splitlines())

if __name__ == "__main__":
    #app.run(debug=True, host = "localhost", port=8050)
    app.run()
