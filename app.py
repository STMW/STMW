# -*- coding: utf-8 -*-
from flask import Flask, jsonify
#import pyodbc
import subprocess

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

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

    p = subprocess.run(['python', 'script.py', '10'])
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
    return jsonify(dictionnaire)

if __name__ == "__main__":
    #app.run(debug=True, host = "localhost", port=8050)
    app.run()
