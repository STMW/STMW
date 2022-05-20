# -*- coding: utf-8 -*-
from flask import Flask, jsonify
import pypyodbc
import subprocess
import ast
#localhost:8050/api/regression/l2/0.0001/1/True/None/lbfgs/100/None/False/None
#localhost:8050/api/arbre/gini/best/None/2/1/0/None/None/None/0/0
#localhost:8050/api/svm/1/rbf/3/scale/False/0.001/-1/None
#localhost:8050/api/randomForest/100/gini/None/2/1/0/None/None/None/True/0/None
app = Flask(__name__)


@app.route("/")
def hello():
    server = 's8server.database.windows.net'
    database = 'BDD_S8'
    username = 'marcel.songo@groupe-esigelec.org@s8server'
    password = '17G2432Qq4LsS8R'
    driver = '{ODBC Driver 17 for SQL Server}'

    connection = pypyodbc.connect(
        'DRIVER=' + driver + ';SERVER=tcp:' + server + ';PORT=1433;DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
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
    # path = os.path.join("")
    p = subprocess.run(args=['python', './script.py'], universal_newlines=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    # print('return code: ', p.returncode)

    # Trusted Connection to Named Instance
    # connection = pyodbc.connect(
    # 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-N5MFSCA\CITADEL;DATABASE=Test;Trusted_Connection=yes;')

    # cursor = connection.cursor()
    # cursor.execute("SELECT * FROM test")

    # while 1:
    # row = cursor.fetchone()
    # if not row:
    # break
    # print(row)

    # cursor.close()
    # connection.close()

    dictionnaire = {
        'type': 'Prévision de température',
        'valeurs': [24, 24, 25, 26, 27, 28],
        'unite': "degrés Celcius",
        'return_code': p.returncode
    }
    return jsonify(p.stdout.splitlines())


@app.route("/api/analytique/")
def methode_analytique():
    p = subprocess.run(args=['python', '-Wignore', './analytique.py'],
                       universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return jsonify(p.stdout.splitlines())


@app.route('/api/arbre/<a>/<b>/<c>/<d>/<e>/<f>/<g>/<h>/<i>/<j>/<k>/')
def arbre_de_decision(a,b,c,d,e,f,g,h,i,j,k):
    p = subprocess.Popen(f'python ./arbre.py {a} {b} {c} {d} {e} {f} {g} {h} {i} {j} {k} ', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode("utf-8")
    out = out.replace("\'", "\"")
    return ast.literal_eval(out)


@app.route('/api/randomForest/<a>/<b>/<c>/<d>/<e>/<f>/<g>/<h>/<i>/<j>/<k>/<l>/')
def random_forest(a,b,c,d,e,f,g,h,i,j,k,l):
    p = subprocess.Popen(f'python ./randomForest.py {a} {b} {c} {d} {e} {f} {g} {h} {i} {j} {k} {l} ', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode("utf-8")
    out = out.replace("\'", "\"")
    return ast.literal_eval(out)


@app.route('/api/regression/<a>/<b>/<c>/<d>/<e>/<f>/<g>/<h>/<i>/<j>')
def regression(a,b,c,d,e,f,g,h,i,j):
    p = subprocess.Popen(f'python ./regression.py {a} {b} {c} {d} {e} {f} {g} {h} {i} {j}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode("utf-8")
    out = out.replace("\'", "\"")
    return ast.literal_eval(out)


@app.route('/api/svm/<a>/<b>/<c>/<d>/<e>/<f>/<g>/<h>')
def svm(a,b,c,d,e,f,g,h):
    p = subprocess.Popen(f'python ./svm.py {a} {b} {c} {d} {e} {f} {g} {h} ', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode("utf-8")
    out = out.replace("\'", "\"")
    return ast.literal_eval(out)


if __name__ == "__main__":
    #app.run(debug=True, host = "localhost", port=8050)
    app.run()
