import mergedata
import mysql.connector

con = mysql.connector.connect(
    host = "" ,
    user = "" ,
    passwd = "" ,
    database = "",
    use_pure = ""
)

def tableExists(mycursor,name):
    stmt = "SHOW TABLE LIKE '" + name + "'"
    mycursor.execute(stmt)
    return mycursor.fetchone()
try:
    mycursor = con.cursor(buffered=True)

    if tableExists(mycursor,'patient'):
        print('process table:', 'patient')
        print("-------")

        sql = "SELECT * FROM patient"

        dbTbl = DBTable(con,sql,2)
        btr = BufferTableIter(dbTbl)

        for rec in dbTbl:
            print("read record:", rec)
    finally:
        cpn.close()

mycursor = con.cursor(buffered=True)

mycursor.execute("CREATE TABLE patient (time VARCHAR(255), hr INT, hxpl INT)")

mycursor.execute("INSERT INTO patient (time, HR, HXPL) VALUES ('01:30',128,19)")
mycursor.execute("INSERT INTO patient (time, HR, HXPL) VALUES ('05:00',124,20)")
mycursor.execute("INSERT INTO patient (time, HR, HXPL) VALUES ('13:00',131,18)")
mycursor.execute("INSERT INTO patient (time, HR, HXPL) VALUES ('20:00',138,24)")
mycursor.execute("INSERT INTO patient (time, HR, HXPL) VALUES ('21:30',122,22)")

con.commit()
con.close()  

