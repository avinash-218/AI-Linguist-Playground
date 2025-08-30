import sqlite3

# connect to sqlite
conn = sqlite3.connect("student.db")

# create cursor for db
cursor = conn.cursor()

# create table
table_info = """
CREATE TABLE STUDENT(NAME VARCHAR(30), CLASS VARCHAR(25), SECTION VARCHAR(25), MARKS INT);
"""

cursor.execute(table_info)

# create records
cursor.execute("""INSERT INTO STUDENT VALUES('AJITH', '12', 'B', 100);""")
cursor.execute("""INSERT INTO STUDENT VALUES('AVINASH', '12', 'B', 100);""")
cursor.execute("""INSERT INTO STUDENT VALUES('PAVI', '11', 'C', 90);""")
cursor.execute("""INSERT INTO STUDENT VALUES('RAJA', '10', 'D', 80);""")
cursor.execute("""INSERT INTO STUDENT VALUES('CHANDRA', '9', 'E', 70);""")

# display records
print('The inserted records are')
data = cursor.execute("SELECT * FROM STUDENT;")

for row in data:
    print(row)

#commit and close connection
conn.commit()
conn.close()