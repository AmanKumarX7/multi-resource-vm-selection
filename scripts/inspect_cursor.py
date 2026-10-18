import mysql.connector.cursor

with open("scratch_output.txt", "w") as f:
    f.write("\n".join(dir(mysql.connector.cursor.MySQLCursor)))
