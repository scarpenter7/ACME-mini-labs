# sql1.py
"""Volume 3: SQL 1 (Introduction).
<Name> Sam Carpenter
<Class>
<Date> 3/22/21
"""
import sqlite3 as sql
import numpy as np
import csv
from matplotlib import pyplot as plt

"""
conn = sql.connect("my_database.db")
try:
    cur = conn.cursor() # Get a cursor object.
    cur.execute("SELECT * FROM MyTable") # Execute a SQL command.
except sql.Error: # If there is an error,
    conn.rollback() # revert the changes
    raise # and raise the error.
else: # If there are no errors,
    conn.commit() # save the changes.
finally:
    conn.close() 
"""

"""
try:
    with sql.connect("my_database.db") as conn:
        cur = conn.cursor() # Get the cursor.
        cur.execute("SELECT * FROM MyTable") # Execute a SQL command.
finally: # Commit or revert, then
    conn.close() # close the connection.
"""

"""
Create table
with sql.connect("my_database.db") as conn:
    cur = conn.cursor()
    cur.execute("CREATE TABLE MyTable (Name TEXT, ID INTEGER, Age REAL)")
conn.close()
"""


# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    # Problem 1
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            # Drop the tables if they exist
            cur.execute("DROP TABLE IF EXISTS MajorInfo")
            cur.execute("DROP TABLE IF EXISTS CourseInfo")
            cur.execute("DROP TABLE IF EXISTS StudentInfo")
            cur.execute("DROP TABLE IF EXISTS StudentGrades")

            # Create the following tables
            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT)")
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT)")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER)")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT)")

            # Populate the tables (problem 2)
            majorInfo = [(1, "Math"), (2, "Science"), (3, "Writing"), (4, "Art")]
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", majorInfo)

            courseInfo = [(1, "Calculus"), (2, "English"), (3, "Pottery"), (4, "History")]
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", courseInfo)

            file = open(student_info)
            lines = file.read().strip().split('\n')
            studentInfo = [tuple(l.split(',')) for l in lines]

            file.close()
            cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", studentInfo)

            # Problem 4, replace -1's with NULL values
            cur.execute("UPDATE StudentInfo SET MajorID=NULL WHERE MajorID=-1")

            file = open(student_grades)
            lines = file.read().strip().split('\n')
            studentGrades = [tuple(l.split(',')) for l in lines]
            file.close()
            cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", studentGrades)
    finally:
        conn.close()

# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    file = open(data_file)
    rows = file.read().strip().split('\n')
    rows = [[float(num) for num in r.split(',')] for r in rows]
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            # Drop the tables if they exist
            cur.execute("DROP TABLE IF EXISTS USEarthquakes")

            # Create the following tables
            cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER,"
                                                    "Hour INTEGER, Minute INTEGER, Second INTEGER,"
                                                    "Latitude REAL, Longitude REAL, Magnitude REAL)")
            # Populate the tables
            cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);", rows)

            # Update the tables (problem 4)
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude=0")
            cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day=0")
            cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour=0")
            cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute=0")
            cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second=0")
    finally:
        conn.close()



# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()

    # Get student with A and A+ and make sure it matches up with Course and Student ID's
    cur.execute("SELECT SI.StudentName, CI.CourseName "
                "FROM StudentGrades as SG, StudentInfo as SI, CourseInfo as CI "
                "WHERE (SG.Grade == 'A' OR SG.Grade == 'A+') AND SI.StudentID == SG.StudentID AND CI.CourseID == SG.CourseID")
    students = cur.fetchall()

    conn.close()
    return students

    # Problem 6

def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()

    # Query the info
    cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year BETWEEN 1800 and 1899")
    mags1800 = cur.fetchall()

    cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year BETWEEN 1900 and 1999")
    mags1900 = cur.fetchall()

    cur.execute("SELECT AVG(Magnitude) FROM USEarthquakes")
    avg = cur.fetchall()

    conn.close()
    # Plot the info
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(np.ravel(mags1800), bins=10)
    ax2.hist(np.ravel(mags1900), bins=10)
    ax1.set_title("EQ's in 19th Century")
    ax1.set(xlabel="Magnitude")
    ax1.set_xticks(list(range(1, 11)))
    ax2.set_title("EQ's in 20th Century")
    ax2.set(xlabel="Magnitude")
    ax2.set_xticks(list(range(1, 11)))
    plt.show()
    return avg[0][0]




# Tests
def prob1Test():
    student_db()
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM StudentInfo;")
        print([d[0] for d in cur.description])

def prob2Test():
    student_db()
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM MajorInfo;"):
            print(row)
        for row in cur.execute("SELECT * FROM CourseInfo;"):
            print(row)
        for row in cur.execute("SELECT * FROM StudentInfo;"):
            print(row)
        for row in cur.execute("SELECT * FROM StudentGrades;"):
            print(row)

def prob3Test():
    earthquakes_db()
    with sql.connect("earthquakes.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM USEarthquakes;"):
            print(row)

def prob4Test():
    student_db()
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM StudentInfo;"):
            print(row)
    earthquakes_db()
    with sql.connect("earthquakes.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM USEarthquakes;"):
            print(row)

def prob5Test():
    student_db()
    print(prob5())

def prob6Test():
    print(prob6())

# prob1Test()
# prob2Test()
# prob3Test()
# prob4Test()
# prob5Test()
# prob6Test()
