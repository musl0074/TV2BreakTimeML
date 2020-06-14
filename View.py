from pymongo import MongoClient
import pandas as pd

##Vi opretter forbindelsen til vores MongoDB server hvor vi peger på en specifik collection.
cluster = MongoClient("mongodb+srv://lonardi:b!n_Cm_93#tEMpr@testcluster-ruvhe.mongodb.net/<dbname>?retryWrites=true&w=majority")
db = cluster["Testcluster"]
collection = db["NoSQL test"]
def program():
    print("Velkommen til Database management Menu")
    print("tryk 1 for at tilføje data til databasen")
    print("tryk 2 for se alle datapunkter i databasen")
    print("tryk 3 for at ændre i et datapunkt")
    print("tryk 4 slette et datapunkt")
    print("tryk 5 for søge i databasen")
    userInput = input()


    if userInput == "1":
        print("Name")
        name = input()
        print ("age")
        age = input()
        print("Height")
        height = input()

        post = {
            "Name": name,
            "Age": int(age),
            "Height": int(height)
        }
        collection.insert_one(post)
        print("tryk enhver knap for at gå tilbage til menuen")
        input()
        program()
    elif userInput == "2":
        df = pd.DataFrame(list(collection.find()))
        print(df)
        print("tryk enhver knap for at gå tilbage til menuen")
        input()
        program()
    elif userInput == "3":
        print("3")
        print("tryk enhver knap for at gå tilbage til menuen")
        input()
        program()
    elif userInput == "4":
        print("4")
        print("tryk enhver knap for at gå tilbage til menuen")
        input()
        program()
    elif userInput == "5":
        print("indtast søgekriterie eksemple: 'Name', 'Age' etc.")
        search = input()
        print ("indtast søgeord eksempel: 'Daniel', '27' etc.")
        searchWord = input()
        df = pd.DataFrame(list(collection.find({search: searchWord})))
        print(df)
        print("tryk enhver knap for at gå tilbage til menuen")
        input()
        program()
    else:
        print("Ugyldigt input")
        print("tryk enhver knap for at gå tilbage til menuen")
        input()
        program()
program()