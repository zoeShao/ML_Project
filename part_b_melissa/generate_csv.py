import os
import csv
import json


def merge_data():
    # A helper function to load the csv file.
    train_path = "../data/train_data.csv"
    question_path = "../data/question_meta.csv"
    student_path = "../data/student_meta.csv"
    if not os.path.exists(train_path):
        raise Exception("The specified path {} does not exist.".format(train_path))
    # Initialize the data.
    data = []
    question_dic = {}
    with open(question_path, "r") as csv_file:
        question = csv.reader(csv_file)
        for row in question:
            if row[0] != "question_id":
                question_dic[row[0]] = row[1]

    student_premium_pupil_dic = {}
    student_gender_dic = {}
    with open(student_path, "r") as csv_file:
        students = csv.reader(csv_file)
        for row in students:
            if row[0] != "user_id":
                student_premium_pupil_dic[row[0]] = row[3]
                student_gender_dic[row[0]] = row[1]

    # Iterate over the row to fill in the data.
    with open(train_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                tmp = {}
                # tmp["question_id"] = int(row[0])
                # tmp["user_id"] = int(row[1])
                tmp["is_correct"] = int(row[2])
                # tmp["subject_id"] = question_dic[row[0]]
                tmp["premium_pupil"] = float(student_premium_pupil_dic[row[1]])
                tmp["gender"] = student_gender_dic[row[1]]
                for i in range(388):
                    subject_ids = question_dic[row[0]].strip('][').split(', ')
                    if str(i) in subject_ids:
                        tmp[str(i)] = 1
                    else:
                        tmp[str(i)] = 0
                data.append(tmp)
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


if __name__ == "__main__":
    data = merge_data()
    # print(data)
    csv_columns = []
    for i in list(range(388)):
        csv_columns.append(str(i))
    csv_columns.extend(["gender", "premium_pupil", "is_correct"])
    with open("merged_training_data.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
