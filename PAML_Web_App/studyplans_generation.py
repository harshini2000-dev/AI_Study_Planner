import datetime
from PAML_Web_App.webscrape_studytopics import get_wikipedia_subtopics
import os
# def days_left_for_exam():
#     tdy = datetime.date.today()
#     exam_date = datetime.date(2025, 7, 30)

#     time_difference = exam_date - tdy
#     days_between = time_difference.days - 1
#     return days_between
#     # print(f"{days_between}")


def create_study_plan(study_hours, student_data):
    main_topic = str(student_data["topic"]).capitalize()
    subtopics = get_wikipedia_subtopics(main_topic)
    days_left = int(student_data["days_left"]) + 1

    content = ""
    content = main_topic + "\n\n"
    for day_no in range(1, days_left):
        
        content = content + f"Day {day_no} : " + "\n"

        pertopic = subtopics[-1]
        content = content + f"    study - {pertopic} for {study_hours} hrs"  + "\n" + "\n"

        subtopics.pop()

        if not subtopics:
            break

    content += "\n Revise every topic at the end of the day. \n Do well in the exam. All the best !"
    print(content)
    return content

# student_data = {
#     "topic" : "Statistics",
#     "days_left" : 20
# }

def download_sp(content):
    msg = ""
    file_name = "studyplan.txt"
    home_directory = os.path.expanduser("~")
    downloads_directory = os.path.join(home_directory, "Downloads")
    file_path = os.path.join(downloads_directory, file_name)
    
    try:
        with open(file_path, "w") as file:
            file.write(content)

        msg = f"File '{file_name}' successfully created and saved in '{downloads_directory}'."
        print(msg)
        return msg
    except IOError as e:
        print(f"Error writing to file: {e}")
        return "error"


# content = create_study_plan(3, student_data)
# download_sp(content)