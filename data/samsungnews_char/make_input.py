import os
import random

def get_random_sample(lst, m):
    """리스트에서 m개의 요소를 랜덤으로 선택하여 반환"""
    if m > len(lst):
        raise ValueError("선택할 개수(m)가 리스트 크기(n)보다 클 수 없습니다.")
    return random.sample(lst, m)




def list_files(directory):
    """주어진 디렉토리 내의 파일 목록을 반환합니다."""
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return files
    except FileNotFoundError:
        print(f"디렉토리 '{directory}'를 찾을 수 없습니다.")
        return []
    except PermissionError:
        print(f"디렉토리 '{directory}'에 대한 권한이 없습니다.")
        return []

def read_text_file(file_path):
    """UTF-8 인코딩으로 텍스트 파일을 읽어 반환합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        print("UTF-8로 읽을 수 없습니다. 다른 인코딩을 시도해 보세요.")
    except FileNotFoundError:
        print(f"파일 '{file_path}'를 찾을 수 없습니다.")
    except PermissionError:
        print(f"파일 '{file_path}'에 대한 접근 권한이 없습니다.")
    return None

def save_text(file_path, text):
    """주어진 텍스트를 파일에 저장"""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


def remove_extra_newlines(text):
    """\n\n을 \n으로 변경"""
    return text.replace("\n\n", "\n")



directory_path = "/home/jhkong/git-project/newscraper/DB/samsung-articles/"  # 원하는 디렉토리 경로 설정
file_list = list_files(directory_path)
m= len(file_list) #10
file_list = get_random_sample(file_list, m)

# print(file_list)
text_data = ""
for filename in file_list:
    # filename = "01101001.20250131092006001.txt"
    file_path = os.path.join(directory_path,filename)
    content = read_text_file(file_path)
    if not content:
        print("Fail to read text.")
        continue
    content = remove_extra_newlines(content)
    content += "\n----------\n"
    # print(content)
    text_data += content
    


file_path = "data/samsung_news_char/input.txt"
save_text(file_path, text_data)
print(f"파일이 {file_path}에 저장되었습니다.")

