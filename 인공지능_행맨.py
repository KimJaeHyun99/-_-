import random

def choose_word():
    word_list = ["coffee","computer","date","elephant"]
    return random.choice(word_list)

def main():
    word = choose_word()  # 무작위 단어 선택
    guessed_letters = set(word)  # 추측한 글자를 저장하는 집합
    correct_guesses = set()  # 올바른 추측을 저장하는 집합
    attempts = 6  # 틀린 추측 횟수 제한

    print("행맨 게임을 시작합니다!")

    while attempts > 0:
        display = ''.join([letter if letter in correct_guesses else '_' for letter in word])  # 현재 상태를 보여주는 문자열 생성
        print("단어:", display)  # 현재 상태 출력

        if display == word:
            print("단어를 맞추셨습니다: " + word)
            break  # 모든 글자를 맞혔으므로 게임 종료

        guess = input("한 글자 추측해보세요: ").lower()  # 사용자에게 추측 입력 요청

        if guess in correct_guesses:
            print("이미 추측한 글자입니다.")
            continue  # 이미 추측한 글자는 무시

        if guess in guessed_letters:
            correct_guesses.add(guess)  # 올바른 추측 집합에 글자 추가
        else:
            attempts -= 1  # 남은 시도 횟수를 감소
            print(f"틀린 추측입니다. 남은 시도 횟수: {attempts}")

    if attempts == 0:
        print(f"게임 종료. 정답은: {word} 입니다.")

if __name__ == "__main__":
    main()
