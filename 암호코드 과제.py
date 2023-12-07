# 알파벳 매핑 딕셔너리
mapping = {
    'a': 'q',
    'b': 'z',
    'c': 'j',
    'd': 'x',
    'e': 'k',
    'f': 'w',
    'g': 'v',
    'h': 'y',
    'i': 'b',
    'j': 'f',
    'k': 'u',
    'l': 'p',
    'm': 'g',
    'n': 'm',
    'o': 'h',
    'p': 'd',
    'q': 'c',
    'r': 'l',
    's': 's',
    't': 'o',
    'u': 'r',
    'v': 't',
    'w': 'n',
    'x': 'a',
    'y': 'i',
    'z': 'e'
}

def decrypt_text_file(input_file, output_file, mapping):
    with open(input_file, "r") as encrypted_file:
        content = encrypted_file.read().lower()

    decrypted_result = [mapping.get(char, char) if char.isalpha() else char for char in content]
    with open(output_file, "w") as decrypted_file:
        decrypted_file.write("".join(decrypted_result))


decrypt_text_file("암호문.txt", "복원.txt", mapping)




















