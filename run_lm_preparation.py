from datasets import load_dataset
import re

chars_to_ignore = [",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�", "‘", "’", "’",
             "\u0142", "\u0144", "\u014d", "\u2013", "\u2014", "\u201c", "\u201d"]
chars_to_ignore_regex = f'[{"".join(chars_to_ignore)}]'


def main():
    cv = load_dataset('dataset/common_voice', 'lg', data_dir="/dataset/Luganda/cv-corpus-7.0-2021-07-21",
                      split="train+validation+other+test")
    with open("5gram.txt", "w") as f:
        for i in range(len(cv)):
            text = cv[i]['sentence'].lower()
            text = text.replace('è', 'é').replace("\u00e1", 'a').replace("\u00e9", 'e').replace("-", ' ')
            text = re.sub("[-,]", " ", text)
            text = re.sub(r" +", " ", text)
            text = re.sub(chars_to_ignore_regex, "", text).strip()
            f.write(f"{text}\n")


if __name__ == "__main__":
    main()
