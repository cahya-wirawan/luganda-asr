from datasets import load_dataset
import re
import argparse


chars_to_ignore = [",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�", "‘", "’", "’",
             "\u0142", "\u0144", "\u014d", "\u2013", "\u2014", "\u201c", "\u201d"]
chars_to_ignore_regex = f'[{"".join(chars_to_ignore)}]'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="The name of dataset")
    parser.add_argument("-c", "--config_name", type=str, required=True,
                        help="The config name of the dataset")
    parser.add_argument("-d", "--data_dir", type=str, required=False, default=None,
                        help="The directory contains the dataset")
    parser.add_argument("-s", "--split", type=str, required=False, default="train+validation+other+test",
                        help="List of the dataset split names")
    parser.add_argument("-o", "--output_file", type=str, required=True,
                        help="The file name of the prediction result")

    args = parser.parse_args()
    cv = load_dataset(args.name, args.config_name, data_dir=args.data_dir, split=args.split)
    with open(args.output_file, "w") as f:
        for i in range(len(cv)):
            text = cv[i]['sentence'].lower()
            text = text.replace('è', 'é').replace("\u00e1", 'a').replace("\u00e9", 'e').replace("-", ' ')
            text = re.sub("[-,]", " ", text)
            text = re.sub(r" +", " ", text)
            text = re.sub(chars_to_ignore_regex, "", text).strip()
            f.write(f"{text}\n")


if __name__ == "__main__":
    main()
