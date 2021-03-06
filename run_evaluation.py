import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, set_seed
import argparse
from pyctcdecode import build_ctcdecoder
from multiprocessing import Pool
from tqdm import tqdm
import re
import unidecode

"""
This is the script to run the prediction on test set of Zindi speech dataset.
Usage:
python run_evaluation.y -m <wav2vec2 model_name> -d <Zindi dataset directory> -o <output file name> \
    -b <optional batch size, default=16>
"""


class KenLM:
    def __init__(self, tokenizer, model_name, num_workers=8, beam_width=128):
        self.num_workers = num_workers
        self.beam_width = beam_width
        vocab_dict = tokenizer.get_vocab()
        self.vocabulary = [x[0] for x in sorted(vocab_dict.items(), key=lambda x: x[1], reverse=False)]
        # Workaround for wrong number of vocabularies:
        if tokenizer.name_or_path == "lucio/wav2vec2-large-xlsr-luganda":
            self.vocabulary += ["_", "-"]
            self.vocabulary[1] = ""  # Remove apostrophe
        elif tokenizer.name_or_path == "lucio/wav2vec2-large-xlsr-kinyarwanda":
            self.vocabulary += ["_"]
        else:
            self.vocabulary = self.vocabulary[:-2]
        self.decoder = build_ctcdecoder(self.vocabulary, model_name)

    @staticmethod
    def lm_postprocess(text):
        return ' '.join([x if len(x) > 1 else "" for x in text.split()]).strip()

    def decode(self, logits):
        probs = logits.cpu().numpy()
        # probs = logits.numpy()
        with Pool(self.num_workers) as pool:
            text = self.decoder.decode_batch(pool, probs)
            text = [KenLM.lm_postprocess(x) for x in text]
        return text

chars_to_ignore = [",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�", "‘", "’", "’"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True,
                        help="The wav2vec2 model name")
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="The name of dataset")
    parser.add_argument("-c", "--config_name", type=str, required=True,
                        help="The config name of the dataset")
    parser.add_argument("-d", "--data_dir", type=str, required=False, default=None,
                        help="The directory contains the dataset")
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=16,
                        help="Batch size")
    parser.add_argument("-k", "--kenlm", type=str, required=False, default=False,
                        help="Path to KenLM model")
    parser.add_argument("--num_workers", type=int, required=False, default=8,
                        help="KenLM's number of workers")
    parser.add_argument("-w", "--beam_width", type=int, required=False, default=128,
                        help="KenLM's beam width")
    parser.add_argument("--test_pct", type=int, required=False, default=100,
                        help="Percentage of the test set")
    parser.add_argument("--cpu", required=False, action='store_true',
                        help="Force to use CPU")
    args = parser.parse_args()

    set_seed(42)  # set the random seed to have reproducible result.
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)
    kenlm = None
    if args.kenlm:
        kenlm = KenLM(processor.tokenizer, args.kenlm)

    resampler = torchaudio.transforms.Resample(48_000, 16_000)
    # Preprocessing the datasets.
    # We need to read the audio files as arrays

    chars_to_ignore_regex = f'[{"".join(chars_to_ignore)}]'

    def speech_file_to_array_fn(batch):
        if "audio" in batch:
            #batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
            speech_array = torch.tensor(batch["audio"]["array"])
        else:
            speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = resampler(speech_array).squeeze().numpy()
        return batch

    def remove_special_characters(batch):
        # word-internal apostrophes are marking contractions
        batch["norm_text"] = re.sub(r'[‘’´`]', r"'", batch["sentence"])
        # most other punctuation is ignored
        batch["norm_text"] = re.sub(chars_to_ignore_regex, "", batch["norm_text"]).lower().strip()
        batch["norm_text"] = re.sub(r"(-|' | '|  +)", " ", batch["norm_text"])
        # remove accents from a few characters (from loanwords, not tones)
        batch["norm_text"] = unidecode.unidecode(batch["norm_text"])
        return batch

    lg_test = load_dataset(args.name, args.config_name, data_dir=args.data_dir,
                           split=f"test[:{args.test_pct}%]")
    if args.cpu:
        device = "cpu"
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lg_test = lg_test.map(speech_file_to_array_fn)
    lg_test = lg_test.map(remove_special_characters)
    model = model.to(device)
    wer = load_metric("wer")

    batch_size = args.batch_size
    def evaluate(batch):
        inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits

        if args.kenlm:
            batch["pred_strings"] = kenlm.decode(logits)
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            batch["pred_strings"] = processor.batch_decode(predicted_ids)
        return batch

    result = lg_test.map(evaluate, batched=True, batch_size=batch_size)
    WER = 100 * wer.compute(predictions=result["pred_strings"], references=result["norm_text"])
    print(f"WER: {WER:.2f}%")


if __name__ == "__main__":
    main()
