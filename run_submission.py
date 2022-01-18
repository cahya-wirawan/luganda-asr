import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, set_seed
import argparse
from pyctcdecode import build_ctcdecoder
from multiprocessing import Pool
from tqdm import tqdm

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True,
                        help="The wav2vec2 model name")
    parser.add_argument("-d", "--data_dir", type=str, required=True,
                        help="The directory contains the Zindi dataset (Train.csv, Test.csv and validated_dataset)")
    parser.add_argument("-o", "--output_file", type=str, required=True,
                        help="The file name of the prediction result")
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=16,
                        help="Batch size")
    parser.add_argument("-k", "--kenlm", type=str, required=False, default=False,
                        help="Path to KenLM model")
    parser.add_argument("-n", "--num_workers", type=int, required=False, default=8,
                        help="KenLM's number of workers")
    parser.add_argument("-w", "--beam_width", type=int, required=False, default=128,
                        help="KenLM's beam width")
    parser.add_argument("--test_pct", type=int, required=False, default=100,
                        help="Percentage of the test set")
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

    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = resampler(speech_array).squeeze().numpy()
        return batch

    # lg_train = load_dataset('dataset/luganda', 'zindi', data_dir="/mnt/mldata/data/Luganda",
    #                         split="train")
    lg_test = load_dataset('dataset/zindi', 'lg', data_dir=args.data_dir,
                           split=f"test[:{args.test_pct}%]")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lg_test = lg_test.map(speech_file_to_array_fn)
    model = model.to(device)

    batch_size = args.batch_size
    with open(args.output_file, "w") as file:
        file.write("Clip_ID,Target\n")
        for i in tqdm(range(0, len(lg_test), batch_size)):
            inputs = processor(lg_test[i:i+batch_size]["speech"],
                               sampling_rate=16_000, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            with torch.no_grad():
                logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

            if args.kenlm:
                predicted_sentences = kenlm.decode(logits)
            else:
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_sentences = processor.batch_decode(predicted_ids)
            for j in range(batch_size):
                if i+j >= len(lg_test):
                    break
                client_id = lg_test[i+j]["path"].split("/")[-1].split(".")[0]
                file.write(f"{client_id},{predicted_sentences[j]}\n")
    print(f"\nThe prediction result has been saved to {args.output_file}")


if __name__ == "__main__":
    main()
