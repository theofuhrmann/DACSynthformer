import argparse
import os

from pydub import AudioSegment


def process_audio(input_path, output_dir):
    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1)

    audio = audio.set_frame_rate(44100)

    audio = audio.set_sample_width(2)

    audio = audio.apply_gain(-audio.max_dBFS)

    os.makedirs(output_dir, exist_ok=True)

    output_filename = (
        os.path.splitext(os.path.basename(input_path))[0] + ".wav"
    )
    output_path = os.path.join(output_dir, output_filename)
    audio.export(output_path, format="wav")

    print(f"Processed file saved: {output_path}")


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path):
            try:
                process_audio(input_path, output_folder)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process all audio files in a folder."
    )
    parser.add_argument(
        "input_folder", help="Path to folder containing input audio files."
    )
    parser.add_argument(
        "output_folder", help="Directory to save processed files."
    )
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)

    process_folder(args.input_folder, args.output_folder)
