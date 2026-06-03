import os
import argparse
import numpy as np
from datasets import load_dataset, Audio
import soundfile as sf
import librosa
from tqdm import tqdm

# Limit thread usage to avoid deadlocks
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def decode_audio(a, fallback_sr=16000):
    """Decode dataset audio entry to (np.ndarray, sample_rate).
    Handles dict-like audio, torchcodec AudioDecoder and path/bytes cases.
    """
    # already decoded dict-like
    if isinstance(a, dict) and "array" in a and a["array"] is not None:
        arr = np.asarray(a["array"], dtype=np.float32)
        sr = int(a.get("sampling_rate", fallback_sr))
        return arr, sr

    # torchcodec AudioDecoder / AudioSamples handling
    if hasattr(a, "get_all_samples"):
        try:
            arr = a.get_all_samples()
        except Exception as e:
            raise TypeError(f"get_all_samples() failed: {e}")

        # helper to robustly materialize various AudioSamples/iterables into a 1D numpy array
        def _materialize_samples(x):
            # numpy array
            if isinstance(x, np.ndarray):
                return x.astype(np.float32)
            # objects exposing conversion
            if hasattr(x, "to_numpy"):
                return np.asarray(x.to_numpy(), dtype=np.float32)
            if hasattr(x, "to_array"):
                return np.asarray(x.to_array(), dtype=np.float32)
            if hasattr(x, "array"):
                return np.asarray(getattr(x, "array"), dtype=np.float32)
            if hasattr(x, "samples"):
                return np.asarray(getattr(x, "samples"), dtype=np.float32)
            if hasattr(x, "get_all_samples"):
                return _materialize_samples(x.get_all_samples())
            # iterable of channels or frames
            if isinstance(x, (list, tuple)):
                parts = []
                for e in x:
                    parts.append(_materialize_samples(e))
                # if all parts are 1D and same length -> treat as channels and average to mono
                if all(isinstance(p, np.ndarray) and p.ndim == 1 for p in parts):
                    lengths = [p.shape[0] for p in parts]
                    if len(set(lengths)) == 1:
                        stacked = np.stack(parts, axis=0)  # (channels, samples)
                        mono = np.mean(stacked, axis=0)
                        return mono.astype(np.float32)
                # try concatenating if shapes allow
                try:
                    return np.asarray([float(v) for v in x], dtype=np.float32)
                except Exception:
                    raise TypeError("Unable to convert nested audio samples to numpy array")
            # last resort
            raise TypeError("Unsupported audio sample type for materialization")

        try:
            arr_np = _materialize_samples(arr)
        except Exception as e:
            raise TypeError(f"Unable to convert audio samples to numpy array: {e}")

        md = getattr(a, "metadata", None)
        if isinstance(md, dict):
            sr = int(md.get("sample_rate", md.get("sampling_rate", fallback_sr)))
        else:
            sr = int(getattr(md, "sample_rate", getattr(md, "sampling_rate", fallback_sr)))
        return arr_np, sr

    # path/bytes cases
    if isinstance(a, dict) and ("path" in a or "bytes" in a):
        if "path" in a and a["path"]:
            path = a["path"]
            arr, sr = sf.read(path, dtype="float32")
            return np.asarray(arr, dtype=np.float32), int(sr)
        if "bytes" in a and a["bytes"] is not None:
            from io import BytesIO
            arr, sr = sf.read(BytesIO(a["bytes"]), dtype="float32")
            return np.asarray(arr, dtype=np.float32), int(sr)

    raise TypeError(f"Unknown audio type: {type(a)}")


def resample_and_save(example, out_dir, split, index, target_sr=16000, ext=".wav"):
    try:
        arr, sr = decode_audio(example["audio"], fallback_sr=target_sr)
    except Exception as e:
        # if decode fails, raise to be handled by caller
        raise

    # mono
    if hasattr(arr, 'ndim') and arr.ndim == 2:
        arr = arr.mean(axis=1)

    # resample if needed
    if int(sr) != int(target_sr):
        arr = librosa.resample(arr, orig_sr=int(sr), target_sr=int(target_sr))

    label = str(example.get("label", "unknown"))
    out_dir_label = os.path.join(out_dir, split, label)
    os.makedirs(out_dir_label, exist_ok=True)

    out_path = os.path.join(out_dir_label, f"{index:08d}{ext}")
    sf.write(out_path, arr, samplerate=int(target_sr))


def main(input_dir, output_dir, target_sr, overwrite):
    # Walk the input directory directly (expecting audiofolder layout: split/label/files)
    # This avoids dataset lazy decoders and handles any file format readable by soundfile.
    import glob

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    print(f"Walking input_dir={input_dir}")

    # Determine splits as subdirectories of input_dir
    splits = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not splits:
        # maybe files are directly inside input_dir under label subfolders
        splits = ["."]

    for split in splits:
        split_path = os.path.join(input_dir, split)
        # labels are subdirectories under split
        labels = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        if not labels:
            # treat files directly under split as a single unlabeled folder
            labels = [""]

        total = 0
        for label in labels:
            label_path = os.path.join(split_path, label) if label else split_path
            # find common audio extensions
            pattern = os.path.join(label_path, "**", "*.*")
            files = glob.glob(pattern, recursive=True)
            # filter by audio extensions
            files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus'}]
            total += len(files)

        print(f"Processing split={split} approx_files={total}")
        idx = 0
        for label in labels:
            label_path = os.path.join(split_path, label) if label else split_path
            pattern = os.path.join(label_path, "**", "*.*")
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus'}]
            for fpath in tqdm(files, desc=f"split={split} label={label}"):
                try:
                    arr, sr = sf.read(fpath, dtype='float32')
                    arr = np.asarray(arr, dtype=np.float32)
                    # mono
                    if arr.ndim == 2:
                        arr = arr.mean(axis=1)
                    if int(sr) != int(target_sr):
                        arr = librosa.resample(arr, orig_sr=int(sr), target_sr=int(target_sr))
                    out_label = label if label else 'unknown'
                    out_dir_label = os.path.join(output_dir, split, out_label)
                    os.makedirs(out_dir_label, exist_ok=True)
                    out_path = os.path.join(out_dir_label, f"{idx:08d}.wav")
                    sf.write(out_path, arr, samplerate=int(target_sr))
                except Exception as e:
                    print(f"Skipping file={fpath} due to error: {e}")
                idx += 1

    print("Done resampling.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample audiofolder dataset to 16k mono WAVs.")
    parser.add_argument("--input_dir", default="./data", help="Input audiofolder directory")
    parser.add_argument("--output_dir", default="./data_16k", help="Output directory for resampled data")
    parser.add_argument("--sr", type=int, default=16000, help="Target sampling rate")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files (not implemented)")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.sr, args.overwrite)
