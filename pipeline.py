# ============================================================
# PHOENIX 2014-T • End-to-End Pipeline
# ============================================================

# ============================================================
# README / HOW TO RUN
# ============================================================

# 1. Set PHOENIX_ROOT path
# 2. Install requirements:
#    pip install tensorflow mediapipe opencv-python pandas numpy
# 3. Enable switches:
#    RUN_BUILD_LANDMARK_DATA = True
#    RUN_TRAIN_LSTM = True
# 4. Run

# ===== Step 0: Imports =====
import time
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

import tensorflow as tf
from tensorflow.keras import layers, models

# ===== Step 1: Schalter / Output-Steuerung =====
RUN_SANITY_DATA_OVERVIEW = True    # zeigt nur: counts + 1 sample
RUN_BUILD_LANDMARK_DATA  = True    # baut X_train/X_dev aus MediaPipe-Features
RUN_TRAIN_LSTM           = True    # trainiert LSTM (Window-Klassifikator)
RUN_TRAIN_CNN_LSTM       = False   # trainiert Hybrid (Conv1D + LSTM)
RUN_RAW_IMAGE_CNN        = False   # Optional Frame-basierte Raw-CNN-Baseline 

SHOW_MODEL_SUMMARY        = False  # True, wenn du model.summary() sehen willst
USE_DISK_CACHE            = False  # speichert/liest fertige Datasets (.npz) -> spart Zeit


# ===== Step 2: Config =====
CFG = {
    "PHOENIX_ROOT": r"C:\Users\goetz\Desktop\Projekt CV\phoenix",
    "FRAMES_SUBDIR": r"features\fullFrame-210x260px",
    "ANN_TRAIN": r"annotations\manual\PHOENIX-2014-T.train.corpus.csv",
    "ANN_DEV":   r"annotations\manual\PHOENIX-2014-T.dev.corpus.csv",
    "ANN_TEST":  r"annotations\manual\PHOENIX-2014-T.test.corpus.csv",

    "STRIDE": 1,
    "WINDOW_SIZE": 60,
    "HOP": 10,

    "TOP_K": 500,

    "EPOCHS": 5,
    "BATCH_SIZE": 32,
    "LR": 1e-3,

    # Limits
    "TRAIN_LIMIT": 600,
    "DEV_LIMIT": 120,
}

ROOT = Path(CFG["PHOENIX_ROOT"])
assert ROOT.exists(), f"PHOENIX_ROOT existiert nicht: {ROOT}"

# Cache-Dateien (optional)
CACHE_DIR = ROOT / "_cache_jupyter"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_LANDMARK = CACHE_DIR / f"landmarks_win{CFG['WINDOW_SIZE']}_hop{CFG['HOP']}_s{CFG['STRIDE']}_top{CFG['TOP_K']}_tr{CFG['TRAIN_LIMIT']}_dv{CFG['DEV_LIMIT']}.npz"

# ===== Step 3: Hilfsfunktionen (Daten lesen / Samples) =====
def list_frame_dirs(root: Path, frames_subdir: str):
    frames_root = root / frames_subdir
    train_dirs = sorted([p for p in (frames_root / "train").iterdir() if p.is_dir()])
    dev_dirs   = sorted([p for p in (frames_root / "dev").iterdir()   if p.is_dir()])
    test_dirs  = sorted([p for p in (frames_root / "test").iterdir()  if p.is_dir()])
    return train_dirs, dev_dirs, test_dirs

def load_corpus_csv(path: Path):
    return pd.read_csv(path, sep="|", dtype=str)

def build_samples(frame_dirs, df_corpus):
    orth_map = dict(zip(df_corpus["name"], df_corpus["orth"]))
    samples = []
    for d in frame_dirs:
        vid = d.name
        if vid in orth_map:
            samples.append({"video_path": str(d), "orth": orth_map[vid]})
    return samples

def parse_glosses(orth: str):
    if orth is None:
        return []
    return orth.strip().split()

# ===== Step 4: Gloss-Vokabular (Text -> Zahl) =====
train_dirs, dev_dirs, test_dirs = list_frame_dirs(ROOT, CFG["FRAMES_SUBDIR"])

df_train = load_corpus_csv(ROOT / CFG["ANN_TRAIN"])
df_dev   = load_corpus_csv(ROOT / CFG["ANN_DEV"])
df_test  = load_corpus_csv(ROOT / CFG["ANN_TEST"])

samples_train = build_samples(train_dirs, df_train)
samples_dev   = build_samples(dev_dirs,   df_dev)
samples_test  = build_samples(test_dirs,  df_test)

# Häufigkeiten nur aus TRAIN
counter = Counter()
for s in samples_train:
    counter.update(parse_glosses(s["orth"]))

most_common = counter.most_common(CFG["TOP_K"])
idx2gloss = [g for g, _ in most_common] + ["UNK"]
gloss2idx = {g: i for i, g in enumerate(idx2gloss)}

if RUN_SANITY_DATA_OVERVIEW:
    print("\n--- Data overview ---")
    print("Videos (frame-folders): train/dev/test =", len(train_dirs), len(dev_dirs), len(test_dirs))
    print("Annotations (rows)    : train/dev/test =", len(df_train), len(df_dev), len(df_test))
    print("Samples (matched)     : train/dev/test =", len(samples_train), len(samples_dev), len(samples_test))
    print("Gloss-Vocab size      :", len(idx2gloss), "(inkl. UNK)")
    print("Top-5 glosses         :", counter.most_common(5))
    print("Example sample        :", samples_train[0])

# ===== Step 5: Weak Alignment (Window -> Gloss) =====
def window_to_gloss_index(window_j, W, G):
    # gi = floor(j * G / W)
    if W <= 0 or G <= 0:
        return None
    gi = int(np.floor(window_j * G / W))
    return min(gi, G - 1)

# ===== Step 6: MediaPipe -> Hand-Landmarks -> Frame-Feature (128) =====
def normalize_single_hand(hand_pts_21x3: np.ndarray):
    # Zentrieren am Handgelenk + Skalieren auf mittlere "Handgröße"
    pts = hand_pts_21x3.astype(np.float32).copy()
    wrist = pts[0]
    pts = pts - wrist
    scale = np.mean(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 1e-6:
        pts = pts / scale
    return pts

def frame_to_feature_vector(hands_dict, fill_value=0.0):
    # Right (63) + Left (63) + MissingFlags (2) = 128
    right = hands_dict.get("Right")
    left  = hands_dict.get("Left")

    right_missing = 1.0 if right is None else 0.0
    left_missing  = 1.0 if left  is None else 0.0

    if right is None:
        right_feat = np.full((21, 3), fill_value, dtype=np.float32)
    else:
        right_feat = normalize_single_hand(right)

    if left is None:
        left_feat = np.full((21, 3), fill_value, dtype=np.float32)
    else:
        left_feat = normalize_single_hand(left)

    return np.concatenate([
        right_feat.reshape(-1),
        left_feat.reshape(-1),
        np.array([right_missing, left_missing], dtype=np.float32)
    ], axis=0).astype(np.float32)

def video_to_feature_sequence(video_dir, hands_model, stride=1):
    # Output: (T, 128)
    video_dir = Path(video_dir)
    frame_paths = sorted(video_dir.glob("images*"))
    if stride > 1:
        frame_paths = frame_paths[::stride]

    feats = []
    for fp in frame_paths:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        res = hands_model.process(rgb)
        hands_dict = {"Left": None, "Right": None}

        if res.multi_hand_landmarks and res.multi_handedness:
            for lm_list, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label  # "Left"/"Right"
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
                hands_dict[label] = pts

        feats.append(frame_to_feature_vector(hands_dict))

    if len(feats) == 0:
        return np.zeros((0, 128), dtype=np.float32)

    return np.stack(feats).astype(np.float32)

# ===== Step 7: Sliding Windows (T x 128 -> W x (win,128)) =====
def make_sliding_windows(seq, window_size=60, hop=10, pad_value=0.0):
    T = seq.shape[0]
    windows = []
    for start in range(0, T, hop):
        end = start + window_size
        chunk = seq[start:end]
        if chunk.shape[0] < window_size:
            pad_len = window_size - chunk.shape[0]
            pad = np.full((pad_len, seq.shape[1]), pad_value, dtype=np.float32)
            chunk = np.vstack([chunk, pad])
        windows.append(chunk)
        if end >= T:
            break
    return windows

def sample_to_windows_and_labels(sample, hands_model, stride, window_size, hop, gloss2idx):
    glosses = parse_glosses(sample["orth"])
    if len(glosses) == 0:
        return [], []

    seq = video_to_feature_sequence(sample["video_path"], hands_model, stride=stride)
    if seq.shape[0] == 0:
        return [], []

    windows = make_sliding_windows(seq, window_size=window_size, hop=hop)
    W = len(windows)
    G = len(glosses)
    if W == 0:
        return [], []

    y = []
    for j in range(W):
        gi = window_to_gloss_index(j, W, G)
        gloss = glosses[gi]
        y.append(gloss2idx.get(gloss, gloss2idx["UNK"]))
    return windows, y

def build_landmark_dataset(samples, hands_model, limit, stride, window_size, hop, gloss2idx):
    X_list, y_list = [], []
    for s in samples[:limit]:
        windows, y = sample_to_windows_and_labels(s, hands_model, stride, window_size, hop, gloss2idx)
        if len(windows) == 0:
            continue
        X_list.extend(windows)
        y_list.extend(y)
    if len(X_list) == 0:
        return np.zeros((0, window_size, 128), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)

# ===== Step 8: Postprocessing (Collapse + optional Smoothing) =====
def collapse_repeats(labels):
    out = []
    for lab in labels:
        if len(out) == 0 or out[-1] != lab:
            out.append(lab)
    return out

def majority_smooth(labels, k=5):
    if k <= 1:
        return labels[:]
    buf = deque(maxlen=k)
    out = []
    last = None
    for lab in labels:
        buf.append(lab)
        maj = Counter(buf).most_common(1)[0][0]
        if maj != last:
            out.append(maj)
            last = maj
    return out

# ===== Step 9: Modelle =====
def build_lstm_baseline(input_dim, num_classes, window_size):
    model = models.Sequential([
        layers.Input(shape=(window_size, input_dim)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(128),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG["LR"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_cnn_lstm(input_dim, num_classes, window_size):
    model = models.Sequential([
        layers.Input(shape=(window_size, input_dim)),
        layers.Masking(mask_value=0.0),

        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        layers.LSTM(128),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG["LR"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ===== Step 10: Landmark-Datasets bauen/lesen =====
X_train = y_train = X_dev = y_dev = None

if RUN_BUILD_LANDMARK_DATA:
    if USE_DISK_CACHE and CACHE_LANDMARK.exists():
        t0 = time.time()
        data = np.load(CACHE_LANDMARK, allow_pickle=True)
        X_train, y_train = data["X_train"], data["y_train"]
        X_dev,   y_dev   = data["X_dev"],   data["y_dev"]
        print(f"\n[Cache] Landmark datasets geladen: {CACHE_LANDMARK.name} ({time.time()-t0:.1f}s)")
    else:
        t0 = time.time()
        # Video-Mode MediaPipe (Tracking) -> stabiler/faster als static_image_mode=True für Sequenzen
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands_model_video:

            X_train, y_train = build_landmark_dataset(
                samples_train, hands_model_video,
                limit=CFG["TRAIN_LIMIT"],
                stride=CFG["STRIDE"],
                window_size=CFG["WINDOW_SIZE"],
                hop=CFG["HOP"],
                gloss2idx=gloss2idx
            )

            X_dev, y_dev = build_landmark_dataset(
                samples_dev, hands_model_video,
                limit=CFG["DEV_LIMIT"],
                stride=CFG["STRIDE"],
                window_size=CFG["WINDOW_SIZE"],
                hop=CFG["HOP"],
                gloss2idx=gloss2idx
            )

        print(f"\n[Build] Landmark datasets gebaut ({time.time()-t0:.1f}s)")
        if USE_DISK_CACHE:
            np.savez_compressed(CACHE_LANDMARK, X_train=X_train, y_train=y_train, X_dev=X_dev, y_dev=y_dev)
            print(f"[Cache] gespeichert: {CACHE_LANDMARK.name}")

    print("X_train:", X_train.shape, "y_train:", y_train.shape, "| classes in y_train:", len(set(y_train.tolist())))
    print("X_dev  :", X_dev.shape,   "y_dev  :", y_dev.shape,   "| classes in y_dev  :", len(set(y_dev.tolist())))

# ===== Step 11: Training + Evaluation (nur wenn Schalter aktiv) =====
results = {}

def train_and_eval(model, name, Xtr, ytr, Xdv, ydv):
    t0 = time.time()
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xdv, ydv),
        epochs=CFG["EPOCHS"],
        batch_size=CFG["BATCH_SIZE"],
        verbose=1
    )
    dev_loss, dev_acc = model.evaluate(Xdv, ydv, verbose=0)
    dt = time.time() - t0
    results[name] = {"dev_loss": float(dev_loss), "dev_acc": float(dev_acc), "train_seconds": float(dt)}
    print(f"\n[{name}] DEV acc = {dev_acc:.4f} | DEV loss = {dev_loss:.4f} | time = {dt:.1f}s")
    return hist

if RUN_TRAIN_LSTM:
    assert X_train is not None, "X_train fehlt (RUN_BUILD_LANDMARK_DATA muss True sein)."
    model_lstm = build_lstm_baseline(input_dim=128, num_classes=len(gloss2idx), window_size=CFG["WINDOW_SIZE"])
    if SHOW_MODEL_SUMMARY:
        model_lstm.summary()
    _ = train_and_eval(model_lstm, "LSTM", X_train, y_train, X_dev, y_dev)

if RUN_TRAIN_CNN_LSTM:
    assert X_train is not None, "X_train fehlt (RUN_BUILD_LANDMARK_DATA muss True sein)."
    model_cnn_lstm = build_cnn_lstm(input_dim=128, num_classes=len(gloss2idx), window_size=CFG["WINDOW_SIZE"])
    if SHOW_MODEL_SUMMARY:
        model_cnn_lstm.summary()
    _ = train_and_eval(model_cnn_lstm, "CNN_LSTM", X_train, y_train, X_dev, y_dev)


# ===== Step 12 (optional): Raw-Image CNN + LSTM (Window-basierte Rohbild-Sequenzen) =====
# pro Zeitfenster NICHT nur 1 Frame, sondern eine SEQUENZ aus window_size Frames (fixe Länge)
# -> TimeDistributed(CNN) extrahiert pro Frame Features
# -> LSTM modelliert die zeitliche Abfolge der Frame-Features
# Labels bleiben identisch (Weak Alignment pro Window -> 1 Gloss pro Window)

if RUN_RAW_IMAGE_CNN:
    IMG_SIZE = (112, 112)
    MAX_WINDOWS_PER_VIDEO = 30
    AUTOTUNE = tf.data.AUTOTUNE

    def get_frame_paths_from_dir(frame_dir):
        return sorted(Path(frame_dir).glob("images*"))

    def build_image_sequence_instances(samples, limit, stride, window_size, hop, gloss2idx, max_windows_per_video=30):
        """
        Baut eine Liste von (paths_seq, y_idx)
        paths_seq: Liste/Array mit genau window_size Frame-Pfaden (Strings)
        """
        X_paths = []
        y_list = []
        skipped_no_frames, skipped_no_gloss = 0, 0

        for s in samples[:limit]:
            glosses = parse_glosses(s["orth"])
            if len(glosses) == 0:
                skipped_no_gloss += 1
                continue

            frame_paths = get_frame_paths_from_dir(s["video_path"])
            if stride > 1:
                frame_paths = frame_paths[::stride]
            T = len(frame_paths)
            if T == 0:
                skipped_no_frames += 1
                continue

            # Windows über Frame-Indizes
            starts = list(range(0, T, hop))
            windows = []
            for st in starts:
                en = st + window_size
                windows.append((st, en))
                if en >= T:
                    break

            W = len(windows)
            G = len(glosses)
            if W == 0:
                continue

            if max_windows_per_video is not None:
                windows = windows[:max_windows_per_video]
                W = len(windows)

            for j, (st, en) in enumerate(windows):
                gi = window_to_gloss_index(j, W, G)
                gloss = glosses[gi]
                y = gloss2idx.get(gloss, gloss2idx["UNK"])

                seq_paths = frame_paths[st:en]  # kann kürzer sein
                if len(seq_paths) == 0:
                    continue

                # Padding auf fixe Länge window_size: mit letztem Frame auffüllen
                if len(seq_paths) < window_size:
                    seq_paths = seq_paths + [seq_paths[-1]] * (window_size - len(seq_paths))

                # als strings speichern
                X_paths.append([str(p) for p in seq_paths[:window_size]])
                y_list.append(int(y))

        X_paths = np.array(X_paths, dtype=str)  # (N, window_size)
        y_arr = np.array(y_list, dtype=np.int64)

        print(f"\n[Raw-CNN-LSTM] instances={len(y_arr)} | skipped_no_gloss={skipped_no_gloss} | skipped_no_frames={skipped_no_frames}")
        return X_paths, y_arr

    def load_and_preprocess_one_image(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def load_sequence(paths_seq, label):
        # paths_seq: (window_size,) tf.string
        frames = tf.map_fn(load_and_preprocess_one_image, paths_seq, fn_output_signature=tf.float32)
        # frames: (window_size, H, W, 3)
        return frames, label

    def make_seq_dataset(X_paths, y, batch_size, training=True):
        ds = tf.data.Dataset.from_tensor_slices((X_paths, y))
        if training:
            ds = ds.shuffle(min(len(y), 5000), reshuffle_each_iteration=True)
        ds = ds.map(load_sequence, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    def build_raw_cnn_lstm(num_classes, window_size):
        # CNN-Backbone pro Frame
        cnn = models.Sequential([
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),  # -> (features,)
        ])

        model = models.Sequential([
            layers.Input(shape=(window_size, IMG_SIZE[0], IMG_SIZE[1], 3)),
            layers.TimeDistributed(cnn),      # -> (window_size, features)
            layers.LSTM(128),
            layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=CFG["LR"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # ---- Instances bauen (Train/Dev) ----
    Xtr_paths, ytr_img = build_image_sequence_instances(
        samples_train, limit=CFG["TRAIN_LIMIT"], stride=CFG["STRIDE"],
        window_size=CFG["WINDOW_SIZE"], hop=CFG["HOP"], gloss2idx=gloss2idx,
        max_windows_per_video=MAX_WINDOWS_PER_VIDEO
    )
    Xdv_paths, ydv_img = build_image_sequence_instances(
        samples_dev, limit=CFG["DEV_LIMIT"], stride=CFG["STRIDE"],
        window_size=CFG["WINDOW_SIZE"], hop=CFG["HOP"], gloss2idx=gloss2idx,
        max_windows_per_video=MAX_WINDOWS_PER_VIDEO
    )

    ds_train_seq = make_seq_dataset(Xtr_paths, ytr_img, batch_size=CFG["BATCH_SIZE"], training=True)
    ds_dev_seq   = make_seq_dataset(Xdv_paths, ydv_img, batch_size=CFG["BATCH_SIZE"], training=False)

    # WICHTIG: repeat + steps_per_epoch, damit Keras NICHT "ran out of data" meldet
    steps_per_epoch = int(np.ceil(len(ytr_img) / CFG["BATCH_SIZE"]))
    val_steps       = int(np.ceil(len(ydv_img) / CFG["BATCH_SIZE"]))

    model_raw_cnn_lstm = build_raw_cnn_lstm(num_classes=len(gloss2idx), window_size=CFG["WINDOW_SIZE"])
    if SHOW_MODEL_SUMMARY:
        model_raw_cnn_lstm.summary()

    t0 = time.time()
    _ = model_raw_cnn_lstm.fit(
        ds_train_seq.repeat(),
        validation_data=ds_dev_seq,
        epochs=CFG["EPOCHS"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=1
    )
    dev_loss_img, dev_acc_img = model_raw_cnn_lstm.evaluate(ds_dev_seq, verbose=0)
    dt = time.time() - t0

    results["RAW_CNN_LSTM"] = {"dev_loss": float(dev_loss_img), "dev_acc": float(dev_acc_img), "train_seconds": float(dt)}
    print(f"\n[RAW_CNN_LSTM] DEV acc = {dev_acc_img:.4f} | DEV loss = {dev_loss_img:.4f} | time = {dt:.1f}s")


