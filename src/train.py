#!/usr/bin/env python3
"""Reproducible 15-class PlantVillage training pipeline for Google Colab/local use."""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

CLASS_ALIASES = {
    "Pepper__bell___Bacterial_spot":"Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy":"Pepper__bell___healthy",
    "Potato___Early_blight":"Potato___Early_blight",
    "Potato___Late_blight":"Potato___Late_blight",
    "Potato___healthy":"Potato___healthy",
    "Tomato_Bacterial_spot":"Tomato__Bacterial_spot","Tomato__Bacterial_spot":"Tomato__Bacterial_spot",
    "Tomato_Early_blight":"Tomato__Early_blight","Tomato__Early_blight":"Tomato__Early_blight",
    "Tomato_Late_blight":"Tomato__Late_blight","Tomato__Late_blight":"Tomato__Late_blight",
    "Tomato_Leaf_Mold":"Tomato__Leaf_Mold","Tomato__Leaf_Mold":"Tomato__Leaf_Mold",
    "Tomato_Septoria_leaf_spot":"Tomato__Septoria_leaf_spot","Tomato__Septoria_leaf_spot":"Tomato__Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite":"Tomato__Spider_mites_Two_spotted_spider_mite",
    "Tomato__Spider_mites_Two_spotted_spider_mite":"Tomato__Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot":"Tomato__Target_Spot","Tomato_Target_Spot":"Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":"Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Tomato_YellowLeaf__Curl_Virus":"Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus":"Tomato__Tomato_mosaic_virus",
    "Tomato_Tomato_mosaic_virus":"Tomato__Tomato_mosaic_virus",
    "Tomato_healthy":"Tomato__healthy","Tomato__healthy":"Tomato__healthy",
}
EXPECTED_CLASSES = [
    "Pepper__bell___Bacterial_spot","Pepper__bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Tomato__Bacterial_spot","Tomato__Early_blight","Tomato__Late_blight",
    "Tomato__Leaf_Mold","Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two_spotted_spider_mite","Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato__Tomato_mosaic_virus","Tomato__healthy",
]
IMAGE_EXTENSIONS = {".jpg",".jpeg",".png",".bmp"}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)

def find_dataset_root(path):
    for candidate in (path, path / "PlantVillage"):
        if candidate.is_dir():
            dirs = [p for p in candidate.iterdir() if p.is_dir()]
            recognized = {CLASS_ALIASES[p.name] for p in dirs if p.name in CLASS_ALIASES}
            if len(recognized) >= 10:
                return candidate
    raise FileNotFoundError(f"PlantVillage class folders not found under {path}. Expected 15 class subfolders.")

def collect_samples(dataset_dir):
    class_dirs = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name in CLASS_ALIASES]
    found = {CLASS_ALIASES[p.name] for p in class_dirs}
    missing = sorted(set(EXPECTED_CLASSES) - found)
    extra = sorted(found - set(EXPECTED_CLASSES))
    if missing or extra or len(found) != len(EXPECTED_CLASSES):
        raise ValueError(f"Dataset must contain exactly 15 expected classes. Missing: {missing}; unexpected: {extra}")
    class_to_index = {name:i for i,name in enumerate(EXPECTED_CLASSES)}
    paths, labels, counts = [], [], {}
    for folder in class_dirs:
        canonical = CLASS_ALIASES[folder.name]
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(files) < 3: raise ValueError(f"Class {folder.name} has fewer than 3 images.")
        counts[canonical] = counts.get(canonical,0) + len(files)
        paths.extend(str(p) for p in files); labels.extend([class_to_index[canonical]] * len(files))
    print(f"Dataset: {dataset_dir}\nTotal images: {len(paths):,}")
    for name in EXPECTED_CLASSES: print(f"  {name}: {counts.get(name,0):,}")
    return np.asarray(paths), np.asarray(labels,dtype=np.int32), counts

def make_dataset(paths, labels, img_size, batch_size, training=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((paths,labels))
    def load_image(path,label):
        image = tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False)
        image.set_shape([None,None,3])
        image = tf.image.resize(image,[img_size,img_size],antialias=True)
        return tf.cast(image,tf.float32)/255.0, label
    if training: ds = ds.shuffle(len(paths),seed=seed,reshuffle_each_iteration=True)
    return ds.map(load_image,num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def build_model(img_size,num_classes):
    inputs = tf.keras.Input(shape=(img_size,img_size,3),name="leaf_image")
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
    ],name="data_augmentation")
    x = augment(inputs)
    # Inputs are [0,1]; MobileNetV2 expects [-1,1].
    x = tf.keras.layers.Rescaling(2.0,offset=-1.0,name="mobilenet_preprocessing")(x)
    base = tf.keras.applications.MobileNetV2(input_shape=(img_size,img_size,3),include_top=False,weights="imagenet")
    base.trainable = False
    x = base(x,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes,activation="softmax",name="predictions")(x)
    return tf.keras.Model(inputs,outputs,name="plant_disease_mobilenetv2"),base

class DriveEpochBackup(tf.keras.callbacks.Callback):
    """Save the latest model to Drive after every epoch; keep best validation model separately."""
    def __init__(self, backup_dir, phase):
        super().__init__()
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.phase = phase
        self.best_val_accuracy = -float("inf")

    def _atomic_save(self, destination):
        destination = Path(destination)
        temporary = destination.with_name(destination.stem + ".tmp.keras")
        self.model.save(temporary)
        os.replace(temporary, destination)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        latest = self.backup_dir / "latest_checkpoint.keras"
        self._atomic_save(latest)
        val_acc = float(logs.get("val_accuracy", -1.0))
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self._atomic_save(self.backup_dir / "best_model.keras")
        state = {
            "phase": self.phase,
            "completed_epoch_in_phase": int(epoch + 1),
            "val_accuracy": val_acc,
            "val_loss": float(logs.get("val_loss", -1.0)),
            "latest_checkpoint": str(latest),
            "note": "Checkpoint saved after epoch. This preserves model weights and optimizer state, but this notebook does not automatically resume an interrupted fit."
        }
        state_path = self.backup_dir / "training_state.json"
        temp_state = state_path.with_suffix(".tmp")
        temp_state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_state, state_path)
        print(f"\n[DRIVE BACKUP] Epoch {epoch + 1}: latest checkpoint saved to {latest}", flush=True)


class EpochProgress(tf.keras.callbacks.Callback):
    def __init__(self, phase_name):
        super().__init__()
        self.phase_name = phase_name
        self.started = None

    def on_train_begin(self, logs=None):
        import time
        self.started = time.time()
        print(f"\n[PROGRESS] {self.phase_name}: bắt đầu huấn luyện.", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        import time
        logs = logs or {}
        elapsed = time.time() - self.started
        print(
            f"[PROGRESS] {self.phase_name} | epoch {epoch + 1} | "
            f"loss={logs.get('loss', float('nan')):.4f} | "
            f"accuracy={logs.get('accuracy', float('nan')):.4f} | "
            f"val_loss={logs.get('val_loss', float('nan')):.4f} | "
            f"val_accuracy={logs.get('val_accuracy', float('nan')):.4f} | "
            f"elapsed={elapsed/60:.1f} phút",
            flush=True
        )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",required=True); p.add_argument("--output-dir",default="training_output")
    p.add_argument("--img-size",type=int,default=128); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--epochs",type=int,default=12); p.add_argument("--fine-tune-epochs",type=int,default=8)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--skip-fine-tuning",action="store_true")
    p.add_argument("--backup-dir",default=None,help="Google Drive folder for per-epoch model backups")
    args=p.parse_args()
    set_seed(args.seed); tf.keras.backend.clear_session()
    print("TensorFlow:",tf.__version__); print("GPU devices:",tf.config.list_physical_devices("GPU"))
    dataset_dir=find_dataset_root(Path(args.data_dir).expanduser())
    paths,labels,counts=collect_samples(dataset_dir)
    # Stratified, reproducible 70/15/15 split.
    train_p,temp_p,train_y,temp_y=train_test_split(paths,labels,test_size=.30,random_state=args.seed,stratify=labels)
    val_p,test_p,val_y,test_y=train_test_split(temp_p,temp_y,test_size=.50,random_state=args.seed,stratify=temp_y)
    print(f"Split sizes — train: {len(train_p)}, validation: {len(val_p)}, test: {len(test_p)}")
    train_ds=make_dataset(train_p,train_y,args.img_size,args.batch_size,True,args.seed)
    val_ds=make_dataset(val_p,val_y,args.img_size,args.batch_size)
    test_ds=make_dataset(test_p,test_y,args.img_size,args.batch_size)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    model_path=out/"best_model.keras"
    model,base=build_model(args.img_size,len(EXPECTED_CLASSES))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    backup_dir = Path(args.backup_dir).expanduser() if args.backup_dir else None
    # Reuse ModelCheckpoint and Drive backup callback across both phases so
    # "best" means best validation accuracy across the entire training run.
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        str(model_path), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
    )
    drive_backup_cb = DriveEpochBackup(backup_dir, "training") if backup_dir else None
    def callbacks_for(phase):
        items = [
            checkpoint_cb,
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True,verbose=1),
            tf.keras.callbacks.CSVLogger(str(out/"training_log.csv"), append=(phase != "head")),
            EpochProgress(phase),
        ]
        if drive_backup_cb:
            drive_backup_cb.phase = phase
            items.append(drive_backup_cb)
        return items
    print("\n=== Phase 1: train classification head ===", flush=True)
    h1=model.fit(train_ds,validation_data=val_ds,epochs=args.epochs,callbacks=callbacks_for("head"),verbose=1)
    h2=None
    if not args.skip_fine_tuning and args.fine_tune_epochs>0:
        print("\n=== Phase 2: fine-tune final MobileNetV2 layers ===", flush=True)
        base.trainable=True
        for layer in base.layers[:-30]: layer.trainable=False
        for layer in base.layers:
            if isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        h2=model.fit(train_ds,validation_data=val_ds,epochs=args.fine_tune_epochs,callbacks=callbacks_for("fine_tune"),verbose=1)
    # Test set is used only after checkpoint selection.
    model=tf.keras.models.load_model(model_path)
    test_loss,test_accuracy=model.evaluate(test_ds,verbose=1)
    probs=model.predict(test_ds,verbose=1); pred=np.argmax(probs,axis=1)
    report_txt=classification_report(test_y,pred,labels=np.arange(len(EXPECTED_CLASSES)),target_names=EXPECTED_CLASSES,zero_division=0)
    report_json=classification_report(test_y,pred,labels=np.arange(len(EXPECTED_CLASSES)),target_names=EXPECTED_CLASSES,output_dict=True,zero_division=0)
    (out/"classification_report.txt").write_text(report_txt,encoding="utf-8")
    (out/"classification_report.json").write_text(json.dumps(report_json,ensure_ascii=False,indent=2),encoding="utf-8")
    cm=confusion_matrix(test_y,pred,labels=np.arange(len(EXPECTED_CLASSES)))
    fig,ax=plt.subplots(figsize=(16,14))
    ConfusionMatrixDisplay(cm,display_labels=EXPECTED_CLASSES).plot(ax=ax,xticks_rotation=90,cmap="Blues",colorbar=False)
    fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=160); plt.close(fig)
    metadata={
        "architecture":"MobileNetV2","pretrained_weights":"ImageNet","input_size":[args.img_size,args.img_size,3],
        "input_range":"[0,1]; model contains Rescaling to [-1,1]","class_names":EXPECTED_CLASSES,
        "class_to_index":{n:i for i,n in enumerate(EXPECTED_CLASSES)},"dataset_counts":counts,
        "split":{"train":len(train_p),"validation":len(val_p),"test":len(test_p)},"seed":args.seed,
        "test_loss":float(test_loss),"test_accuracy":float(test_accuracy),"tensorflow_version":tf.__version__,
    }
    (out/"class_names.json").write_text(json.dumps(EXPECTED_CLASSES,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"training_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    histories=[h1]+([h2] if h2 is not None else [])
    def combine(key): return sum([h.history.get(key,[]) for h in histories],[])
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(combine("accuracy"),label="train"); axes[0].plot(combine("val_accuracy"),label="validation"); axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(combine("loss"),label="train"); axes[1].plot(combine("val_loss"),label="validation"); axes[1].set_title("Loss"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out/"learning_curves.png",dpi=160); plt.close(fig)
    print("\n=== Held-out test results ===")
    print(f"Test accuracy: {test_accuracy:.4f}\nTest loss: {test_loss:.4f}")
    print(f"Artifacts saved to: {out.resolve()}")
    print("Do not report these metrics until this run has completed successfully.")

if __name__=="__main__": main()
