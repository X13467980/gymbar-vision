import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS_HEAD, EPOCHS_FINETUNE, SEED, MODEL_DIR, MODEL_PATH, LABELS_PATH

def make_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names

def build_model(num_classes):
    # 画像の一般的前処理
    base = keras.applications.EfficientNetB0(
        include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet"
    )
    base.trainable = False  # まずはヘッドのみ

    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
    ])

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = aug(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base

def get_class_weights(train_ds, class_names):
    # ラベル配列を収集してクラス不均衡を補正
    y = []
    for _, labels in train_ds.unbatch():
        y.append(int(labels.numpy()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=list(range(len(class_names))),
        y=y
    )
    return {i: float(w) for i, w in enumerate(weights)}

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, class_names = make_datasets()

    # クラス名を保存
    with open(LABELS_PATH, "w") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print("labels:", class_names)

    model, base = build_model(len(class_names))
    class_weights = get_class_weights(train_ds, class_names)

    # ヘッド学習
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights
    )

    # ファインチューニング（下位層は凍結）
    base.trainable = True
    fine_tune_at = max(0, len(base.layers) - 80)  # 最後の方だけ学習
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "checkpoint.keras"),
            monitor="val_accuracy", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    print("Saved model:", MODEL_PATH)

if __name__ == "__main__":
    main()