# Class 14 — Tensors, TensorFlow, and Keras
## সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **Tensors, TensorFlow, and Keras**। এখানে technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

### Learning Outcomes

এই lecture শেষে আপনি:

- Tensor-এর rank, shape এবং dtype পড়তে পারবেন।
- `tf.GradientTape` দিয়ে automatic differentiation explain করতে পারবেন।
- Raw tabular data কীভাবে scaled ও one-hot encoded tensor-এ যায় তা trace করতে পারবেন।
- Keras `Sequential` model build, compile, train, evaluate এবং save/load করতে পারবেন।
- Training loss ও validation loss দেখে underfitting/overfitting-এর প্রাথমিক signal ধরতে পারবেন।
- MAE/MSE model যে scaled target-এ শিখেছে নাকি original 0–100 scale-এ—এই পার্থক্যটি বুঝতে পারবেন।

## কেন এই Topic দরকার?

Real-world AI system বানাতে শুধু library function call জানলেই হয় না। Input data কোথা থেকে আসে, algorithm কীভাবে decision নেয়, কোন limitation আছে এবং output কীভাবে validate করতে হয়—এই পুরো flow বোঝা দরকার। এই lecture সেই problem-solving mindset তৈরি করবে।

## শেখার সহজ Workflow

Problem বোঝা → Core concept ও intuition → Step-by-step workflow → Practical example → কখন ব্যবহার করবেন বা করবেন না।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?** এই প্রশ্নগুলোর উত্তর দিতে পারলে topic-টি শুধু মুখস্থ নয়, সত্যি বোঝা হয়েছে।

> **Practical mindset:** Example code run করার আগে expected input এবং output লিখে নিন। Run করার পরে result expectation-এর সাথে compare করুন এবং ভুল হলে কোন pipeline step-এ সমস্যা হয়েছে তা isolate করুন।

---

## 1. Tensors, TensorFlow, and Keras

### 1.1 What is a Tensor?

ডিপ লার্নিংয়ের একদম মূল ডেটা স্ট্রাকচার হলো **Tensor**। এটি মূলত একটি মাল্টি-ডাইমেনশনাল অ্যারে (Multi-dimensional array), যা স্কেলার, ভেক্টর এবং ম্যাট্রিক্সকে যেকোনো সংখ্যক ডাইমেনশনে জেনারেলাইজ করতে পারে।

| Name | Dimensions | Shape Example | Real-World Example  |
| --- | --- | --- | --- |
| **Scalar** | 0D | `()` | একটি সিঙ্গেল তাপমাত্রা রিডিং: `72.5` |
| **Vector** | 1D | `(5,)` | ৫টি তাপমাত্রার লিস্ট: `[72.5, 68.0, 75.3, 71.2, 69.8]` |
| **Matrix** | 2D | `(28, 28)` | একটি গ্রে-স্কেল ইমেজ (Grayscale image): `28×28` পিক্সেল |
| **3D Tensor** | 3D | `(64, 64, 3)` | একটি রঙিন ছবি (Color image): Height × Width × RGB channels |
| **4D Tensor** | 4D | `(32, 64, 64, 3)` | ৩২টি রঙিন ছবির একটি পুরো ব্যাচ (Batch) |
| **5D Tensor** | 5D | `(10, 32, 64, 64, 3)` | একটি ভিডিও: ১০টি ফ্রেম × ব্যাচ সাইজ × Height × Width × Channels |

> **Key Insight:** ডিপ লার্নিংয়ে যা কিছু আছে, সবই টেনসর। আপনার ইনপুট ডেটা, মডেলের ওয়েটস (Weights), গ্রাডিয়েন্টস (Gradients) এবং ফাইনাল প্রেডিকশন — এভরিথিং ইজ আ টেনসর।

### 1.2 TensorFlow — The Engine 

**TensorFlow** হলো গুগলের তৈরি একটি ওপেন-সোর্স ডিপ লার্নিং ফ্রেমওয়ার্ক। এটি আমাদের ৩টি মেইন সুবিধা দেয়:

* **Automatic Differentiation**: ব্যাকপ্রপাগেশনের জন্য এটি নিজে নিজেই গ্রাডিয়েন্ট হিসাব করে ফেলে (`GradientTape` দিয়ে)।
* **GPU/TPU Acceleration**: Compatible hardware, driver এবং runtime configure করা থাকলে TensorFlow heavy tensor operation accelerate করতে পারে। Hardware setup ও workload অনুযায়ী speed-up বদলায়।
* **Eager Execution**: Python-এর মতো কোড লেখার সাথে সাথেই ইনস্ট্যান্ট রেজাল্ট বা আউটপুট দেখা যায় (TF 2.0 থেকে ডিফল্ট)।

### 1.3 Keras — The Interface 

**Keras** হলো একটি হাই-লেভেল এপিআই (API) যা টেনসরফ্লোর ওপর একটি সহজ লেয়ার বা ইন্টারফেস হিসেবে কাজ করে (এখন এটি `tf.keras` নামে বিল্ট-ইন)। এটি নিউরাল নেটওয়ার্ক তৈরির জটিল গাণিতিক কোডকে একদম রিডেবল এবং সহজ স্ক্রিপ্টে রূপান্তর করে।

---

## 2. Why It Is Related (AI Engineering-এ ফ্রেমওয়ার্কের গুরুত্ব)

### 2.1 The Problem with Building from Scratch (NumPy দিয়ে স্ক্র্যাচ থেকে করার প্যারা)

এর আগের ক্লাসে আমরা শুধু NumPy ব্যবহার করে একটি ২-লেয়ার নেটওয়ার্ক **from scratch** বানিয়েছিলাম। সেখানে আমাদের কী কী করতে হয়েছিল?

1. ম্যানুয়ালি ওয়েট ম্যাট্রিক্স ও বায়াস ভেক্টর ইনিশিয়ালাইজ করা।
2. ফরওয়ার্ড পাস (Matrix multiplication + Activation) নিজে কোড করা।
3. ক্যালকুলাস খাটিয়ে ব্যাকপ্রপাগেশনের ইকুয়েশন ডেরিভ এবং ইমপ্লিমেন্ট করা।
4. গ্রাডিয়েন্ট ডিসেন্টের আপডেট রুল নিজে লেখা।

**শেখার জন্য এটি দারুণ, কিন্তু প্র্যাক্টিক্যাল কাজের জন্য একদমই নয়।** রিয়েল-ওয়ার্ল্ড মডেলে যেখানে লাখ লাখ বা কোটি কোটি প্যারামিটার থাকে, সেখানে ম্যানুয়াল কোডিং করা:

* **Error-prone**: ব্যাকপ্রপাগেশনে একটি প্লাস-মাইনাস বা সাইনের ভুল পুরো মডেল নষ্ট করে দেয়।
* **Extremely Slow**: NumPy সরাসরি GPU ব্যবহার করতে পারে না, ফলে মডেল ট্রেইন হতে কয়েক দিন লেগে যেতে পারে।
* **Inflexible**: মডেলের আর্কিটেকচারে মাত্র একটি লেয়ার চেঞ্জ করতে গেলেও পুরো ম্যাথমেটিক্যাল কোড নতুন করে লিখতে হয়।

### 2.2 Frameworks to the Rescue

| Task | From Scratch (NumPy) | With TensorFlow/Keras |
| --- | --- | --- |
| **Forward Pass** | ২০+ লাইনের জটিল ম্যাট্রিক্স ডট প্রোডাক্ট | মাত্র ১ লাইন: `model(X)` বা `model.predict(X)` |
| **Backpropagation** | ম্যানুয়ালি ক্যালকুলাস ডেরিভেশন | অটোমেটিক: `tape.gradient()` |
| **Hardware Scaling** | সাধারণ NumPy workflow CPU-কেন্দ্রিক | Compatible runtime configure থাকলে GPU/TPU acceleration |
| **Layer Changing** | পুরো ম্যাথ রিরাইট করতে হয় | শুধু লেয়ারের নাম চেঞ্জ: `layers.Dense()` থেকে `layers.LSTM()` |
| **Save/Load Model** | কাস্টম জটিল পিকল (Pickle) কোড | মাত্র ১ লাইন: `model.save('my_model.keras')` |

> **Frameworks শুধু কোডিং সহজ করে না — এটি বড় স্কেলে ডিপ লার্নিং মডেল তৈরি ও প্রোডাকশনে ডেপ্লয় করা সম্ভব করে তোলে।**

---

## 3. How It Works (কার্যপ্রণালী)

### 3.1 The TensorFlow/Keras Workflow

```
Step 1: ডেটা রেডি করুন এবং তাকে টেনসরে কনভার্ট করুন (tf.constant বা tf.data.Dataset)।
Step 2: Keras Sequential বা Functional API দিয়ে লেগো ব্লকের মতো একটার পর একটা লেয়ার সাজিয়ে মডেল তৈরি করুন।
Step 3: মডেল কম্পাইল (Compile) করুন — এখানে Optimizer (যেমন Adam) এবং Loss Function (যেমন MSE) সিলেক্ট করুন।
Step 4: model.fit() রান করে অটোমেটিক ট্রেনিং লুপ চালু করুন (Forward pass → Loss → Backprop → Update)।
Step 5: নতুন আনসিন ডেটার ওপর model.predict() চালিয়ে প্রেডিকশন নিন।
Step 6: `.keras` format-এ model save করুন, reload করে prediction parity check করুন, তারপর deployment target অনুযায়ী export করুন।

```

### 3.2 Tensor Operations in Action 

```python
import tensorflow as tf

# ── টেনসর তৈরি করা ──
matrix_a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
matrix_b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# ১. এলিমেন্ট-ওয়াইজ যোগ (Element-wise Addition)
print(matrix_a + matrix_b)  # Output: [[6, 8], [10, 12]]

# ২. ম্যাট্রিক্স মাল্টিপ্লিকেশন (ডট প্রোডাক্ট - Dense Layer-এর প্রাণ)
print(tf.matmul(matrix_a, matrix_b))  # Output: [[19, 22], [43, 50]]

# ৩. রিシェেপিং (Reshaping)
flattened = tf.reshape(matrix_a, [4])  # Output: [1, 2, 3, 4]

# ── অটোমেটিক গ্রাডিয়েন্ট (The Heart of Backprop) ──
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2  # y = x²

# dy/dx = 2x = 2 * 3.0 = 6.0
dy_dx = tape.gradient(y, x)
print(dy_dx)  # Output: tf.Tensor(6.0, shape=(), dtype=float32)

```

### 3.3 Building a Sequential Model

Keras-এর **Sequential API** দিয়ে আমরা খুব সহজে একটির পর আরেকটি লেয়ার স্ট্যাক (Stack) করতে পারি:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # Explicit input shape দিলে model summary ও shape checking পরিষ্কার থাকে
    layers.Input(shape=(10,)),

    # প্রথম হিডেন লেয়ার (১২৮টি নিউরন, ReLU অ্যাক্টিভেশন)
    layers.Dense(128, activation='relu'),
    
    # ওভারফিটিং কমানোর জন্য ড্রপআউট লেয়ার
    layers.Dropout(0.3),
    
    # দ্বিতীয় হিডেন লেয়ার
    layers.Dense(64, activation='relu'),
    
    # ট্রেনিংয়ের স্ট্যাবিলিটির জন্য ব্যাচ নরমালইজেশন
    layers.BatchNormalization(),
    
    # আউটপুট লেয়ার (রিগ্রেশনের জন্য ১টি নিউরন এবং Linear অ্যাক্টিভেশন)
    layers.Dense(1, activation='linear')
], name="IndustrialSuccessPredictor")

# মডেল কম্পাইল করা
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',              # Mean Squared Error
    metrics=['mae']          # Mean Absolute Error ট্র্যাকিংয়ের জন্য
)

model.summary()

```

---

## 4. Details: Loss Functions & Optimizers

### 4.1 Loss Functions (কোনটি কখন ব্যবহার করবেন?)

লস ফাংশন আমাদের বলে মডেলের প্রেডিকশন আসল উত্তরের চেয়ে কতটা দূরে আছে। টাস্ক অনুযায়ী সঠিক লস ফাংশন বেছে নেওয়া জরুরি:

| Loss Function | Use Case | When to Use (কখন ব্যবহার করবেন) |
| --- | --- | --- |
| **MSE** (Mean Squared Error) | Regression | কন্টিনিউয়াস টার্গেট (যেমন দাম বা স্কোর প্রেডিকশন)। এটি বড় ভুলকে বেশি পেনাল্টি দেয়। |
| **MAE** (Mean Absolute Error) | Regression | ডেটাসেটে প্রচুর আউটলায়ার (Outliers) থাকলে এটি MSE-র চেয়ে ভালো কাজ করে। |
| **Binary Crossentropy** | Binary Classification | আউটপুট যখন হ্যাঁ/না বা ০/১ হয় (যেমন মেইলটি স্প্যাম নাকি স্প্যাম না)। |
| **Categorical Crossentropy** | Multi-class Classification | আউটপুট যখন একাধিক ক্লাসের একটি হয় (যেমন ছবি দেখে বিড়াল, কুকুর নাকি পাখি তা চেনা) এবং লেবেলগুলো One-hot encoded থাকে। |

### 4.2 The Optimizer: Adam (কীভাবে কাজ করে?)

**Adam (Adaptive Moment Estimation)** একটি widely used adaptive optimizer। এটি gradient-এর first moment এবং squared gradient-এর second moment-এর exponential moving average ব্যবহার করে প্রতিটি parameter-এর update scale করে:

* **First moment:** Gradient-এর direction-এর smoothed estimate update-কে less noisy করে।
* **Second moment:** Squared gradient-এর smoothed estimate parameter-wise effective step size adjust করে।

> **Practical default, not a guarantee:** Adam অনেক tabular ও deep-learning experiment-এর ভালো starting point। কিন্তু learning rate, data scaling, batch size এবং validation behavior check না করে কোনো optimizer-কে automatic best ধরা উচিত নয়। SGD বা RMSprop-এর সাথে controlled comparison দরকার হতে পারে।

---

## 5. Related Analogy: রেস্টুরেন্টের কিচেন (A Restaurant Kitchen)

> **ডিপ লার্নিং ফ্রেমওয়ার্কের পুরো প্রসেসটিকে একটি বড় রেস্টুরেন্টের রান্নাঘরের সাথে তুলনা করা যাক:**

| Deep Learning Concept | Kitchen Analogy (সহজ উপমা) |
| --- | --- |
| **Tensor** | **উপাদানের পাত্র (Containers):** একটি সিঙ্গেল মশলার কৌটা (Scalar), মশলার পুরো রেক (Vector), বা ফ্রিজের বিভিন্ন সেলফে সাজানো কাঁচামাল (Matrix/3D Tensor)। |
| **TensorFlow** | **কিচেনের ইনফ্রাস্ট্রাকচার:** রেস্টুরেন্টের গ্যাস লাইন, বিদ্যুৎ, পানি এবং চুলার মেইন কানেকশন যা পুরো রান্নাঘরকে সচল রাখে। |
| **Keras** | **হেড শেফের রেসিপি কার্ড:** অত্যন্ত সহজ ভাষায় লেখা রান্নার নির্দেশনাবলী, যা পড়লেই জুনিয়র রাঁধুনিরা বুঝে ফেলে কী করতে হবে, পেছনের গ্যাস লাইনের জটিল মেকানিজম তাদের জানতে হয় না। |
| **Sequential Model** | **ধাপে ধাপে রান্না:** প্রথমে কাটাকাটি $\to$ তারপর ফ্রাই $\to$ তারপর মিক্সিং $\to$ শেষে প্লেটিং। এক স্টেপের আউটপুট পরের স্টেপের ইনপুট হয়। |
| **Loss Function** | **কাস্টমারের ফিডব্যাক বা রেটিং:** কাস্টমার খাবার খেয়ে কত কম রেটিং দিল (ভুল বা লস)। লক্ষ্য হলো এই ব্যাড রেটিংকে শূন্যে নামিয়ে আনা। |
| **Optimizer (Adam)** | **অভিজ্ঞ হেড শেফ:** যিনি তরকারি একটু মুখে দিয়ে নুন-ঝাল পরখ করেন (Gradient check) এবং অভিজ্ঞতার আলোকে ঠিক যতটুকু দরকার ততটুকুই মশলা বাড়িয়ে বা কমিয়ে পারফেক্ট টেস্টে নিয়ে আসেন। |
| **Dropout** | **সারপ্রাইজ ব্রেক:** প্র্যাকটিস সেশনের সময় হঠাৎ কিচেনের ৩০% স্টাফকে ছুটি দিয়ে দেওয়া, যাতে বাকি স্টাফরা অলস না হয়ে যেকোনো স্টেশন একাই হ্যান্ডেল করতে পারে (Co-adaptation রোধ করা)। |

---

## Achievement: Industrial Equipment Success Predictor

আমরা এখন TensorFlow এবং Keras ব্যবহার করে একটি end-to-end educational project তৈরি করব, যা industrial equipment-এর **"Success Score"** predict করে। Raw dataset-এ ১৩টি predictor আছে—১০টি numeric ও ৩টি categorical। Project-এর fixed catalog অনুযায়ী one-hot encoding-এর পরে model input ২৬টি column হয়। Trainer configuration-এর hard-coded count বিশ্বাস না করে processed data থেকে input dimension infer করে।

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ১. মডেল আর্কিটেকচার ডিফাইন করা
model = keras.Sequential([
    layers.Input(shape=(26,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')  # কন্টিনিউয়াস স্কোরের জন্য Linear আউটপুট
])

# ২. মডেল কম্পাইলেশন
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# ৩. আর্লি স্টপিং এবং লার্নিং রেট শিডিউলার যোগ করা (Callbacks)
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]

# ৪. মডেল ট্রেনিং শুরু করা
print("Training the Industrial Success Predictor Engine...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ── ৫. নতুন ডেটার ওপর প্রেডিকশন নেওয়া ──
predictions = model.predict(X_new)
print("Top 5 Equipment Success Scores:\n", predictions[:5])

```

> **Scale check:** Project target-কে `StandardScaler` দিয়ে scale করে train করে। তাই raw model output সরাসরি 0–100 score নয়। User-facing score দেখানোর আগে saved target scaler দিয়ে `inverse_transform` করতে হবে।

---

## 6. Lecture Lab এবং Project Mapping

| Lecture concept | Lab/Project implementation | কী verify করবেন |
| --- | --- | --- |
| Tensor shape ও dtype | `Class14_TensorFlow_Keras_Lab.ipynb` | Rank, shape, dtype expected কি না |
| Automatic differentiation | `tf.GradientTape` lab cell | $x=3$ হলে $d(x^2)/dx=6$ কি না |
| Synthetic equipment data | `src/data/generator.py` | Same seed-এ reproducible data |
| Scaling + one-hot encoding | `src/data/preprocessor.py` | 13 raw predictors → 26 encoded columns |
| Sequential regression model | `src/models/builder.py` | Input width 26, output shape `(None, 1)` |
| Training callbacks | `src/models/builder.py` | Early stopping, checkpoint, LR reduction |
| Evaluation | `src/models/evaluator.py` | Scaled metric ও original-scale error গুলিয়ে না ফেলা |
| Interactive workflow | `app/main.py` | Preprocessor ও model দুটোই load হচ্ছে কি না |

## 7. Hands-On Sequence

1. Lecture folder থেকে `Class14_TensorFlow_Keras_Lab.ipynb` খুলুন।
2. Tensor ও gradient cells আগে run করুন; expected shape/gradient লিখে তারপর output compare করুন।
3. In-memory synthetic sample preprocess করে raw এবং encoded feature count compare করুন।
4. ছোট model train করে train বনাম validation loss plot পড়ুন।
5. Prediction original scale-এ ফেরত এনে MAE interpret করুন।
6. `.keras` save/load round-trip-এর prediction difference near zero কি না check করুন।
7. এরপর `../Class 2 Project/notebooks/eda.ipynb` দিয়ে data assumptions এবং leakage risk inspect করুন।

## 8. Validation Checklist

- [ ] একই random seed-এ data এবং model experiment reproducible।
- [ ] Train/validation/test split preprocessing ও evaluation flow-এ আলাদা আছে।
- [ ] Model input width transformed array-এর width-এর সাথে মেলে।
- [ ] Regression output layer-এ ১টি unit ও linear activation আছে।
- [ ] Validation loss-এর trend training loss-এর সাথে compare করা হয়েছে।
- [ ] Prediction inverse-transform করে 0–100 scale-এ interpret করা হয়েছে।
- [ ] Saved model reload করার পরে prediction materially বদলায়নি।

### Interview Reflection

নিজের ভাষায় explain করুন: TensorFlow engine, Keras interface, loss function feedback signal এবং optimizer update rule—এই চারটি কীভাবে একটি training loop-এ একসাথে কাজ করে?
