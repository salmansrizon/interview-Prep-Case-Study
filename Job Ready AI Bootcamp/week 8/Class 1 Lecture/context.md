## 1. Topic: Convolutional Neural Networks (CNNs)

একটি **Convolutional Neural Network (CNN)** হলো এমন এক ধরণের ডিপ লার্নিং আর্কিটেকচার যা মূলত গ্রিড-লাইক ডেটা — বিশেষ করে **ইমেজ বা ছবি** প্রসেস করার জন্য ডিজাইন করা হয়েছে। সাধারণ নিউরাল নেটওয়ার্কের মতো ছবিকে টেনে-হিঁচড়ে ১D ভেক্টরে ফ্ল্যাটেন (Flatten) না করে, CNN ছবির স্পেশাল স্ট্রাকচার বা ত্রিমাত্রিক রূপ ধরে রাখে। এটি করার জন্য মডেলটি ছবির ওপর দিয়ে কিছু লার্নেবল **Filters** স্ক্রল বা স্ক্যান করায়।

**আমরা এই লেকচারে মূলত ৪টি মেইন কম্পোনেন্ট কভার করব:**

* **Convolutional Layers**: লোকাল ফিচার যেমন — এজ (Edges), টেক্সচার (Textures) এবং আকৃতি (Shapes) এক্সট্রাক্ট করে।
* **Max Pooling**: গুরুত্বপূর্ণ ফিচারগুলো অক্ষুণ্ন রেখে ছবির স্পেশাল সাইজ বা ডাইমেনশন কমিয়ে ছোট করে ফেলে।
* **Padding & Stride**: আউটপুট ডাইমেনশন এবং ফিল্টারটি কতটা জায়গা জুড়ে ঘুরবে তা কন্ট্রোল করে।
* **Image Normalization**: স্টেবল ট্রেনিংয়ের জন্য পিক্সেল ভ্যালুগুলোকে একটি নির্দিষ্ট স্কেলে নিয়ে আসে।

**Project Goal:** আমরা একটি MNIST Handwritten Digit Classifier তৈরি করব যা মানুষের হাতের লেখা ০ থেকে ৯ পর্যন্ত ডিজিটগুলো (১০টি ক্লাস) নিখুঁতভাবে চিনতে পারবে।

---

## 2. Why It Is Related (ইমেজ প্রসেসিংয়ে CNN কেন অপরিহার্য?)

### The Problem with Flattening Images (ছবি ফ্ল্যাটেন করার প্যারা)

ঐতিহ্যবাহী ট্র্যাডিশনাল নিউরাল নেটওয়ার্কে (Dense/FC Layers) একটি ২D ইমেজকে জোর করে ১D ভেক্টরে রূপান্তর করতে হয়। এতে ছবির ভেতরের পিক্সেলগুলোর পারস্পরিক দূরত্ব বা স্পেশাল রিলেশনশিপ (Spatial relationships) সম্পূর্ণ ধ্বংস হয়ে যায়।

নিচের টেবিলটি খেয়াল করুন (ধরা যাক হিডেন লেয়ারে ২৫৬টি নিউরন আছে):

| Image Size | Flattened Size (1D Vector) | Dense Layer Weights Calculation | Total Parameters |
| --- | --- | --- | --- |
| 28×28 Gray (MNIST) | 784 | $784 \times 256$ | **~200,704** |
| 64×64 RGB | 12,288 | $12,288 \times 256$ | **~3.1 Million** |
| 224×224 RGB | 150,528 | $150,528 \times 256$ | **~38.5 Million** |

**এখানে ২টি মারাত্মক সমস্যা দেখা দেয়:**

1. **Parameter Explosion**: ছবির সাইজ সামান্য বাড়লেই প্যারামিটার বা ওয়েটের সংখ্যা কোটিতে গিয়ে ঠেকে। এত বিশাল নেটওয়ার্ক ট্রেইন করা অত্যন্ত ব্যয়বহুল এবং ওয়ান-কাইন্ড-অফ অসম্ভব।
2. **Spatial Destruction**: ১ নম্বর পিক্সেল আর ৭৮৪ নম্বর পিক্সেল ছবির দুই প্রান্তে থাকলেও ফ্ল্যাটেন করার পর তারা পাশাপাশি চলে আসে। ফলে মডেল বুঝতে পারে না চোখের পাশে নাক থাকে, নাকি কানের পাশে!

### Why CNNs Solve This

CNN ছবির দুটি বিশেষ গুণের চমৎকার সুবিধা নেয়:

* **Local Connectivity**: একটি নিউরন পুরো ছবির সাথে কানেক্ট না হয়ে ছোট একটি লোকাল অঞ্চলের (যেমন: $3\times3$ পিক্সেল) সাথে কানেক্ট হয়।
* **Weight Sharing**: একটি সিঙ্গেল ফিল্টার পুরো ছবির ওপর দিয়ে স্লাইড করে যায়। অর্থাৎ, ছবির উপরে বামে চোখ খোঁজার ফিল্টারটিই নিচে ডানেও চোখ খুঁজতে পারে। আলাদা কোনো ওয়েট লাগে না।

> **Result:** যেখানে সাধারণ Dense নেটওয়ার্কে লাখ লাখ প্যারামিটার লাগত, সেখানে CNN মাত্র ২ লাখ (200K) প্যারামিটার দিয়েই MNIST ডেটাসেটে **99%+ Accuracy** অর্জন করে ফেলে!

---

## 3. How It Works 

### 3.1 The Big Picture Pipeline

```
Input Image (28×28×1)
    │
    ▼
[Conv2D]  ──→ ছবির এজ, কার্ভ এবং লাইনগুলো ডিটেক্ট করে
    │
    ▼
[ReLU]    ──→ নেগেটিভ ভ্যালুগুলোকে ০ বানিয়ে নন-লিনিয়ারিটি যোগ করে
    │
    ▼
[MaxPool] ──→ সবচেয়ে স্ট্রং সিগন্যালগুলো রেখে ছবির সাইজ অর্ধেক করে ফেলে
    │
    ▼
[Conv2D]  ──→ এবার জটিল আকৃতি, লুপ বা ইন্টারসেকশন ডিটেক্ট করে
    │
    ▼
[MaxPool] ──→ সাইজ আরও একবার ছোট করে
    │
    ▼
[Flatten] ──→ ২D ফিচার ম্যাপকে ১D ভেক্টরে রূপান্তর করে
    │
    ▼
[Dense]   ──→ ক্লাসিফিকেশনের জন্য সব ফিচারকে একসাথে কম্বাইন করে
    │
    ▼
[Softmax] ──→ ফাইনাল ১০টি ক্লাসের প্রবাবিলিটি বা আউটপুট দেয়

```

### 3.2 The Core Operation: Convolution 

একটি ছোট ম্যাট্রিক্স বা **Filter** (যেমন: $3\times3$) পুরো ইমেজের ওপর দিয়ে স্লাইড করে এবং প্রতিটি পজিশনে একটি **Dot Product** হিসাব করে একটি নতুন **Feature Map** তৈরি করে:

$$\text{Output Value} = \sum (\text{Input Region} \times \text{Filter Weights})$$

**পজিশন (0,0)-তে হিসাবটি কেমন হয়?**


$$(1\times1) + (1\times0) + (1\times1) + (0\times0) + (1\times1) + (1\times0) + (0\times1) + (0\times0) + (1\times1) = 4$$

প্রতিটি ফিল্টার ছবির একেকটি নির্দিষ্ট প্যাটার্ন (যেমন সোজা দাগ বা বাঁকা দাগ) চিনতে শেখে। একটি লেয়ারে ৩২টি ফিল্টার থাকলে আমরা ৩২টি ভিন্ন ভিন্ন ফিচার ম্যাপ পাই।

---

## 4. Details: Controlling Dimensions & Inputs

### 4.1 Padding — Controlling the Border 

* **Valid Padding**: কোনো এক্সট্রা বর্ডার দেওয়া হয় না। ফিল্টার বসানোর কারণে প্রতি লেয়ারে ছবির সাইজ $(F-1)$ পরিমাণ ছোট হতে থাকে (যেখানে $F$ হলো ফিল্টার সাইজ)।
* **Same Padding**: ছবির চারপাশ দিয়ে ০ (Zero) দিয়ে একটি বর্ডার দেওয়া হয়, যাতে কনভোলিউশন করার পরও আউটপুট সাইজ ইনপুট সাইজের একদম সমান থাকে।

```
Without Padding (Valid)              With Same Padding (Zeros added)
    ┌───┬───┬───┐                       ┌───┬───┬───┬───┬───┐
    │ 1 │ 2 │ 3 │                       │ 0 │ 0 │ 0 │ 0 │ 0 │
    ├───┼───┼───┤                       ├───┼───┼───┼───┼───┤
    │ 4 │ 5 │ 6 │   3×3 Filter          │ 0 │ 1 │ 2 │ 3 │ 0 │
    ├───┼───┼───┤  ────────────→        ├───┼───┼───┼───┼───┤
    │ 7 │ 8 │ 9 │                       │ 0 │ 4 │ 5 │ 6 │ 0 │
    └───┴───┴───┘                       ├───┼───┼───┼───┼───┤
     Output: 1×1                        │ 0 │ 7 │ 8 │ 9 │ 0 │
                                        ├───┼───┼───┼───┼───┤
                                        │ 0 │ 0 │ 0 │ 0 │ 0 │
                                        └───┴───┴───┴───┴───┘
                                         Output: 3×3 (Same!)

```

### 4.2 Stride & Max Pooling

* **Stride**: ফিল্টারটি একবারে কয় পিক্সেল লাফ দেবে তা নির্ধারণ করে। Stride = 1 হলে ১ পিক্সেল করে সরবে, Stride = 2 হলে ২ পিক্সেল করে লাফ দেবে (যা আউটপুট সাইজকে অর্ধেক করে ফেলে)।
* **Max Pooling**: ফিচার ম্যাপকে নন-ওভারল্যাপিং উইন্ডোতে (যেমন $2\times2$) ভাগ করে এবং প্রতি উইন্ডো থেকে শুধুমাত্র **সর্বোচ্চ (Maximum)** ভ্যালুটি লুফে নেয়।

> **Valid Point for Max Pooling:** এটি নেটওয়ার্কে **Translation Invariance** তৈরি করে। অর্থাৎ, হাতের লেখার '৫' সংখ্যাটি ছবির সামান্য ডানে বা বামে সরলেও ম্যাক্স পুলিং উইন্ডো থেকে সবচেয়ে স্ট্রং ফিচারটিই পিক হয়, ফলে মডেল কনফিউজড হয় না।

### 4.3 Image Normalization (কেন করবেন?)

ছবির রিউ পিক্সেল ভ্যালুগুলো ০ থেকে ২৫৫-এর মধ্যে থাকে। এগুলোকে সরাসরি মডেল দিলে গ্রাডিয়েন্ট এক্সপ্লোড বা লার্নিং আনস্টেবল হতে পারে। তাই আমরা পুরো ডেটাকে ২৫৫ দিয়ে ভাগ করে $[0, 1]$ রেঞ্জে নিয়ে আসি।

```python
# Scale pixels to a 0-1 range
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

```

---

## 5. Related Analogy: গোয়েন্দার আতশ কাচ (The Detective's Magnifying Glass)

> **মনে করুন একজন ক্রাইম সিন ডিটেকটিভ একটি আঙুলের ছাপ (Fingerprint) পরীক্ষা করছেন:**

| CNN Component | Detective Analogy (সহজ উপমা) |
| --- | --- |
| **Input Image** | মূল ফিঙ্গারপ্রিন্ট কার্ড বা আলামত। |
| **Convolutional Filter** | গোয়েন্দার হাতের **আতশ কাচ (Magnifying Glass)** — যা একবারে পুরো ছবি না দেখে ছোট একটি অংশ ফোকাস করে। |
| **Sliding the Filter** | গোয়েন্দা যেভাবে কাচটি পুরো ফিঙ্গারপ্রিন্টের ওপর দিয়ে সিস্টেমেটিকভাবে বাম থেকে ডানে সরায়। |
| **Feature Map** | গোয়েন্দার নোটপ্যাড — যেখানে তিনি দাগিয়ে রাখছেন কোথায় লুপ (Loops) বা স্পেশাল রেখা পাওয়া গেল। |
| **Max Pooling** | চিফ ইন্সপেক্টর শুধু নোটপ্যাডের মেইন ও স্ট্রং এভিডেন্সগুলো দেখছেন, ছোটখাটো নয়েজ বা ময়লা ইগনোর করছেন। |
| **Deep Layers** | জুনিয়র অফিসাররা দেখছেন সাধারণ রেখা $\to$ সিনিয়র অফিসাররা রেখা মিলিয়ে বানাচ্ছেন প্যাটার্ন $\to$ পরিশেষে চিফ ইন্সপেক্টর আসামি সনাক্ত করছেন। |

---

## Achievement: MNIST Digit Classifier Implementation

নিচে টেনসরফ্লো এবং কেরাস ব্যবহার করে হাতের লেখা চেনার একটি কমপ্লিট CNN কোড দেওয়া হলো:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ১. ডেটা লোড এবং রিসেপ ও নরমালইজেশন
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# CNN-এর জন্য ইনপুট শেপ হতে হবে (Batch, Height, Width, Channels)
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# ২. সিএনএন মডেল আর্কিটেকচার তৈরি
model = keras.Sequential([
    # ১ম কনভোলিউশন ব্লক
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # ২য় কনভোলিউশন ব্লক
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # ক্লাসিফিকেশন হেড (Dense Layer)
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax') # ১০টি ক্লাসের প্রবাবিলিটির জন্য Softmax
], name="MNIST_CNN_Classifier")

# ৩. মডেল কম্পাইল
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # ইনপুট ইন্টিজার লেবেল (0-9) হওয়ায় এটি বেস্ট
    metrics=['accuracy']
)

# ৪. মডেল ট্রেনিং
print("Starting CNN Training Loop...")
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# ৫. ইভালুয়েশন
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🚀 Test Accuracy achieved: {test_acc*100:.2f}%")

```

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Filter Count Experiment**: প্রথম `Conv2D`-তে ৩২টির জায়গায় ১৬টি ফিল্টার দিলে মডেলের ট্রেইনিং টাইম এবং অ্যাকুরেসিতে কী তফাত হয় দেখুন।
2. **Padding Check**: কোডের `padding='same'`-কে বদলে `padding='valid'` করে দিন। এবার `model.summary()` দেখে হিসাব করুন ফাইনাল Flatten লেয়ারের ডাইমেনশন কত সংকুচিত হলো।
3. **Loss Function Analysis**: আমরা এখানে `sparse_categorical_crossentropy` ব্যবহার করেছি। যদি আমরা লেবেলগুলোকে One-hot এনকোড করতাম (`to_categorical`), তবে আমাদের লস ফাংশনে কী চেঞ্জ আনতে হতো?
