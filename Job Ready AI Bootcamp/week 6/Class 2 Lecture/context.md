# সহজ ভাষায় আজকের ক্লাস

এই ক্লাসে আমরা **Unsupervised Learning** শিখব। এখানে training data-তে correct label থাকে না। Model নিজে data-এর ভেতরের hidden group, direction এবং relationship খুঁজে বের করে।

আমরা তিনটি topic cover করব:

1. **K-Means Clustering** — similar customer-দের group করা
2. **Principal Component Analysis (PCA)** — অনেক feature-কে কম dimension-এ summarize করা
3. **Market Basket Analysis** — কোন product-গুলো একসাথে কেনা হয় তা বের করা

## Unsupervised Learning কেন দরকার?

একটি shop জানে customer কতবার এসেছে, কত টাকা spend করেছে এবং কী product কিনেছে। কিন্তু প্রতিটি customer-এর পাশে `Budget Shopper`, `Premium Buyer` বা `Loyal Customer` label লেখা থাকে না। Manually label করা expensive এবং subjective। Unsupervised Learning answer key ছাড়াই useful pattern-এর candidate খুঁজে দেয়।

তবে এখানে ground-truth accuracy থাকে না। Mathematical metric-এর পাশাপাশি result business-এর জন্য meaningful কি না, সেটিও human expert-কে check করতে হয়।

## K-Means — Customer-দের Meeting Point

একটি মাঠে customer-রা তাদের behavior অনুযায়ী দাঁড়িয়ে আছে ভাবুন। আমরা `K`-টি flag রাখলাম:

1. প্রতিটি customer nearest flag-এর কাছে যায়।
2. প্রতিটি flag নিজের group-এর মাঝখানে move করে।
3. Assignment এবং movement বারবার repeat হয়।
4. Flag আর না সরলে cluster final হয়।

**কী problem solve করেছে:** `spend > 500`-এর মতো arbitrary manual rule-এর বদলে একাধিক numeric feature একসাথে দেখে natural group খুঁজে দেয়।

**কখন ভালো:** group-গুলো compact, roughly round এবং একই রকম size-এর হলে। Strong outlier, unusual shape বা খুব unequal cluster size হলে result misleading হতে পারে।

**Scaling কেন mandatory:** `total_spend` যদি ০–১০,০০০ range-এ এবং `satisfaction` ১–৫ range-এ থাকে, spend distance-কে পুরো dominate করবে। StandardScaler feature-গুলোকে comparable scale-এ আনে।

## PCA — Best Camera Angle

একটি 3D object-এর 2D shadow কল্পনা করুন। সঠিক angle থেকে shadow নিলে object-এর shape-এর বেশিরভাগ information থাকে। PCA high-dimensional data-এর এমন direction খুঁজে বের করে যেখানে maximum variation দেখা যায়।

- প্রথম best direction → `PC1`
- এর perpendicular পরের best direction → `PC2`
- এভাবে প্রয়োজনমতো আরও component

**কী problem solve করেছে:** অনেক correlated feature model-কে slow করে, noise বাড়ায় এবং visualization কঠিন করে। PCA repeated information-কে কম component-এ compress করে।

**কখন ভালো:** visualization, compression, noise reduction এবং downstream model fast করার জন্য। Trade-off হলো `PC1` original feature-এর মতো সহজে explain করা যায় না এবং কিছু information হারায়।

**Example:** `annual_spend`, `average_order_value` এবং `items_purchased`—তিনটিই buying power-এর signal হতে পারে। PCA এদের shared information একটি component-এ summarize করতে পারে।

## Market Basket Analysis — Shopping Habit খোঁজা

Market Basket Analysis receipt দেখে এমন rule খুঁজে:

```text
{Bread, Butter} → {Jam}
```

এর মানে bread এবং butter কিনলে jam unusually often দেখা যায়; এটি প্রমাণ করে না যে bread jam কেনার কারণ।

তিনটি main metric:

- **Support:** সব transaction-এর কত অংশে complete combination আছে
- **Confidence:** left side থাকলে right side কতবার থাকে
- **Lift:** right side-এর normal popularity-এর তুলনায় rule কতটা stronger

৯০% Confidence শুনতে impressive, কিন্তু যদি এমনিতেই ৯০% customer sugar কেনে, `{Coffee} → {Sugar}` rule নতুন information দেয় না। তখন Lift প্রায় ১ হবে।

**কী problem solve করেছে:** সব possible product combination brute-force check করলে computation explode করে। Apriori frequent না এমন ছোট combination বাদ দিয়ে তার সব বড় combination-ও prune করে।

## তিনটি Method কীভাবে একসাথে কাজ করতে পারে?

```text
Customer Table
   ↓
Scaling
   ↓
PCA (optional compression)
   ↓
K-Means
   ↓
Customer Segments

Transaction Baskets
   ↓
Apriori
   ↓
Association Rules
```

PCA correlated noise কমিয়ে K-Means-কে faster করতে পারে। কিন্তু অতিরিক্ত component বাদ দিলে useful cluster pattern-ও হারাতে পারে। PCA-এর আগে এবং পরে cluster quality compare করা উচিত।

## Label ছাড়া Result Judge করব কীভাবে?

| Method | কী Check করবেন | Warning |
|---|---|---|
| K-Means | Elbow, Silhouette, cluster size, stability, business meaning | `K` বাড়লে Inertia সবসময় কমে |
| PCA | Explained variance, cumulative variance, loadings | High variance সবসময় useful information নয় |
| Basket Rules | Support, Confidence, Lift, transaction count | Association causation প্রমাণ করে না |

> **মনে রাখবেন:** Unsupervised model pattern-এর candidate দেয়; final meaning, segment name এবং business action human/domain expert validate করে।

---

# Part 1: K-Means Clustering

## Topic

**K-Means Clustering** — এটি একটি ইটারেটিভ (Iterative) অ্যালগরিদম যা ডেটাসেটের ভেতরের Variance বা WCSS (Within-Cluster Sum of Squares) মিনিমাইজ করে পুরো ডেটাকে $K$ সংখ্যক ক্লাস্টারে ভাগ করে।

## Why It Is Related (AI Engineering-এ এর গুরুত্ব কী?)

বাস্তব দুনিয়ায় (Real-world) প্রায় ৯০% ডেটাই হলো **Unlabeled**। লাখ লাখ কাস্টমার রেকর্ড বা ইমেজকে মানুষের পক্ষে ম্যানুয়ালি ট্যাগ বা লেবেল করা অসম্ভব। K-Means মেশিনকে স্বয়ংক্রিয়ভাবে ডেটার ভেতরের ন্যাচারাল গ্রুপিং খুঁজে পেতে সাহায্য করে:

* **Customer Segmentation**: কাস্টমারদের পারচেজ বিহেভিয়ার অ্যানালাইসিস করে আলাদা গ্রুপ তৈরি করা (যেমনটা Amazon বা Netflix করে)।
* **Image Compression**: কালার পিক্সেলগুলোকে ক্লাস্টার করে ছবির সাইজ ছোট করা।
* **Anomaly Detection**: মেইন ক্লাস্টার থেকে অনেক দূরে থাকা আউটলায়ার (Outlier) পয়েন্টগুলোকে ফ্রড বা অ্যানোমালি হিসেবে ফ্ল্যাগ করা।
* **Geographic Analysis**: GPS কোঅর্ডিনেট ক্লাস্টার করে ডেলিভারি জোন বা রাইডশেয়ারিং হটস্পট নির্ধারণ করা।

K-Means বিশেষ করে তখন প্রাসঙ্গিক যখন:

* আপনার ডেটায় কোনো **Target Label** বা গ্রাউন্ড ট্রুথ থাকে না।
* আপনার একটি **Fast এবং Scalable** অ্যালগরিদম দরকার যা লাখ লাখ ডেটাপয়েন্ট হ্যান্ডেল করতে পারে।
* আপনার **Interpretable** রেজাল্ট চাই (যেখানে প্রতিটি ক্লাস্টার একটি সেন্ট্রয়েড বা কেন্দ্রবিন্দু দ্বারা ডিফাইনড থাকে)।

## How It Works 

### The Algorithm (Lloyd's Algorithm)

```
Step 1: ডেটাসেটে র‍্যান্ডমলি K সংখ্যক সেন্ট্রয়েড (Centroid) বা কেন্দ্রবিন্দু বসান (অথবা k-means++ ব্যবহার করুন)।
Step 2: প্রতিটি ডেটাপয়েন্টকে তার সবচেয়ে কাছের সেন্ট্রয়েডের ক্লাস্টারে অ্যাসাইন (Assign) করুন।
Step 3: এবার প্রতিটি ক্লাস্টারের ভেতরের পয়েন্টগুলোর গড় (Mean) বের করে সেন্ট্রয়েডটিকে নতুন পজিশনে আপডেট করুন।
Step 4: সেন্ট্রয়েডগুলো নড়াচড়া বন্ধ না করা পর্যন্ত (Convergence) Step 2 এবং Step 3 বারবার চালাতে থাকুন।

```

### The Objective Function

K-Means মূলত **Within-Cluster Sum of Squares (WCSS)**-কে মিনিমাইজ করার চেষ্টা করে:

$$\text{WCSS} = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2$$

এখানে:

* $C_i$ = $i$-তম ক্লাস্টারের সবগুলো পয়েন্টের সেট।
* $\mu_i$ = $i$-তম ক্লাস্টারের সেন্ট্রয়েড (Mean)।
* $\|x - \mu_i\|^2$ = পয়েন্ট $x$ থেকে সেন্ট্রয়েড $\mu_i$-এর স্কোয়ার্ড ইউক্লিডিয়ান দূরত্ব (Squared Euclidean Distance)।

### K-Means++ Initialization

র‍্যান্ডমলি সেন্ট্রয়েড বসালে অনেক সময় মডেল খারাপ লোকাল অপ্টিমাতে (Poor local optima) আটকে যায়। এই সমস্যা দূর করতে **k-means++** ব্যবহার করা হয়:

1. প্রথম সেন্ট্রয়েডটি সম্পূর্ণ র‍্যান্ডমলি সিলেক্ট করা হয়।
2. পরবর্তী সেন্ট্রয়েডগুলো সিলেক্ট করার সময় এমন পয়েন্টকে প্রায়োরিটি দেওয়া হয় যা অলরেডি এক্সিস্টিং সেন্ট্রয়েডগুলো থেকে সবচেয়ে দূরে আছে।
3. এর ফলে সেন্ট্রয়েডগুলো পুরো স্পেসে চমৎকারভাবে ছড়িয়ে পড়ে, যা মডেলকে দ্রুত কনভার্জ হতে সাহায্য করে।

## Details: How It Works & Valid Points for Using It

### Choosing K: The Elbow Method

$K$ (ক্লাস্টার সংখ্যা) একটি হাইপারপ্যারামিটার, তাই এটি আগে থেকে নির্ধারণ করার জন্য আমরা **Elbow Method** ব্যবহার করি:

```
K = 1, 2, 3, ..., 15 এর জন্য WCSS ক্যালকুলেট করে একটি গ্রাফ প্লট করুন।
গ্রাফের যে পয়েন্টে এসে WCSS কমার হার হঠাৎ করে শার্পলি ড্রপ করে (কনুই বা Elbow এর মতো দেখায়), 
সেটিকেই অপ্টিমাল K হিসেবে ধরে নেওয়া হয়।

```

### Silhouette Score

ক্লাস্টারের কোয়ালিটি পরিমাপ করার আরও একটি নিখুঁত গাণিতিক মেট্রিক হলো **Silhouette Score**:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

এখানে:

* $a(i)$ = একই ক্লাস্টারের অন্যান্য পয়েন্ট থেকে পয়েন্ট $i$-এর গড় দূরত্ব (Cohesion)।
* $b(i)$ = সবচেয়ে কাছের অন্য ক্লাস্টারের পয়েন্টগুলো থেকে পয়েন্ট $i$-এর গড় দূরত্ব (Separation)।

**Score Interpretation:**

* **+1**: পয়েন্টটি পারফেক্টলি ক্লাস্টারড (নিজের ক্লাস্টারের কাছে, অন্য ক্লাস্টার থেকে অনেক দূরে)।
* **0**: পয়েন্টটি একদম দুটি ক্লাস্টারের বর্ডারে আছে।
* **-1**: পয়েন্টটি ভুল ক্লাস্টারে অ্যাসাইন হয়েছে।

### Valid Points for Using K-Means

| Strength | Explanation|
| --- | --- |
| **Scalable & Fast** | কম্পিউটেশনাল কমপ্লেক্সিটি লিনিয়ার $O(n \times K \times d \times i)$, তাই লাখ লাখ ডেটায় নিমিষেই চলে। |
| **Simple & Interpretable** | প্রত্যেকটি ক্লাস্টারের একটি নির্দিষ্ট সেন্ট্রয়েড থাকে, যা বিজনেস স্টেকহোল্ডারদের সহজে বোঝানো যায়। |
| **Foundation Step** | অনেক জটিল মডেলে প্রিপ্রসেসিং বা ফিচার ইঞ্জিনিয়ারিং স্টেপ হিসেবে এটি ব্যবহৃত হয়। |

### When to Avoid 

| Weakness  | Explanation|
| --- | --- |
| **Spherical Assumption** | এটি ধরে নেয় ক্লাস্টারগুলো গোল বা স্ফেরিক্যাল আকৃতির। ডেটা যদি আঁকাবাঁকা বা লম্বাটে (Elongated) হয়, তবে K-Means ফেইল করে। |
| **Sensitive to Outliers** | মাত্র একটি আউটলায়ার বা চরম ডেটাপয়েন্ট পুরো সেন্ট্রয়েডকে নিজের দিকে টেনে সরিয়ে দিতে পারে। |
| **Requires K upfront** | লুপ চালিয়ে কনুই খোঁজার আগে আপনি জানতে পারবেন না পারফেক্ট ক্লাস্টার সংখ্যা আসলে কত। |

## you can think it like : পিৎজা ডেলিভারি জোন (Pizza Delivery Zones)

মনে করুন আপনি উত্তরায় একটি পিৎজা শপ খুললেন এবং পুরো উত্তরাকে **৫টি ডেলিভারি জোনে** ভাগ করতে চান যাতে রাইডারদের অহেতুক পুরো শহর চষে বেড়াতে না হয়।

* **Step 1:** আপনি ম্যাপের ওপর চোখ বন্ধ করে ৫টি ডেলিভারি হাব (Hubs) চিহ্নিত করলেন।
* **Step 2:** প্রতিটি বাড়িকে তার সবচেয়ে কাছের হাবের আন্ডারে অ্যাসাইন করে দিলেন।
* **Step 3:** এবার দেখা গেল কোনো জোনে বেশি বাড়ি পড়েছে, কোনোটিতে কম। আপনি হাবগুলোকে ওই জোনের বাড়িগুলোর একদম ভৌগোলিক কেন্দ্রবিন্দুতে (Geographic center) সরিয়ে নিলেন।
* **Step 4:** হাব সরানোর পর কিছু বাড়ির দূরত্ব চেঞ্জ হলো। আপনি আবার নতুন করে বাড়ি অ্যাসাইন করলেন এবং হাব রিকেন্দ্রীকরণ করলেন। কয়েকবার করার পর জোনগুলো ফিক্সড হয়ে গেল।

**এখানে কী ভুল হতে পারে?**

* **নদী বা লেক (Irregular Shape):** সেক্টর ১০ আর সেক্টর ১১-র মাঝে যদি একটা বড় লেক থাকে, K-Means সোজা দূরত্ব মেপে একই হাব দিতে পারে, যদিও বাস্তবে রাইডারকে ঘুরে আসতে হবে।
* **আউটলায়ার (Outlier):** অনেক দূরে হাইওয়ের ওপারে মাত্র একটা আলিশান বাড়ি আছে। সেই একটা বাড়ির চক্করে আপনার একটা ডেলিভারি হাব মেইন শহর থেকে দূরে সরে যাবে।

---

# Part 2: Principal Component Analysis (PCA)

## Topic

**PCA (Principal Component Analysis)** — এটি একটি লিনিয়ার ট্রান্সফর্মেশন টেকনিক যা ডেটার মূল ইনফরমেশন বা ভ্যারিয়েন্স (Variance) সর্বোচ্চ বজায় রেখে হাই-ডাইমেনশনাল ডেটাকে লো-ডাইমেনশনাল স্পেসে (New orthogonal axes) প্রজেক্ট করে।

## Why It Is Related

মডার্ন ডেটাসেটে প্রায়ই শত শত বা হাজার হাজার ফিচার থাকে। এত বেশি ফিচারের কারণে মডেল স্লো হয়ে যায় এবং ওভারফিটিং দেখা দেয় (যাকে Curse of Dimensionality বলে)। AI Engineering-এ PCA-র ভূমিকা অনন্য:

* **Visualization**: ৫০টি ফিচার বা ডাইমেনশনের কাস্টমার ডেটাকে ২D বা ৩D গ্রাফে নামিয়ে এনে চোখের সামনে প্লট করা।
* **Speed Up**: ১০০০টি র ফিচারের বদলে মাত্র ১০টি প্রিন্সিপাল কম্পোনেন্টের ওপর মডেল ট্রেইন করে গতি ১০০ গুণ বাড়ানো।
* **Noise Removal**: যে কম্পোনেন্টগুলোর ভ্যারিয়েন্স একদম কম, সেগুলোকে ড্রপ করে ডেটা থেকে নয়েজ বা ময়লা পরিষ্কার করা।

## How It Works 

### Mathematical Foundation (

**Step 1: Center the Data **
প্রতিটি ফিচারের গড় (Mean) থেকে ডেটাপয়েন্টগুলো বিয়োগ করা হয়, যাতে পুরো ডেটাসেট অরিজিন বা $(0,0)$ পয়েন্টে কেন্দ্রবিন্দু পায়।

**Step 2: Compute Covariance Matrix **


$$\Sigma = \frac{1}{n} X^T X$$


এই ম্যাট্রিক্সটি আমাদের বলে একটি ফিচারের পরিবর্তনের সাথে আরেকটি ফিচারের কী সম্পর্ক (Variance এবং Correlation)।

**Step 3: Eigenvalue Decomposition**
কোভ্যারিয়েন্স ম্যাট্রিক্সের আইগেনভ্যালু ($\lambda$) এবং আইগেনভেক্টর ($v$) বের করা হয়:


$$\Sigma v = \lambda v$$

**Step 4: Sort and Select**
আইগেনভেক্টরগুলোই হলো আমাদের নতুন এক্সিস বা **Principal Components (PCs)**। যে আইগেনভেক্টরের আইগেনভ্যালু ($\lambda$) সবচেয়ে বড়, সেটিই ডেটার সবচেয়ে বেশি ভ্যারিয়েন্স (ইনফরমেশন) ধরে রাখে। আমরা আমাদের প্রয়োজনমতো টপ $k$ সংখ্যক কম্পোনেন্ট বেছে নিই।

### Explained Variance Ratio

একটি নির্দিষ্ট কম্পোনেন্ট পুরো ডেটার কত পার্সেন্ট ইনফরমেশন কভার করছে তা বের করার সূত্র:


$$\text{Explained Variance Ratio } (PC_i) = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

**উদাহরণ:** আপনার কাছে ১০০টি ফিচার আছে। আপনি PCA চালিয়ে মাত্র ৫টি কম্পোনেন্ট নিলেন এবং দেখলেন তাদের কিউমুলেটিভ এক্সপ্লেইনড ভ্যারিয়েন্স ৯৫%। এর মানে মাত্র ৫টি নতুন ফিচার দিয়েই আপনি ডেটার ৯৫% মূল ইনফরমেশন অক্ষুণ্ন রাখতে পেরেছেন!

### Valid Points for Using PCA

* **Removes Multicollinearity:** প্রিন্সিপাল কম্পোনেন্টগুলো একে অপরের সাথে $90^\circ$ অ্যাঙ্গেলে (Orthogonal) থাকে। অর্থাৎ এদের মাঝে কোনো কোরিলেশন থাকে না, যা লিনিয়ার রিগ্রেশনের জন্য পারফেক্ট।
* **Reduces Overfitting:** ডাইমেনশন কমে যাওয়ার কারণে মডেলের কমপ্লেক্সিটি কমে এবং জেনারেলাইজেশন ক্ষমতা বাড়ে।

### When to Avoid (কখন ব্যবহার করবেন না)

* **Linear Limitation Only:** PCA শুধুমাত্র সোজা বা লিনিয়ার প্যাটার্ন ধরতে পারে। ডেটার স্ট্রাকচার যদি জটিল ও বাঁকানো (Non-linear manifold) হয়, তবে **t-SNE** বা **UMAP** ব্যবহার করতে হবে।
* **Loss of Interpretability:** PCA করার পর "PC1" বা "PC2" আসলে কোন রিয়েল ফিচার (যেমন বয়স নাকি বেতন) তা আর আলাদা করা যায় না। এটি সমস্ত ফিচারের একটি লিনিয়ার কম্বিনেশন বা ব্ল্যাক-বক্স হয়ে যায়।
* **Scaling Sensitive:** PCA চালানোর আগে **StandardScaler** করা ১০০% বাধ্যতামূলক। নতুবা যে ফিচারের রেঞ্জ বড় (যেমন স্যালারি), সে পুরো মডেল ডোমিনেট করবে।

## you can think it like : The Shadow Puppet

মনে করুন আপনার সামনে একটি সুন্দর ত্রিমাত্রিক (3D) ভাস্কর্য আছে। আপনি কাউকে সেটি দেখাতে চান, কিন্তু আপনার কাছে শুধু একটি ২D প্রজেক্টর বা টর্চলাইট আছে, যার ছায়া ওয়ালের ওপর পড়বে।

* **PC1 (প্রথম কোণ):** আপনি ভাস্কর্যটিকে এমন এক কোণে ঘুরালেন যাতে দেওয়ালে তার সবচেয়ে বড় এবং স্পষ্ট ছায়া পড়ে (সর্বোচ্চ ভ্যারিয়েন্স)। ছায়া দেখেই যেন বোঝা যায় এটি একটি মানুষের মূর্তি।
* **PC2 (দ্বিতীয় কোণ):** প্রথম কোণের সাপেক্ষে একদম লম্বালম্বি আরেকটি কোণ নিলেন যা মূর্তির গভীরতা বা থিকনেস ফুটিয়ে তোলে।
* **বাকি কোণগুলো বাদ:** মূর্তির আঙুলের নখের নিখুঁত নকশা হয়তো ছায়ায় আসবে না (সেটি হলো Noise বা Minor Variance), যা আপনি বাদ দিয়ে দিলেন।

> **Key Insight:** PCA হলো এমন এক স্মার্ট প্রজেক্টর যা কম ডাইমেনশনেও ডেটার আসল রূপ বা আকৃতি ধরে রাখে।

---

# Part 3: Market Basket Analysis

## Topic

**Market Basket Analysis (MBA)** — এটি একটি অ্যাসোসিয়েশন রুল মাইনিং (Association Rule Mining) টেকনিক, যা **Apriori Algorithm** ব্যবহার করে কাস্টমারের ট্রানজেকশন ডেটা থেকে "যারা আইটেম X কেনে, তারা আইটেম Y-ও কেনে" এর মতো গুরুত্বপূর্ণ প্যাটার্ন খুঁজে বের করে।

## Why It Is Related

সুপারশপ বা ই-কমার্স সাইটের রেকমেন্ডেশন ইঞ্জিন এবং প্রোডাক্ট প্লেসমেন্টের পেছনে এটি অন্যতম প্রধান চালিকাশক্তি:

* **Amazon/Daraz**: "Customers who bought this also bought..." এর মাধ্যমে তাদের রেভিনিউ বহুগুণ বাড়িয়ে নেয়।
* **Supermarket Layout**: চাল এবং ডালকে সুপারশপের দুই প্রান্তে রাখা হয়, যেন কাস্টমার ডাল খুঁজতে গিয়ে পুরো দোকান ঘুরে আরও ৪টি জিনিস কার্টে তোলে।

## How It Works: The Apriori Principle

Apriori অ্যালগরিদম একটি গোল্ডেন রুলের ওপর ভিত্তি করে কাজ করে:

> **"যদি কোনো আইটেমসেট ইনফ্রিকুয়েন্ট (Infrequent/অপ্রচলিত) হয়, তবে তার ভেতরের কোনো কম্বিনেশন বা সুপারসেটও কখনো ফ্রিকুয়েন্ট হতে পারবে না।"**

এর ফলে কম্পিউটারের সার্চ স্পেস অনেক কমে যায় (Aggressive Pruning)। যেমন- যদি {দুধ, কলা, পেঁয়াজ} কম্বিনেশনটি খুবই বিরল হয়, তবে এর সাথে আরও আইটেম যোগ করে {দুধ, কলা, পেঁয়াজ, চিপস} চেক করার কোনো দরকারই নেই।

### Key Metrics 

অ্যাসোসিয়েশন রুলের শক্তি পরিমাপ করার জন্য ৩টি প্রধান মেট্রিক ব্যবহার করা হয়:

| Metric | Formula | Meaning (সহজ বাংলা অর্থ) |
| --- | --- | --- |
| **Support** | $\text{Support}(A \cup B) = \frac{\text{count}(A, B)}{\text{Total Transactions}}$ | পুরো স্টোরের মোট ট্রানজেকশনের কত পার্সেন্টে $A$ এবং $B$ একসাথে বিক্রি হয়েছে। |
| **Confidence** | $\text{Confidence}(A \to B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$ | কাস্টমার যখন আইটেম $A$ অলরেডি কিনেছে, তখন তার $B$ কেনার সম্ভাবনা বা প্রবাবিলিটি কত। |
| **Lift** | $\text{Lift}(A \to B) = \frac{\text{Confidence}(A \to B)}{\text{Support}(B)}$ | কাকতালীয় বা কো-ইন্সিডেন্সের বাইরে গিয়ে $A$ আসার কারণে $B$ কেনার চান্স আসলে কত গুণ বাড়ল। |

**Lift Interpretation:**

* **Lift > 1**: পজিটিভ অ্যাসোসিয়েশন (অর্থাৎ, $A$ কেনার কারণেই কাস্টমার $B$ বেশি কিনছে)।
* **Lift = 1**: ইন্ডিপেন্ডেন্ট (একে অপরের সাথে কোনো সম্পর্ক নেই, যার যার মতো বিক্রি হচ্ছে)।
* **Lift < 1**: নেগেটিভ অ্যাসোসিয়েশন (অর্থাৎ, $A$ কিনলে কাস্টমার সাধারণত $B$ কেনে না)।

## you can think it like : ওয়ান ফ্লু ওভার দ্য ডিনার টেবিল 

মনে করুন একটি রেস্টুরেন্টের ১০০টি ডিনার মেনু অ্যানালাইসিস করে আপনি দুটি রুল পেলেন:

**রুল ১: Steak $\to$ Red Wine**

* Confidence = ৭৫% (অর্থাৎ যারা স্টেক অর্ডার দেয়, তাদের ৭৫% রেড ওয়াইন নেয়)।
* কিন্তু দেখা গেল ওই রেস্টুরেন্টের সব কাস্টমারের এমনিতেই ওয়াইন পানের বেসলাইন রেট ৭০%।
* তাহলে এটার **Lift** হলো: $0.75 / 0.70 = 1.07$ (১-এর খুব কাছাকাছি)। অর্থাৎ এই কম্বিনেশনটা স্পেশাল কিছু না, মানুষ এমনিতেই ওয়াইন বেশি খাচ্ছিল।

**রুল ২: Oysters $\to$ Champagne**

* এই কম্বিনেশনের Support কম (মাত্র ১৫টি মেনুতে আছে)।
* কিন্তু এর Confidence ৮৫%! আর রেস্টুরেন্টে শ্যাম্পেনের নরমাল বেসলাইন সেল মাত্র ২০%।
* এর **Lift** কত? $0.85 / 0.20 = 4.25$!
* **অর্থ:** ঝিনুক বা অয়েস্টার অর্ডার করলে কাস্টমারের শ্যাম্পেন কেনার সম্ভাবনা সাধারণ কাস্টমারের চেয়ে **৪.২৫ গুণ বেড়ে যায়**! এটি একটি অত্যন্ত একশনেবল এবং পাওয়ারফুল বিজনেস রুল।

> **Key Insight:** মার্কেট বাস্কেট অ্যানালাইসিস লজিক আর ইনটুইশনের বাইরের এমন সব হিডেন কানেকশন খুঁজে বের করে যা সাধারণ চোখে ধরা পড়ে না (যেমন বিখ্যাত Diapers $\to$ Beer কম্বিনেশন)।

---

# Part 4: Comparative Analysis 

## Algorithm Comparison Matrix

| Aspect | K-Means | PCA | Market Basket |
| --- | --- | --- | --- |
| **Core Goal** | ডেটাপয়েন্টগুলোকে গ্রুপ করা | ডাইমেনশন/ফিচার কমানো | আইটেমের সহ-অবস্থান খুঁজে বের করা |
| **Input Data** | নিউমেরিক্যাল ম্যাট্রিক্স | নিউমেরিক্যাল ম্যাট্রিক্স | ট্রানজেকশন বা রিসিট লিস্ট |
| **Output** | Cluster Labels (০, ১, ২...) | Principal Components | Association Rules ($A \to B$) |
| **Key Metric** | WCSS / Silhouette | Explained Variance | Support / Confidence / Lift |

## Decision Flowchart

```
START: আপনার ডেটার ধরন এবং গোল কী?
│
├─→ সিমিলার কাস্টমার বা পয়েন্ট গ্রুপিং করতে চান?
│   └─→ ক্লাস্টারগুলো গোল আকৃতির এবং সমান ঘনত্বের? ──→ K-MEANS
│
├─→ ফিচারের সংখ্যা অনেক বেশি এবং মডেল স্লো হচ্ছে?
│   ├─→ ২D/৩D গ্রাফে প্লট করতে চান? ──→ PCA (২-৩টি কম্পোনেন্ট নিন)
│   └─→ লিনিয়ার রিগ্রেশনের কোরিলেশন দূর করতে চান? ──→ PCA
│
└─→ কাস্টমারের ক্রয়ের রিসিট বা বাস্কেট ডেটা আছে?
    └─→ ক্রস-সেলিং প্রোডাক্ট সাজেস্ট করতে চান? ──→ MARKET BASKET ANALYSIS

```

---

# Part 5: Real-World Engineering Pipeline 

প্রোডাকশন লেভেলে একজন AI Engineer হিসেবে আপনি প্রায়ই এই টেকনিকগুলো একসাথে পাইপলাইনে ব্যবহার করবেন।

### PCA + K-Means Pipeline 

ফিচার যখন অনেক বেশি থাকে, তখন সরাসরি K-Means চালালে দূরত্ব পরিমাপ ভুল হয়। তাই বেস্ট প্র্যাক্টিস হলো প্রথমে PCA দিয়ে ডাইমেনশন কমিয়ে নেওয়া, তারপর K-Means চালানো:

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ১. ডেটা স্কেলিং করা (ম্যান্ডেটরি)
X_scaled = StandardScaler().fit_transform(X)

# ২. PCA দিয়ে টপ ডাইমেনশন সিলেক্ট করা
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# ৩. কমানো ডাইমেনশনের ওপর K-Means চালানো
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)

```

---

# Part 6: Student Exercises 

## Exercise 1: K-Means Deep Dive

1. K-Means অ্যালগরিদমে WCSS কখনো কি বাড়তে পারে? এর কনভার্জেন্স গ্যারান্টেড কেন?
2. `scikit-learn`-এ `init='random'` বনাম `init='k-means++'` এর মধ্যে কোনটি দ্রুত রান করবে এবং কেন?
3. যদি আপনার ডেটাসেটে দুটি ক্লাস্টার চাঁদের মতো বাঁকা (`make_moons`) আকৃতির হয়, তবে K-Means কেন ব্যর্থ হবে? চিত্র এঁকে ডিসিশন বাউন্ডারি কল্পনা করুন।

## Exercise 2: PCA Mastery

1. যদি আপনি কোনো ডেটাসেটে MinMaxScaler বা StandardScaler না করে PCA রান করেন, তবে আউটপুটে কী মারাত্মক ভুল হবে?
2. স্ক্রী প্লট (Scree Plot) এবং কিউমুলেティブ এক্সপ্লেইনড ভ্যারিয়েন্স গ্রাফ দেখে কীভাবে বুঝবেন আপনার ঠিক কতটি কম্পোনেন্ট নেওয়া উচিত?
3. PCA-র পর প্রাপ্ত নতুন ফিচারগুলো কেন ইন্টারপ্রেট করা (Interpretability) কঠিন, তা ব্যাখ্যা করুন।

## Exercise 3: Market Basket Analysis

1. ধরুন একটি রুল $\{ \text{Coffee} \} \to \{ \text{Sugar} \}$-এর Confidence ৯০%, কিন্তু এর Lift মাত্র ১.০। এই রুলটি কি বিজনেস ডিসিশন নেওয়ার জন্য কাজের? আপনার উত্তরের সপক্ষে যুক্তি দিন।
2. Apriori অ্যালগরিদমের "Downward Closure Property" কম্পিউটারের মেমোরি এবং প্রসেসিং টাইম কীভাবে বাঁচায়?

---

# Key Takeaways (আজকের ক্লাসের সারসংক্ষেপ)

> ১. **K-Means** ডেটার ভেতরের কোহেশন বা ইন্টারনাল ভ্যারিয়েন্স কমিয়ে ন্যাচারাল ক্লাস্টার খোঁজে। এর জন্য Elbow Method এবং Silhouette Score আমাদের গাইড।

> ২. **PCA** হলো ডেটার স্মার্ট লসি কম্প্রেশন (Lossy Compression)। এটি গুরুত্বপূর্ণ ইনফরমেশন বা ভ্যারিয়েন্স বাঁচিয়ে রেখে নয়েজ ঝেড়ে ফেলে এবং ডাইমেনশন কমায়।

> ৩. **Market Basket Analysis** কাস্টমারের অবচেতন মনের পারচেজ কম্বিনেশনগুলো নম্বর ও মেট্রিক্সের (Support, Confidence, Lift) সাহায্যে নিখুঁতভাবে সামনে নিয়ে আসে।

> ৪. **Unsupervised Learning**-এ কোনো পরম সত্য বা "Ground Truth Accuracy" বলতে কিছু নেই। এখানে ম্যাথমেটিক্যাল মেট্রিক্সের পাশাপাশি বিজনেস ডোমেন নলেজ দিয়ে মডেলের কার্যকারিতা জাজ করতে হয়।
